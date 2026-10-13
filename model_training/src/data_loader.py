import logging
import math

import h5py
import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, Dataset, random_split


# Fixed split seed, independent of env.seed
generator = torch.Generator().manual_seed(42)


class FusionDataset(Dataset):
    """Load and normalize PF-MLP features and geometric labels."""

    def __init__(self, mat_file_path, data_key, feature_columns=None):
        super().__init__()
        self.feature_columns = feature_columns or [
            'distance', 'angle_norm', 'distance', 'angle_norm'
        ]
        if self.feature_columns != ['distance', 'angle_norm', 'distance', 'angle_norm']:
            raise ValueError("PF-MLP requires distance, angle_norm, distance, angle_norm.")
        data = scipy.io.loadmat(mat_file_path)[data_key][0, 0]
        metadata = data['metadata'][0, 0]
        self.normalization_L = metadata['scene_L'][0, 0]
        self.y_scale = self.normalization_L / 2.0
        self.x_center = math.sqrt(3) * self.normalization_L / 3.0
        self.x_scale = self.x_center
        raw_inputs = torch.tensor(data['inputs'], dtype=torch.float32)
        if raw_inputs.shape[1] < 4:
            raise ValueError("PF-MLP requires at least four input columns.")
        self.inputs = raw_inputs[:, :4]
        self.labels = torch.tensor(data['labels'], dtype=torch.float32)
        logging.info("Loaded PF-MLP dataset: %s (%d samples)", mat_file_path, len(self))

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        normalized_input = self.inputs[idx].clone()
        for i, col_type in enumerate(self.feature_columns):
            if col_type == 'distance':
                normalized_input[i] = (
                    normalized_input[i] - 1.5 * self.normalization_L
                ) / (0.5 * self.normalization_L)
            else:
                # Angle [0, 2*pi] -> [-1, 1]
                normalized_input[i] = (normalized_input[i] - 3.14159) / 3.14159
        label = self.labels[idx]
        normalized_label = torch.tensor([
            (label[0] - self.x_center) / self.x_scale,
            label[1] / self.y_scale
        ], dtype=torch.float32)
        return normalized_input, normalized_label


class SignalFusionDataset(Dataset):
    """Read MATLAB v7/v7.3 amplitudes and return (2, 1, Nc) tensors."""

    def __init__(self, mat_file_path, data_key, db_norm_params=None):
        super().__init__()
        if h5py.is_hdf5(mat_file_path):
            with h5py.File(mat_file_path, 'r') as mat_data:
                data = mat_data[data_key]
                raw_inputs = np.array(data['inputs']).T
                raw_labels = np.array(data['labels']).T
                metadata = data['metadata']
                self.normalization_L = float(np.array(metadata['scene_L']).item())
                self.Nc = int(np.array(metadata['Nc']).item())
                self.Ns = int(np.array(metadata['Ns']).item())
        else:
            data = scipy.io.loadmat(mat_file_path)[data_key][0, 0]
            raw_inputs, raw_labels = data['inputs'], data['labels']
            metadata = data['metadata'][0, 0]
            self.normalization_L = float(metadata['scene_L'][0, 0])
            self.Nc = int(metadata['Nc'][0, 0])
            self.Ns = int(metadata['Ns'][0, 0])
        if self.Ns != 1:
            raise ValueError("SF-CNN requires Ns=1.")
        if raw_inputs.shape[1] != 2 * self.Nc * self.Ns:
            raise ValueError("Signal input dimension does not match Nc and Ns.")
        self.inputs = torch.tensor(raw_inputs, dtype=torch.float32)
        self.labels = torch.tensor(raw_labels, dtype=torch.float32)
        self.signal_len = self.Nc * self.Ns
        self.epsilon = 1e-20
        if db_norm_params is None:
            (self.db_min, self.db_max, self.db_offset,
             self.db_half_range) = self._compute_db_normalization_params()
        else:
            self.db_min = db_norm_params['db_min']
            self.db_max = db_norm_params['db_max']
            self.db_offset = db_norm_params['db_offset']
            self.db_half_range = db_norm_params['db_half_range']
        self.y_scale = self.normalization_L / 2.0
        self.x_center = math.sqrt(3) * self.normalization_L / 3.0
        self.x_scale = self.x_center
        logging.info("Loaded SF-CNN dataset: %s (%d samples)", mat_file_path, len(self))
        logging.info("dB offset=%.8f, half range=%.8f", self.db_offset, self.db_half_range)

    def _compute_db_normalization_params(self):
        """dB range of a random subset of samples, widened by a 10% margin."""
        num_samples = len(self)
        sample_count = max(100, min(int(num_samples * 0.1), 5000))
        sample_indices = torch.randperm(num_samples)[:sample_count]
        all_db_values = []
        for idx in sample_indices:
            signal = self.inputs[int(idx)]
            all_db_values.append(20 * torch.log10(signal[:self.signal_len] + self.epsilon))
            all_db_values.append(20 * torch.log10(signal[self.signal_len:] + self.epsilon))
        all_db = torch.cat(all_db_values)
        db_min = torch.min(all_db).item()
        db_max = torch.max(all_db).item()
        db_offset = (db_max + db_min) / 2.0
        db_half_range = ((db_max - db_min) / 2.0) * (1 + 0.1)
        return db_min, db_max, db_offset, db_half_range

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        signal = self.inputs[int(idx)]
        rx1_db = 20 * torch.log10(signal[:self.signal_len] + self.epsilon)
        rx2_db = 20 * torch.log10(signal[self.signal_len:] + self.epsilon)
        rx1 = ((rx1_db - self.db_offset) / self.db_half_range).reshape(self.Ns, self.Nc)
        rx2 = ((rx2_db - self.db_offset) / self.db_half_range).reshape(self.Ns, self.Nc)
        label = self.labels[idx]
        normalized_label = torch.tensor([
            (label[0] - self.x_center) / self.x_scale,
            label[1] / self.y_scale
        ], dtype=torch.float32)
        return torch.stack([rx1, rx2], dim=0), normalized_label


def create_dataloaders(config, batch_size, num_workers=4):
    """Split the full dataset into train, validation, and test subsets."""
    data = config['data']
    if data.get('dataset_type', 'fusion') == 'signal_fusion':
        dataset = SignalFusionDataset(
            data['path'], data['mat_data_key'], data.get('db_norm_params')
        )
    else:
        dataset = FusionDataset(
            data['path'], data['mat_data_key'], data.get('feature_columns')
        )
    val_split, test_split = data.get('val_split', 0.1), data.get('test_split', 0.1)
    if not (0 < val_split < 1 and 0 < test_split < 1 and val_split + test_split < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than one.")
    total = len(dataset)
    val_size, test_size = int(total * val_split), int(total * test_split)
    sizes = [total - val_size - test_size, val_size, test_size]
    subsets = random_split(dataset, sizes, generator=generator)
    loaders = [DataLoader(
        subset, batch_size=batch_size, shuffle=(i == 0),
        num_workers=num_workers, pin_memory=True
    ) for i, subset in enumerate(subsets)]
    logging.info("Train/validation/test samples: %s", sizes)
    norm_params = {
        'L': dataset.normalization_L,
        'y_scale': dataset.normalization_L / 2.0,
        'x_center': math.sqrt(3) * dataset.normalization_L / 3.0,
        'x_scale': math.sqrt(3) * dataset.normalization_L / 3.0,
        'type': 'geometric'
    }
    if isinstance(dataset, SignalFusionDataset):
        norm_params.update({key: getattr(dataset, key) for key in
                            ('db_min', 'db_max', 'db_offset', 'db_half_range')})
    return *loaders, norm_params
