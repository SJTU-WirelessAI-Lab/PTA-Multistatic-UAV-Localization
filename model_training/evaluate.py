import argparse
import logging
import os

import torch
from tqdm import tqdm
import yaml

from src.data_loader import create_dataloaders
from src.models import create_model
from src.utils import (calculate_performance_metrics, denormalize_positions,
                       load_norm_params, log_model_size, plot_predictions_scatter,
                       set_seed, setup_logging)


def get_all_predictions(model, dataloader, device):
    """Collect normalized predictions and labels from the test subset."""
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Running Inference'):
            predictions.append(model(inputs.to(device)).cpu())
            targets.append(labels.cpu())
    return torch.cat(predictions), torch.cat(targets)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained paper model.')
    parser.add_argument('--experiment_dir', required=True, help='Training output directory.')
    args = parser.parse_args()
    with open(os.path.join(args.experiment_dir, 'config.yaml')) as stream:
        config = yaml.safe_load(stream)
    name = config['output']['experiment_name']
    setup_logging(args.experiment_dir, name)
    set_seed(config['env']['seed'])
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    saved_norm = load_norm_params(os.path.join(args.experiment_dir, 'norm_params.yaml'))
    if 'db_offset' in saved_norm:
        config['data']['db_norm_params'] = {
            key: saved_norm[key] for key in ('db_min', 'db_max', 'db_offset', 'db_half_range')
        }
    _, _, test_loader, _ = create_dataloaders(
        config, config['training']['batch_size'], config['training'].get('num_workers', 4)
    )
    model = create_model(config).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.experiment_dir, f'{name}.pth'), map_location=device,
        weights_only=True
    ), strict=True)
    logging.info('Test samples: %d; device: %s', len(test_loader.dataset), device)
    log_model_size(model, test_loader, device)
    predictions, targets = get_all_predictions(model, test_loader, device)
    predictions = denormalize_positions(predictions, saved_norm)
    targets = denormalize_positions(targets, saved_norm)
    metrics = calculate_performance_metrics(predictions, targets)
    for key, value in metrics.items():
        logging.info('Test %s: %.6f m', key, value)
    plot_predictions_scatter(
        predictions, targets, os.path.join(args.experiment_dir, f'{name}_predictions.png')
    )


if __name__ == '__main__':
    main()
