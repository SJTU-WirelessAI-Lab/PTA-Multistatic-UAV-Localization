import copy
import logging
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def set_seed(seed):
    """Seed Python, NumPy and PyTorch, and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir, log_basename):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(os.path.join(log_dir, f'{log_basename}.log')),
                  logging.StreamHandler()]
    )


def to_builtin_types(obj):
    """Convert NumPy values to portable YAML/JSON types."""
    if isinstance(obj, dict):
        return {key: to_builtin_types(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin_types(value) for value in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_norm_params(path):
    """Load normalization parameters saved by training."""
    with open(path) as stream:
        return yaml.safe_load(stream)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_model_size(model, dataloader, device):
    """Profile a copy so THOP cannot alter checkpoint parameter names."""
    logging.info('Trainable parameters: %s', f'{count_parameters(model):,}')
    try:
        import thop
        dummy_input = next(iter(dataloader))[0][0:1].to(device)
        model_clone = copy.deepcopy(model)
        flops, params = thop.profile(model_clone, inputs=(dummy_input,), verbose=False)
        logging.info('THOP FLOPs per sample: %.2f M; parameters: %.2f M',
                     flops / 1e6, params / 1e6)
    except ImportError:
        logging.warning('THOP is not installed; skipping FLOPs profiling.')
    except Exception as error:
        logging.warning('FLOPs profiling failed: %s', error)


def denormalize_positions(positions, norm_params):
    result = positions.clone()
    result[:, 0] = positions[:, 0] * norm_params['x_scale'] + norm_params['x_center']
    result[:, 1] = positions[:, 1] * norm_params['y_scale']
    return result


def calculate_performance_metrics(predictions, targets):
    if predictions.shape != targets.shape:
        raise ValueError('Prediction and target shapes must match.')
    errors = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=1))
    errors = errors.detach().cpu().numpy()
    return {'rmse': np.sqrt(np.mean(errors ** 2)), 'mean': np.mean(errors),
            'max': np.max(errors), 'min': np.min(errors), 'p95': np.percentile(errors, 95)}


def plot_loss_curves(history, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logging.info('Loss curve saved to %s', save_path)


def plot_predictions_scatter(predictions, targets, save_path, limit=500):
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    if len(predictions) > limit:
        indices = np.random.choice(len(predictions), limit, replace=False)
        predictions, targets = predictions[indices], targets[indices]
    plt.figure(figsize=(10, 10))
    plt.scatter(targets[:, 0], targets[:, 1], c='blue', label='True Positions',
                alpha=0.6, s=50, marker='o')
    plt.scatter(predictions[:, 0], predictions[:, 1], c='red', label='Predicted Positions',
                alpha=0.8, s=50, marker='x')
    for prediction, target in zip(predictions, targets):
        plt.plot([target[0], prediction[0]], [target[1], prediction[1]],
                 'gray', linestyle='--', linewidth=0.5)
    plt.title('True vs. Predicted Positions Scatter Plot')
    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(save_path)
    plt.close()
    logging.info('Prediction plot saved to %s', save_path)
