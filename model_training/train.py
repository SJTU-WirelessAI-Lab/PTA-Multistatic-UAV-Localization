import argparse
import ast
import logging
import os

import torch
from torch import nn, optim
import yaml

from src.data_loader import create_dataloaders
from src.engine import evaluate, train_one_epoch
from src.models import create_model
from src.utils import log_model_size, plot_loss_curves, set_seed, setup_logging, to_builtin_types


def _set_nested(config, key_path, raw_value):
    """Set config[key1][key2]... from a dotted key; values are parsed as Python literals."""
    keys = key_path.split('.')
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    try:
        node[keys[-1]] = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        node[keys[-1]] = raw_value


def main():
    parser = argparse.ArgumentParser(description='Train PF-MLP or SF-CNN.')
    parser.add_argument('--config', required=True, help='YAML configuration path.')
    parser.add_argument('--override', action='append', default=[], metavar='KEY=VALUE',
                        help='Override a dotted config key; may be repeated.')
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    for item in args.override:
        if '=' not in item:
            raise ValueError(f'Expected KEY=VALUE for --override, received {item!r}.')
        key, _, value = item.partition('=')
        _set_nested(config, key.strip(), value.strip())

    experiment_name = config['output']['experiment_name']
    output_dir = os.path.join('saved_models', experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir, experiment_name)
    # Save effective overrides for evaluation and export.
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(to_builtin_types(config), stream, sort_keys=False)
    set_seed(config['env']['seed'])
    logging.info('Random seed: %d', config['env']['seed'])
    training = config['training']
    device_name = training['device']
    if 'cuda' in device_name and not torch.cuda.is_available():
        logging.warning('CUDA is unavailable; using CPU.')
        device_name = 'cpu'
    device = torch.device(device_name)
    logging.info('Training device: %s', device)

    train_loader, val_loader, test_loader, norm_params = create_dataloaders(
        config, training['batch_size'], training.get('num_workers', 4)
    )
    with open(os.path.join(output_dir, 'norm_params.yaml'), 'w') as stream:
        yaml.safe_dump(to_builtin_types(norm_params), stream, sort_keys=False)
    logging.info('Normalization parameters: %s', norm_params)
    model = create_model(config).to(device)
    logging.info('Model: %s', model)
    log_model_size(model, train_loader, device)
    if training['loss_function']['name'] != 'MSELoss':
        raise ValueError('Only MSELoss is supported.')
    if training['optimizer'] != 'AdamW':
        raise ValueError('Only AdamW is supported.')
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=training['learning_rate'],
                            weight_decay=training['weight_decay'])
    schedule = training['scheduler']
    if schedule.get('type', 'ReduceLROnPlateau') != 'ReduceLROnPlateau':
        raise ValueError('Only ReduceLROnPlateau is supported.')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=schedule['factor'], patience=schedule['patience']
    )
    history = {key: [] for key in ('train_loss', 'train_rmse', 'val_loss', 'val_rmse',
                                  'val_mean_error', 'val_std_error')}
    best_val_loss, best_val_rmse = float('inf'), float('inf')
    patience_counter = 0
    model_path = os.path.join(output_dir, f'{experiment_name}.pth')
    for epoch in range(training['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        _, train_metrics = evaluate(model, train_loader, loss_fn, device, norm_params)
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device, norm_params)
        history['train_loss'].append(train_loss)
        history['train_rmse'].append(train_metrics['rmse'])
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mean_error'].append(val_metrics['mean_error'])
        history['val_std_error'].append(val_metrics['std_error'])
        logging.info(
            'Epoch [%d/%d] | Train Loss: %.6f | Train RMSE: %.4fm | '
            'Val Loss: %.6f | Val RMSE: %.4fm | Mean/Std: %.4f/%.4fm | '
            'Median: %.4fm | LR: %.6f',
            epoch + 1, training['epochs'], train_loss, train_metrics['rmse'],
            val_loss, val_metrics['rmse'], val_metrics['mean_error'],
            val_metrics['std_error'], val_metrics['median_error'],
            optimizer.param_groups[0]['lr']
        )
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss, best_val_rmse = val_loss, val_metrics['rmse']
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            logging.info('Saved best model: %s', model_path)
        else:
            patience_counter += 1
        if patience_counter >= training['early_stopping']['patience']:
            logging.info('Early stopping after %d epochs without improvement.', patience_counter)
            break
    logging.info('Best validation MSE: %.6f; RMSE: %.4fm', best_val_loss, best_val_rmse)
    plot_loss_curves(history, os.path.join(output_dir, f'{experiment_name}_loss_curve.png'))


if __name__ == '__main__':
    main()
