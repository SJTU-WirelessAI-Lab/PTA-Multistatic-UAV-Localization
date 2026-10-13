import torch
from tqdm import tqdm

from src.utils import denormalize_positions


def train_one_epoch(model, dataloader, optimizer, loss_function, device):
    """Train one epoch and return the mean batch MSE."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    progress = tqdm(dataloader, desc='Training', leave=False)
    for inputs, labels in progress:
        inputs, labels = inputs.to(device), labels.to(device)
        loss = loss_function(model(inputs), labels)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        progress.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)


def evaluate(model, dataloader, loss_function, device, norm_params):
    """Return mean batch MSE and position error statistics in meters."""
    model.eval()
    total_loss = 0.0
    all_errors = []
    with torch.no_grad():
        progress = tqdm(dataloader, desc='Evaluating', leave=False)
        for inputs, labels in progress:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            outputs_real = denormalize_positions(outputs, norm_params)
            labels_real = denormalize_positions(labels, norm_params)
            errors = torch.sqrt(torch.sum((outputs_real - labels_real) ** 2, dim=1))
            all_errors.extend(errors.cpu().numpy())
            progress.set_postfix(loss=loss.item())
    all_errors = torch.tensor(all_errors)
    metrics = {
        'rmse': torch.sqrt(torch.mean(all_errors ** 2)).item(),
        'mean_error': torch.mean(all_errors).item(),
        'std_error': torch.std(all_errors).item(),
        'median_error': torch.median(all_errors).item(),
        'max_error': torch.max(all_errors).item()
    }
    return total_loss / len(dataloader), metrics
