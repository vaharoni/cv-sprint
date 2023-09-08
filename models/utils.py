import torch
from tempfile import TemporaryDirectory
import os
import shutil
from tqdm.std import tqdm
from collections import defaultdict
import math
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from torch.utils.data import Subset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def metric_accuracy(y_pred, y_true):
    return ['accuracy', torch.sum(torch.argmax(y_pred, dim=-1) == y_true).item()]

class EarlyStopException(Exception): pass

def train_model(
        model: torch.nn.Module, 
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        loss_fn: callable, 
        optimizer: torch.optim.Optimizer, 
        num_epochs: int = 10,
        tensorboard_dir: str = '',
        patience: int = 0,                                          # 0 means early stop is inactive
        metrics: list[callable] = [],
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,     # this is epoch-based scheduler
):
    dataloaders = { 'train': train_dataloader, 'val': val_dataloader }
    tb_writer = None
    if tensorboard_dir:
        tb_writer = SummaryWriter(tensorboard_dir)
        if os.path.exists(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_loss = math.inf
        best_loss_epoch = 0

        try: 
            for epoch in range(num_epochs):
                running_metrics = { 'train': defaultdict(lambda: 0), 'val': defaultdict(lambda: 0) }

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode

                    # Iterate over data.
                    for inputs, y_true in tqdm(dataloaders[phase]):
                        inputs: torch.Tensor = inputs.to(device)
                        y_true: torch.Tensor = y_true.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            y_pred = model(inputs)
                            loss = loss_fn(y_pred, y_true)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_metrics[phase]['loss'] += loss.item() * inputs.shape[0]
                        for metric in metrics:
                            metric_name, metric_val = metric(y_pred, y_true)
                            running_metrics[phase][metric_name] += metric_val

                    if phase == 'train' and scheduler is not None:
                        scheduler.step()

                mean_metrics = { f"{phase}_{metric_name}": running_metrics[phase][metric_name] / len(dataloaders[phase].dataset)
                                for phase in running_metrics  
                                for metric_name in running_metrics[phase] }

                # Print loss and metrics
                print(f"Epoch {epoch + 1}/{num_epochs}: " + ", ".join([f"{metric_name}={mean_metrics[metric_name]:.4f}" for metric_name in mean_metrics]))

                if tb_writer is not None:
                    tb_metrics = defaultdict(lambda: dict())
                    for full_metric_name in mean_metrics:
                        phase, *metric_name = full_metric_name.split('_')
                        metric_name = "_".join(metric_name)
                        tb_metrics[metric_name][phase] = mean_metrics[full_metric_name]
                    for metric_name in tb_metrics:
                        tb_writer.add_scalars(metric_name, tb_metrics[metric_name], epoch)

                # deep copy the model
                if mean_metrics['val_loss'] < best_loss:
                    best_loss = mean_metrics['val_loss']
                    best_loss_epoch = epoch
                    torch.save(model.state_dict(), best_model_params_path)

                if patience > 0 and epoch - best_loss_epoch >= patience:
                    raise EarlyStopException
                
        except EarlyStopException:
            pass

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

def evaluate_model(
        model: torch.nn.Module, 
        dataloader: DataLoader, 
        loss_fn: callable,
        metrics: list[callable] = [],
):
    model.eval()
    running_metrics = defaultdict(lambda: 0)
    with torch.no_grad():
        for inputs, y_true in dataloader:
            inputs: torch.Tensor = inputs.to(device)
            y_true: torch.Tensor = y_true.to(device)

            y_pred = model(inputs)
            loss = loss_fn(y_pred, y_true)
            running_metrics['loss'] += loss.item() * inputs.shape[0]
            for metric in metrics:
                metric_name, metric_val = metric(y_pred, y_true)
                running_metrics[metric_name] += metric_val
        
        mean_metrics = { metric_name: running_metrics[metric_name] / len(dataloader.dataset)
                        for metric_name in running_metrics }
        
        return mean_metrics


def make_dataloaders(batch_size=32):
    train_set_full = CIFAR100(root='../data', train=True, transform=ToTensor(), download=True)
    test_set = CIFAR100(root='../data', train=False, transform=ToTensor(), download=True)

    indices = torch.randperm(len(train_set_full))
    train_set = Subset(train_set_full, indices[:-5000])
    valid_set = Subset(train_set_full, indices[-5000:])

    train_dl = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=2)
    valid_dl = DataLoader(valid_set, shuffle=False, batch_size=batch_size, num_workers=2)
    test_dl = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=2)

    return train_dl, valid_dl, test_dl, train_set_full.classes