import torch
from tempfile import TemporaryDirectory
import os
import shutil
from tqdm.std import tqdm
from collections import defaultdict
import math
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from torch.utils.data import Subset
import datasets


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
        epochs: int = 10,
        last_epoch: int = 0,                                        # when continuing training, the last epoch the model trained on
        tensorboard_dir: str = '',
        patience: int = 0,                                          # 0 means early stop is inactive
        metrics: list[callable] = [],
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,     # this is epoch-based scheduler
):
    dataloaders = { 'train': train_dataloader, 'val': val_dataloader }
    tb_writer = None
    if tensorboard_dir:
        tb_writer = SummaryWriter(tensorboard_dir)
        if last_epoch == 0 and os.path.exists(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        if last_epoch == 0:
            best_loss_epoch = 0
            best_loss = math.inf
        else:
            best_loss_epoch = last_epoch - 1
            best_loss = evaluate_model(model, val_dataloader, loss_fn)['loss']
            print(f'Best loss: {best_loss:.4f}')

        try: 
            for epoch in range(last_epoch, last_epoch + epochs):
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
                print(f"Epoch {epoch + 1}/{epochs}: " + ", ".join([f"{metric_name}={mean_metrics[metric_name]:.4f}" for metric_name in mean_metrics]))

                if tb_writer is not None:
                    tb_metrics = defaultdict(lambda: dict())
                    for full_metric_name in mean_metrics:
                        phase, *metric_name = full_metric_name.split('_')
                        metric_name = "_".join(metric_name)
                        tb_metrics[metric_name][phase] = mean_metrics[full_metric_name]
                    for metric_name in tb_metrics:
                        tb_writer.add_scalars(metric_name, tb_metrics[metric_name], epoch + 1)

                # deep copy the model
                if mean_metrics['val_loss'] < best_loss:
                    best_loss = mean_metrics['val_loss']
                    best_loss_epoch = epoch
                    print(f"Saving params from epoch {epoch + 1}. Best loss: {best_loss:.4f}")
                    torch.save(model.state_dict(), best_model_params_path)

                if patience > 0 and epoch - best_loss_epoch >= patience:
                    raise EarlyStopException
                
        except EarlyStopException:
            pass

        # load best model weights
        print(f"Loading model params from epoch {best_loss_epoch + 1}")
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


# train_dl, valid_dl, test_dl, class_names
def make_cifar_dataloaders(batch_size=64):
    train_set_full = CIFAR100(root='../data', train=True, transform=ToTensor(), download=True)
    test_set = CIFAR100(root='../data', train=False, transform=ToTensor(), download=True)

    indices = torch.randperm(len(train_set_full))
    train_set = Subset(train_set_full, indices[:-5000])
    valid_set = Subset(train_set_full, indices[-5000:])

    train_dl = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4)
    valid_dl = DataLoader(valid_set, shuffle=False, batch_size=batch_size, num_workers=4)
    test_dl = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=4)

    return train_dl, valid_dl, test_dl, train_set_full.classes


class TinyNetDataset(Dataset):
    def __init__(self, origin):
        self.origin = origin
        self.converter = ToTensor()

    def __getitem__(self, idx):
        entry = self.origin[idx] 
        label = entry['label']
        img = entry['image']
        img = img.convert('RGB')
        img = self.converter(img)
        return img, label
    
    def __len__(self):
        return len(self.origin)

TINY_IMAGE_NET_CLASSES = 200    
TINY_IMAGE_NET_SAMPLES_PER_CLASS = 500

def make_tiny_imagenet_dataloaders(batch_size=64, val_split=0.1):
    origin_ds = datasets.load_dataset("zh-plus/tiny-imagenet")

    train_set_full = TinyNetDataset(origin_ds['train'])
    test_set = TinyNetDataset(origin_ds['valid'])

    val_take_samples_n = math.floor(TINY_IMAGE_NET_SAMPLES_PER_CLASS * val_split)
    train_indices = []
    val_indices = []
    for class_idx in range(TINY_IMAGE_NET_CLASSES):
        class_indices = torch.arange(class_idx * TINY_IMAGE_NET_SAMPLES_PER_CLASS, (class_idx + 1) * TINY_IMAGE_NET_SAMPLES_PER_CLASS)
        class_indices = class_indices[torch.randperm(TINY_IMAGE_NET_SAMPLES_PER_CLASS)].tolist()
        train_indices += class_indices[val_take_samples_n:]
        val_indices += class_indices[:val_take_samples_n]

    train_set = Subset(train_set_full, train_indices)
    valid_set = Subset(train_set_full, val_indices)    

    train_dl = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4)
    valid_dl = DataLoader(valid_set, shuffle=False, batch_size=batch_size, num_workers=4)
    test_dl = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=4)

    return train_dl, valid_dl, test_dl

    

        

