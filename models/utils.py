import numpy as np
import humanize
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
from torchvision import transforms
from torch.utils.data import Subset
import datasets
import matplotlib.pyplot as plt
from pathlib import Path

# = Usage

"""
Put this inside models folder. Import using:

import utils
from importlib import reload
reload(utils)
utils.set_namespace('my_namespace')
"""

root_folder = Path('..')
noetbook_namespace = None
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def set_root_folder(root):
    global root_folder
    root_folder = Path(root)

def set_namespace(namespace):
    global noetbook_namespace
    noetbook_namespace = namespace

def _ensure_namespace():
    if not noetbook_namespace:
        raise Exception('Must call utils.set_namespace()')
    return True

def get_tensorboard_dir(model_name) -> Path:
    _ensure_namespace()
    return root_folder / 'runs' / noetbook_namespace / model_name

def get_checkpoint_path(model_name) -> Path:
    _ensure_namespace()
    return root_folder / 'ckpts' / noetbook_namespace / f'{model_name}.pt'

_model_registry = {}

# If checkpoint exist, it will be used to initialize the model parameters
def register_model(name, model, description=None):
    obj = _model_registry.get(name)
    if not obj:
        obj = ModelLifecycle(name, model, description)
        _model_registry[name] = obj
        obj.load_checkpoint()
        print(f'{name} registered.')
    else:
        print(f'{name} loaded from memory.')
    obj.print_params(full=False)
    return obj

def delete_model(name):
    found_anything = False
    obj = _model_registry.get(name)
    if obj:
        del(_model_registry[name])
        found_anything = True

    checkpoints_path = get_checkpoint_path(name)
    if checkpoints_path.exists():
        os.remove(checkpoints_path)
        found_anything = True

    tensorboard_dir = get_tensorboard_dir(name)
    if tensorboard_dir.exists():
        shutil.rmtree(tensorboard_dir)
        found_anything = True

    if found_anything:
        print(f'Deleted model {name}')
    else:
        print(f'Did not find any asset for {name}')
    return found_anything

# = Training helpers

def metric_accuracy(y_pred, y_true):
    return ['accuracy', torch.sum(torch.argmax(y_pred, dim=-1) == y_true).item()]

class EarlyStopException(Exception): pass

def plot_metrics(hist):
    x = hist['epoch']
    y = defaultdict(lambda: {})
    for metric_name in hist:
        if metric_name == 'epoch':
            continue
        chart_name = "_".join(metric_name.split('_')[1:])
        y[chart_name][metric_name] = hist[metric_name]
    charts = len(y)
    rows = math.ceil(charts / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3 * rows))
    axes = axes.ravel()
    for i, chart_name in enumerate(y):
        ax: plt.Axes = axes[i]
        ax.set_title(chart_name)
        for metric_name in y[chart_name]:
            ax.plot(x, hist[metric_name], label=metric_name)
        ax.legend(loc='upper left')
        ax.set_xticks(x)
        ax.grid()

# = Main class

class ModelLifecycle:
    def __init__(self, model_name, model, description=None):
        self.model_name = model_name
        self.model = model.to(device)
        self.description = description
        self.persisted = False
        self.compiled = False
        self.optimizer = None
        self.optimizer_args = None
        self.scheduler = None
        self.scheduler_args = None
        self.loss_fn = None
        self.loss_fn_args = None
        self.epoch = 0
        self.step = 0
        self.metrics = {}

    def __repr__(self):
        return self.model.__repr__()

    def get_tensorboard_dir(self) -> Path:
        return get_tensorboard_dir(self.model_name)

    def get_checkpoint_path(self) -> Path:
        return get_checkpoint_path(self.model_name)

    # Only runs if the model is not compiled, i.e. before first training
    def setup(self, *, optimizer_cls, loss_fn_cls, optimizer_args={}, loss_fn_args={}, scheduler_cls=None, scheduler_args={}):
        if not self.compiled:
            self.compile(optimizer_cls=optimizer_cls, 
                         loss_fn_cls=loss_fn_cls, 
                         optimizier_args=optimizer_args,
                         loss_fn_args=loss_fn_args, 
                         scheduler_cls=scheduler_cls, 
                         scheduler_args=scheduler_args)
            print('Compiled')
        else:
            print('Skipped compilation')
        return self

    # Forces changes to the entry regardless of model state
    def compile(self, *, optimizer_cls, loss_fn_cls, optimizier_args={}, loss_fn_args={}, scheduler_cls=None, scheduler_args={}):    
        optimizer = optimizer_cls(self.model.parameters(), **optimizier_args)
        loss_fn = loss_fn_cls(**loss_fn_args)
        scheduler = None
        if scheduler_cls:
            scheduler = scheduler_cls(optimizer, **scheduler_args)

        self.optimizer = optimizer
        self.optimizer_args = optimizier_args
        self.loss_fn = loss_fn
        self.loss_fn_args = loss_fn_args
        self.scheduler = scheduler
        self.scheduler_args = scheduler_args
        self.compiled = True
        return self

    def save_checkpoint(self, epoch=None, step=None, metrics=None):
        _ensure_namespace()
        save_path = self.get_checkpoint_path()
        os.makedirs(save_path.parent, exist_ok=True)
        if epoch:
            self.epoch = epoch
        if step:
            self.step = step
        if metrics:
            self.metrics = metrics

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'description': self.description,
            'loss_fn_cls': self.loss_fn.__class__,
            'loss_fn_args': self.loss_fn_args,
            'epoch': self.epoch,
            'step': self.step,
            'metrics': self.metrics,
            'compiled': self.compiled
        }
        if self.optimizer:
            checkpoint['optimizer_cls'] = self.optimizer.__class__
            checkpoint['optimizer_args'] = self.optimizer_args
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

            # Should not have a scheduler without an optimizer
            if self.scheduler:
                checkpoint['scheduler_cls'] = self.scheduler.__class__
                checkpoint['scheduler_args'] = self.scheduler_args
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, save_path)
        self.persisted = True

        saved_str = f"Saved checkpoint for {self.model_name}. epoch={self.epoch}, step={self.step}."
        metrics_str = ", ".join([f'{metric_name}={metric_val:4f}' for metric_name, metric_val in self.metrics.items()])
        print(" ".join([x for x in [saved_str, metrics_str] if x]))
        return True

    def load_checkpoint(self):
        _ensure_namespace()
        load_path = self.get_checkpoint_path()
        if not load_path.exists():
            return False
        
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)

        # Allow overriding the saved description
        if not self.description:
            self.description = checkpoint['description']

        loss_fn_args = checkpoint['loss_fn_args'] or {}
        self.loss_fn = checkpoint['loss_fn_cls'](**loss_fn_args)
        self.loss_fn_args = checkpoint['loss_fn_args']
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.metrics = checkpoint['metrics']
        self.compiled = checkpoint['compiled']

        if checkpoint.get('optimizer_cls'):
            optimizer_args = checkpoint['optimizer_args'] or {}
            self.optimizer = checkpoint['optimizer_cls'](self.model.parameters(), **optimizer_args)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.optimizer_args = checkpoint['optimizer_args']

            if checkpoint.get('scheduler_cls'):
                scheduler_args = checkpoint['scheduler_args'] or {}
                self.scheduler = checkpoint['scheduler_cls'](self.optimizer, **scheduler_args)
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.scheduler_args = checkpoint['scheduler_args']

        self.persisted = True
        
        loaded_str = f"Loaded model {self.model_name} from checkpoint. epoch={self.epoch}, step={self.step}."
        metrics_str = ", ".join([f'{metric_name}={metric_val:4f}' for metric_name, metric_val in self.metrics.items()])
        print(" ".join([x for x in [loaded_str, metrics_str] if x]))
        return True        

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int = 10, patience: int = 0, metrics: list[callable] = []):
        if not self.compiled:
            raise Exception(f'model {self.model_name} was not compiled. Call setup() or compile()')

        tensorboard_dir = self.get_tensorboard_dir()
        os.makedirs(tensorboard_dir, exist_ok=True)

        dataloaders = { 'train': train_dataloader, 'val': val_dataloader }
        tb_writer = SummaryWriter(tensorboard_dir)
        history = defaultdict(lambda: [])
        
        step_counter = self.step
        epoch_counter = self.epoch

        if self.epoch == 0:
            best_loss_epoch = 0
            best_loss = math.inf
            if tb_writer is not None:
                inputs, _ = next(iter(train_dataloader))
                tb_writer.add_graph(self.model, inputs.to(device))
                tb_writer.flush()
        else:
            best_loss_epoch = self.epoch - 1
            best_loss = self.evaluate(val_dataloader)['loss']
            print(f'Best loss: {best_loss:.4f}')

        try: 
            for epoch in range(epochs):
                running_metrics = { 'train': defaultdict(lambda: 0), 'val': defaultdict(lambda: 0) }

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        self.model.train()  # Set model to training mode
                    else:
                        self.model.eval()   # Set model to evaluate mode

                    # Iterate over data.
                    for inputs, y_true in tqdm(dataloaders[phase]):
                        inputs: torch.Tensor = inputs.to(device)
                        y_true: torch.Tensor = y_true.to(device)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            y_pred = self.model(inputs)
                            loss = self.loss_fn(y_pred, y_true)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()

                        # statistics
                        running_metrics[phase]['loss'] += loss.item() * inputs.shape[0]
                        for metric in metrics:
                            metric_name, metric_val = metric(y_pred, y_true)
                            running_metrics[phase][metric_name] += metric_val

                        if phase == 'train':
                            step_counter += 1

                    if phase == 'train' and self.scheduler is not None:
                        self.scheduler.step()

                epoch_counter += 1

                mean_metrics = { f"{phase}_{metric_name}": running_metrics[phase][metric_name] / len(dataloaders[phase].dataset)
                                for phase in running_metrics  
                                for metric_name in running_metrics[phase] }

                # Print loss and metrics
                print(f"Epoch {epoch + 1}/{epochs}: " + ", ".join([f"{metric_name}={mean_metrics[metric_name]:.4f}" for metric_name in mean_metrics]))

                # Store metrics in history
                history['epoch'].append(epoch_counter)
                for full_metric_name in mean_metrics:
                    history[full_metric_name].append(mean_metrics[full_metric_name])

                # Report to tensorboard
                if tb_writer is not None:
                    tb_metrics = defaultdict(lambda: dict())
                    for full_metric_name in mean_metrics:
                        phase, *metric_name = full_metric_name.split('_')
                        metric_name = "_".join(metric_name)
                        tb_metrics[metric_name][phase] = mean_metrics[full_metric_name]
                    for metric_name in tb_metrics:
                        tb_writer.add_scalars(metric_name, tb_metrics[metric_name], epoch_counter)
                    tb_writer.flush()

                # deep copy the model
                if mean_metrics['val_loss'] < best_loss:
                    best_loss = mean_metrics['val_loss']
                    best_loss_epoch = epoch
                    self.save_checkpoint(epoch=epoch_counter, step=step_counter, metrics=mean_metrics)

                if patience > 0 and epoch - best_loss_epoch >= patience:
                    raise EarlyStopException
                
        except EarlyStopException:
            pass

        # load best model weights
        self.load_checkpoint()
        return history

    def evaluate(self, dataloader: DataLoader, metrics: list[callable] = []):
        if not self.compiled:
            raise Exception(f'model {self.model_name} was not compiled. Call setup() or compile()')
        
        self.model.eval()
        running_metrics = defaultdict(lambda: 0)
        with torch.no_grad():
            for inputs, y_true in dataloader:
                inputs: torch.Tensor = inputs.to(device)
                y_true: torch.Tensor = y_true.to(device)

                y_pred = self.model(inputs)
                loss = self.loss_fn(y_pred, y_true)
                running_metrics['loss'] += loss.item() * inputs.shape[0]
                for metric in metrics:
                    metric_name, metric_val = metric(y_pred, y_true)
                    running_metrics[metric_name] += metric_val
            
            mean_metrics = { metric_name: running_metrics[metric_name] / len(dataloader.dataset)
                            for metric_name in running_metrics }
            
            return mean_metrics
        
    def print_params(self, full=True):
        stats = defaultdict(lambda: dict())
        trainable = 0
        untrainable = 0
        buffers = 0
        human_fn = humanize.metric 

        for full_name, param in self.model.named_parameters():
            *layer_name, param_name = full_name.split('.')
            layer_name = '.'.join(layer_name)
            param_count = np.prod(param.shape)
            stats[layer_name][param_name] = param
            if param.requires_grad: 
                trainable += param_count
            else:
                untrainable += param_count

        for full_name, buff in self.model.named_buffers():
            *layer_name, buff_name = full_name.split('.')
            layer_name = '.'.join(layer_name)
            param_count = np.prod(param.shape)
            stats[layer_name][buff_name] = buff
            buffers += param_count

        longest_layer_name = np.max([len(x) for x in stats.keys()])
        longest_param_or_buff_name = np.max([len(x) for l in stats for x in stats[l].keys()])

        if full:
            for layer_name in stats:
                for i, param_or_buff_name in enumerate(stats[layer_name]):
                    param_or_buff = stats[layer_name][param_or_buff_name]
                    param_count = np.prod(param_or_buff.shape)
                    if isinstance(param_or_buff, torch.nn.Parameter):
                        type_char = '✅' if param_or_buff.requires_grad else '🔒'
                    else:
                        type_char = '⛔'
                    if i == 0:
                        print_name = f'{layer_name:{longest_layer_name}}'
                    else:
                        pad = ' '
                        print_name = f'{pad:{longest_layer_name}}'

                    print(print_name, f'{param_or_buff_name:{longest_param_or_buff_name}}', type_char, '{:10}'.format(human_fn(param_count)), list(param_or_buff.shape))

            print('-' * 100)        
            print(f"✅ Trainable params:   {human_fn(trainable)}")
            print(f"🔒 Untrainable params: {human_fn(untrainable)}")
            print(f"⛔ Buffers:            {human_fn(buffers)}")
            print('-' * 100)
        else:
            print(f"Trainable params: {human_fn(trainable)}. Untrainable params: {human_fn(untrainable)}. Buffers: {human_fn(buffers)}.")

# = Data loaders

# train_dl, valid_dl, test_dl, class_names
def make_cifar_dataloaders(batch_size=64):
    train_set_full = CIFAR100(root='../data', train=True, transform=transforms.ToTensor(), download=True)
    test_set = CIFAR100(root='../data', train=False, transform=transforms.ToTensor(), download=True)

    indices = torch.randperm(len(train_set_full))
    train_set = Subset(train_set_full, indices[:-5000])
    valid_set = Subset(train_set_full, indices[-5000:])

    train_dl = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4)
    valid_dl = DataLoader(valid_set, shuffle=False, batch_size=batch_size, num_workers=4)
    test_dl = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=4)

    return train_dl, valid_dl, test_dl, train_set_full.classes

class TinyNetDataset(Dataset):
    def __init__(self, origin, normalize=False, augment=False):
        self.origin = origin
        transformations = []
        if augment:
            transformations = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ]
        transformations.append(transforms.ToTensor())
        if normalize:
            transformations.append(
                transforms.Normalize(mean=[0.4804, 0.4482, 0.3977], std=[0.2764, 0.2688, 0.2816])
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        self.converter = transforms.Compose(transformations)

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

def make_tiny_imagenet_dataloaders(batch_size=64, val_split=0.1, train_split=None, normalize=False, augment=False):
    if not train_split:
        train_split = 1 - val_split

    if val_split + train_split > 1:
        val_split /= val_split + train_split
        train_split /= val_split + train_split

    val_take_samples_n = math.floor(TINY_IMAGE_NET_SAMPLES_PER_CLASS * val_split)
    train_take_samples_n = math.floor(TINY_IMAGE_NET_SAMPLES_PER_CLASS * train_split)

    train_indices = []
    val_indices = []
    for class_idx in range(TINY_IMAGE_NET_CLASSES):
        class_indices = torch.arange(class_idx * TINY_IMAGE_NET_SAMPLES_PER_CLASS, (class_idx + 1) * TINY_IMAGE_NET_SAMPLES_PER_CLASS)
        class_indices = class_indices[torch.randperm(TINY_IMAGE_NET_SAMPLES_PER_CLASS)].tolist()
        train_indices += class_indices[val_take_samples_n:val_take_samples_n+train_take_samples_n]
        val_indices += class_indices[:val_take_samples_n]

    origin_ds = datasets.load_dataset("zh-plus/tiny-imagenet")

    train_set = TinyNetDataset(Subset(origin_ds['train'], train_indices), normalize=normalize, augment=augment)
    valid_set = TinyNetDataset(Subset(origin_ds['train'], val_indices), normalize=normalize)
    test_set = TinyNetDataset(origin_ds['valid'], normalize=normalize)

    train_dl = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4)
    valid_dl = DataLoader(valid_set, shuffle=False, batch_size=batch_size, num_workers=4)
    test_dl = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=4)

    return train_dl, valid_dl, test_dl
