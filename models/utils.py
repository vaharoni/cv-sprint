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
import pprint
from copy import deepcopy

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
        print(f'{name} fetched from memory.')
    obj.print_params(full=False)
    return obj

def fork_model(original_name, original_model, new_name, description=None):
    obj = _model_registry.get(new_name)
    if obj:
        print(f'{new_name} fetched from memory.')
        return obj

    elif get_checkpoint_path(new_name).exists():
        obj = ModelLifecycle(model_name=new_name, model=original_model, description=description)
        obj.load_checkpoint()
        _model_registry[new_name] = obj
        print(f'{new_name} registered.')
        return obj

    elif get_checkpoint_path(original_name).exists():
        obj = ModelLifecycle(model_name=original_name, model=original_model)
        obj.load_checkpoint()
        obj._reset_attr(model_name=new_name, description=description)
        _model_registry[new_name] = obj
        print(f'{new_name} registered.')
        return obj

    else:
        print(f'Cannot find checkpoint for model {original_name}. Was it saved?')
        return

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
        self.model = model.to(device)
        self._reset_attr(model_name, description)
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

    def _reset_attr(self, model_name, description):
        self.model_name = model_name
        self.description = description
        self.persisted = False

    def __repr__(self):
        return pprint.pformat({
            'model_name': self.model_name,
            'description': self.description,
            'persisted': self.persisted,
            'compiled': self.compiled,
            'optimizer_cls': self.optimizer.__class__,
            'optimizer_args': self.optimizer_args,
            'loss_fn_cls': self.loss_fn.__class__,
            'loss_fn_args': self.loss_fn_args,
            'scheduler_cls': self.scheduler.__class__,
            'scheduler_args': self.scheduler_args,
            'epoch': self.epoch,
            'step': self.step,
            'metrics': self.metrics
        }, width=110, sort_dicts=False)

    def get_tensorboard_dir(self) -> Path:
        return get_tensorboard_dir(self.model_name)

    def get_checkpoint_path(self) -> Path:
        return get_checkpoint_path(self.model_name)

    # Only runs if the model is not compiled, i.e. before first training.
    # The point is that when initializing a model like so:
    #   my_model = utils.register_model(...)
    #   my_model.setup(...)
    #
    # Then if the model is loaded from a checkpoint, in fact the checkpoing value will override the parameters sent to setup.
    #
    def setup(self, *, optimizer_cls, loss_fn_cls, optimizer_args={}, loss_fn_args={}, scheduler_cls=None, scheduler_args={}):
        if not self.compiled:
            self.compile(optimizer_cls=optimizer_cls, 
                         loss_fn_cls=loss_fn_cls, 
                         optimizer_args=optimizer_args,
                         loss_fn_args=loss_fn_args, 
                         scheduler_cls=scheduler_cls, 
                         scheduler_args=scheduler_args)
            print('Compiled')
        else:
            print('Skipped compilation')
        return self

    # TODO: allow only to change the defaults rather than redefine everything
    # Forces changes to the entry regardless of model state
    def compile(self, *, optimizer_cls, loss_fn_cls, optimizer_args={}, loss_fn_args={}, scheduler_cls=None, scheduler_args={}):    
        optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)
        loss_fn = loss_fn_cls(**loss_fn_args)
        scheduler = None
        if scheduler_cls:
            scheduler = scheduler_cls(optimizer, **scheduler_args)

        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
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
        metrics_str = ", ".join([f'{metric_name}={metric_val:.4f}' for metric_name, metric_val in self.metrics.items()])
        print(" ".join([x for x in [saved_str, metrics_str] if x]))
        return True

    def load_checkpoint(self):
        _ensure_namespace()
        load_path = self.get_checkpoint_path()
        if not load_path.exists():
            print('Skipping load: checkpoint file does not exist')
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
        metrics_str = ", ".join([f'{metric_name}={metric_val:.4f}' for metric_name, metric_val in self.metrics.items()])
        print(" ".join([x for x in [loaded_str, metrics_str] if x]))
        return True
    
    def fork(self, model_name, description=None):
        return fork_model(original_name=self.model_name, original_model=self.model, new_name=model_name, description=description)

    def delete(self):
        return delete_model(self.model_name)  
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int = 10, patience: int = 0, warmup: int = 1, metrics: list[callable] = []):
        if not self.compiled:
            raise Exception(f'model {self.model_name} was not compiled. Call setup() or compile()')

        tensorboard_dir = self.get_tensorboard_dir()
        os.makedirs(tensorboard_dir, exist_ok=True)

        dataloaders = { 'train': train_dataloader, 'val': val_dataloader }
        tb_writer = SummaryWriter(tensorboard_dir)
        history = defaultdict(lambda: [])
        
        step_counter = self.step
        epoch_counter = self.epoch

        best_loss_epoch = max(self.epoch - 1, 0)
        best_loss = self.evaluate(val_dataloader)['loss']
        print(f'Initial val_loss: {best_loss:.4f}')

        if self.epoch == 0 and tb_writer is not None:
            inputs, _ = next(iter(train_dataloader))
            tb_writer.add_graph(self.model, inputs.to(device))
            tb_writer.flush()

        warmup_scheduler = None
        if warmup > 0 and warmup > epoch_counter:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1e-3, total_iters=len(train_dataloader) * warmup)

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
                                if warmup_scheduler and warmup > epoch_counter:
                                    warmup_scheduler.step()

                        # statistics
                        running_metrics[phase]['loss'] += loss.item() * inputs.shape[0]
                        for metric in metrics:
                            metric_name, metric_val = metric(y_pred, y_true)
                            running_metrics[phase][metric_name] += metric_val

                        if phase == 'train':
                            step_counter += 1

                    if phase == 'train' and self.scheduler is not None and warmup <= epoch_counter:
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
                    best_loss_epoch = epoch_counter
                    self.save_checkpoint(epoch=epoch_counter, step=step_counter, metrics=mean_metrics)

                if patience > 0 and epoch_counter - best_loss_epoch >= patience:
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

class CompareModelWeights():
    def __init__(self):
        self.combined_stats = defaultdict(lambda: [])

    def record(self, model):
        state_dict = deepcopy(model.state_dict())    
        for key in state_dict.keys():
            *layer_name, param_name = key.split('.')
            layer_name = '.'.join(layer_name)
            self.combined_stats[(layer_name, param_name)].append(state_dict[key])

    def get_agg_stats(self):
        stats = defaultdict(lambda: [])
        for (layer_name, param_name), param_arr in self.combined_stats.items():
            for param in param_arr:
                if param.dim() == 0:
                    stats[(layer_name, param_name)].append(param.item())
                else:
                    stats[(layer_name, param_name)].append((round(param.mean().item(), 4), round(param.std().item(), 4)))
        return stats
        
    def print_diff(self):
        agg_stats = self.get_agg_stats()
        longest_layer_name = max([len(x[0]) for x in agg_stats.keys()])
        longest_param_name = max([len(x[1]) for x in agg_stats.keys()])
        for key, value_arr in agg_stats.items():
            if len(set(value_arr)) > 1:
                print(f"{key[0]:{longest_layer_name}}   ", f"{key[1]:{longest_param_name}}   ", *[f"{str(x):20}" for x in value_arr])


# Per channel, for normalization calculations. This should be called after ToTensor().
def calc_image_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_pixels = 0, 0, 0

    for data, _ in dataloader:
        channels_sum += torch.sum(data, dim=[0,2,3])
        channels_squared_sum += torch.sum(data**2, dim=[0,2,3])
        num_pixels += data.size(0) * data.size(2) * data.size(3)

    mean = channels_sum / num_pixels
    std = (channels_squared_sum/num_pixels - mean**2)**0.5

    return mean, std

# = Data loaders

# train_dl, valid_dl, test_dl, class_names
def make_cifar_dataloaders(batch_size=64):
    # Mine
    # train_transforms = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(10),
    #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5072, 0.4866, 0.4410), (0.2673, 0.2564, 0.2760)),
    # ])

    # pytorch-CIFAR100
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5072, 0.4866, 0.4410), (0.2673, 0.2564, 0.2760))        
    ])

    valid_test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5072, 0.4866, 0.4410), (0.2673, 0.2564, 0.2760)),
    ])

    train_set_full = CIFAR100(root='../data', train=True, download=True)
    test_set = CIFAR100(root='../data', train=False, transform=valid_test_transforms, download=True)

    # indices = torch.randperm(len(train_set_full))
    indices = torch.arange(len(train_set_full))
    train_indices = indices[:-5000]
    valid_indices = indices[-5000:]

    train_set = Subset(train_set_full, train_indices)
    train_set = [(train_transforms(img), label) for img, label in train_set]

    valid_set = Subset(train_set_full, valid_indices)
    valid_set = [(valid_test_transforms(img), label) for img, label in valid_set]

    train_dl = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4)
    valid_dl = DataLoader(valid_set, shuffle=False, batch_size=batch_size, num_workers=4)
    test_dl = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=4)

    return train_dl, valid_dl, test_dl, train_set_full.classes


def make_cifar_dataloaders_without_validation(batch_size=64):
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5072, 0.4866, 0.4410), (0.2673, 0.2564, 0.2760))        
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5072, 0.4866, 0.4410), (0.2673, 0.2564, 0.2760))
    ])

    train_set = CIFAR100(root='../data', train=True, transform=train_transforms, download=True)
    test_set = CIFAR100(root='../data', train=False, transform=test_transforms, download=True)

    train_dl = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4)
    test_dl = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=4)

    return train_dl, test_dl, train_set.classes


# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
        class_indices = torch.arange(class_idx * TINY_IMAGE_NET_SAMPLES_PER_CLASS, (class_idx + 1) * TINY_IMAGE_NET_SAMPLES_PER_CLASS).tolist()
        train_indices += class_indices[val_take_samples_n:val_take_samples_n+train_take_samples_n]
        val_indices += class_indices[:val_take_samples_n]

    origin_ds = datasets.load_dataset("zh-plus/tiny-imagenet")

    train_set = TinyNetDataset(Subset(origin_ds['train'], train_indices), normalize=normalize, augment=augment)
    valid_set = TinyNetDataset(Subset(origin_ds['train'], val_indices), normalize=normalize)
    test_set = TinyNetDataset(origin_ds['valid'], normalize=normalize)

    train_dl = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4)
    valid_dl = DataLoader(valid_set, shuffle=False, batch_size=batch_size, num_workers=4)
    test_dl = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=4)

    return train_dl, valid_dl, test_dl, TINY_IMAGE_NET_CLASS_KEY

TINY_IMAGE_NET_CLASS_KEY = [
    ('n01443537', 'goldfish, Carassius auratus'),
    ('n01629819', 'European fire salamander, Salamandra salamandra'),
    ('n01641577', 'bullfrog, Rana catesbeiana'),
    ('n01644900',
    'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui'),
    ('n01698640', 'American alligator, Alligator mississipiensis'),
    ('n01742172', 'boa constrictor, Constrictor constrictor'),
    ('n01768244', 'trilobite'),
    ('n01770393', 'scorpion'),
    ('n01774384', 'black widow, Latrodectus mactans'),
    ('n01774750', 'tarantula'),
    ('n01784675', 'centipede'),
    ('n01882714',
    'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus'),
    ('n01910747', 'jellyfish'),
    ('n01917289', 'brain coral'),
    ('n01944390', 'snail'),
    ('n01950731', 'sea slug, nudibranch'),
    ('n01983481',
    'American lobster, Northern lobster, Maine lobster, Homarus americanus'),
    ('n01984695',
    'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish'),
    ('n02002724', 'black stork, Ciconia nigra'),
    ('n02056570', 'king penguin, Aptenodytes patagonica'),
    ('n02058221', 'albatross, mollymawk'),
    ('n02074367', 'dugong, Dugong dugon'),
    ('n02094433', 'Yorkshire terrier'),
    ('n02099601', 'golden retriever'),
    ('n02099712', 'Labrador retriever'),
    ('n02106662',
    'German shepherd, German shepherd dog, German police dog, alsatian'),
    ('n02113799', 'standard poodle'),
    ('n02123045', 'tabby, tabby cat'),
    ('n02123394', 'Persian cat'),
    ('n02124075', 'Egyptian cat'),
    ('n02125311',
    'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'),
    ('n02129165', 'lion, king of beasts, Panthera leo'),
    ('n02132136', 'brown bear, bruin, Ursus arctos'),
    ('n02165456', 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle'),
    ('n02226429', 'grasshopper, hopper'),
    ('n02231487', 'walking stick, walkingstick, stick insect'),
    ('n02233338', 'cockroach, roach'),
    ('n02236044', 'mantis, mantid'),
    ('n02268443',
    "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk"),
    ('n02279972',
    'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus'),
    ('n02281406', 'sulphur butterfly, sulfur butterfly'),
    ('n02321529', 'sea cucumber, holothurian'),
    ('n02364673', 'guinea pig, Cavia cobaya'),
    ('n02395406', 'hog, pig, grunter, squealer, Sus scrofa'),
    ('n02403003', 'ox'),
    ('n02410509', 'bison'),
    ('n02415577',
    'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis'),
    ('n02423022', 'gazelle'),
    ('n02437312', 'Arabian camel, dromedary, Camelus dromedarius'),
    ('n02480495', 'orangutan, orang, orangutang, Pongo pygmaeus'),
    ('n02481823', 'chimpanzee, chimp, Pan troglodytes'),
    ('n02486410', 'baboon'),
    ('n02504458', 'African elephant, Loxodonta africana'),
    ('n02509815',
    'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens'),
    ('n02666347', 'abacus'),
    ('n02669723', "academic gown, academic robe, judge's robe"),
    ('n02699494', 'altar'),
    ('n02769748', 'backpack, back pack, knapsack, packsack, rucksack, haversack'),
    ('n02788148', 'bannister, banister, balustrade, balusters, handrail'),
    ('n02791270', 'barbershop'),
    ('n02793495', 'barn'),
    ('n02795169', 'barrel, cask'),
    ('n02802426', 'basketball'),
    ('n02808440', 'bathtub, bathing tub, bath, tub'),
    ('n02814533',
    'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon'),
    ('n02814860', 'beacon, lighthouse, beacon light, pharos'),
    ('n02815834', 'beaker'),
    ('n02823428', 'beer bottle'),
    ('n02837789', 'bikini, two-piece'),
    ('n02841315', 'binoculars, field glasses, opera glasses'),
    ('n02843684', 'birdhouse'),
    ('n02883205', 'bow tie, bow-tie, bowtie'),
    ('n02892201', 'brass, memorial tablet, plaque'),
    ('n02909870', 'bucket, pail'),
    ('n02917067', 'bullet train, bullet'),
    ('n02927161', 'butcher shop, meat market'),
    ('n02948072', 'candle, taper, wax light'),
    ('n02950826', 'cannon'),
    ('n02963159', 'cardigan'),
    ('n02977058',
    'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM'),
    ('n02988304', 'CD player'),
    ('n03014705', 'chest'),
    ('n03026506', 'Christmas stocking'),
    ('n03042490', 'cliff dwelling'),
    ('n03085013', 'computer keyboard, keypad'),
    ('n03089624', 'confectionery, confectionary, candy store'),
    ('n03100240', 'convertible'),
    ('n03126707', 'crane'),
    ('n03160309', 'dam, dike, dyke'),
    ('n03179701', 'desk'),
    ('n03201208', 'dining table, board'),
    ('n03255030', 'dumbbell'),
    ('n03355925', 'flagpole, flagstaff'),
    ('n03373237', 'fly'),
    ('n03388043', 'fountain'),
    ('n03393912', 'freight car'),
    ('n03400231', 'frying pan, frypan, skillet'),
    ('n03404251', 'fur coat'),
    ('n03424325', 'gasmask, respirator, gas helmet'),
    ('n03444034', 'go-kart'),
    ('n03447447', 'gondola'),
    ('n03544143', 'hourglass'),
    ('n03584254', 'iPod'),
    ('n03599486', 'jinrikisha, ricksha, rickshaw'),
    ('n03617480', 'kimono'),
    ('n03637318', 'lampshade, lamp shade'),
    ('n03649909', 'lawn mower, mower'),
    ('n03662601', 'lifeboat'),
    ('n03670208', 'limousine, limo'),
    ('n03706229', 'magnetic compass'),
    ('n03733131', 'maypole'),
    ('n03763968', 'military uniform'),
    ('n03770439', 'miniskirt, mini'),
    ('n03796401', 'moving van'),
    ('n03814639', 'neck brace'),
    ('n03837869', 'obelisk'),
    ('n03838899', 'oboe, hautboy, hautbois'),
    ('n03854065', 'organ, pipe organ'),
    ('n03891332', 'parking meter'),
    ('n03902125', 'pay-phone, pay-station'),
    ('n03930313', 'picket fence, paling'),
    ('n03937543', 'pill bottle'),
    ('n03970156', "plunger, plumber's helper"),
    ('n03977966',
    'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria'),
    ('n03980874', 'poncho'),
    ('n03983396', 'pop bottle, soda bottle'),
    ('n03992509', "potter's wheel"),
    ('n04008634', 'projectile, missile'),
    ('n04023962', 'punching bag, punch bag, punching ball, punchball'),
    ('n04070727', 'refrigerator, icebox'),
    ('n04074963', 'remote control, remote'),
    ('n04099969', 'rocking chair, rocker'),
    ('n04118538', 'rugby ball'),
    ('n04133789', 'sandal'),
    ('n04146614', 'school bus'),
    ('n04149813', 'scoreboard'),
    ('n04179913', 'sewing machine'),
    ('n04251144', 'snorkel'),
    ('n04254777', 'sock'),
    ('n04259630', 'sombrero'),
    ('n04265275', 'space heater'),
    ('n04275548', "spider web, spider's web"),
    ('n04285008', 'sports car, sport car'),
    ('n04311004', 'steel arch bridge'),
    ('n04328186', 'stopwatch, stop watch'),
    ('n04356056', 'sunglasses, dark glasses, shades'),
    ('n04366367', 'suspension bridge'),
    ('n04371430', 'swimming trunks, bathing trunks'),
    ('n04376876', 'syringe'),
    ('n04398044', 'teapot'),
    ('n04399382', 'teddy, teddy bear'),
    ('n04417672', 'thatch, thatched roof'),
    ('n04456115', 'torch'),
    ('n04465666', 'tractor'),
    ('n04486054', 'triumphal arch'),
    ('n04487081', 'trolleybus, trolley coach, trackless trolley'),
    ('n04501370', 'turnstile'),
    ('n04507155', 'umbrella'),
    ('n04532106', 'vestment'),
    ('n04532670', 'viaduct'),
    ('n04540053', 'volleyball'),
    ('n04560804', 'water jug'),
    ('n04562935', 'water tower'),
    ('n04596742', 'wok'),
    ('n04598010', 'wooden spoon'),
    ('n06596364', 'comic book'),
    ('n07056680', 'reel'),
    ('n07583066', 'guacamole'),
    ('n07614500', 'ice cream, icecream'),
    ('n07615774', 'ice lolly, lolly, lollipop, popsicle'),
    ('n07646821', 'goose'),
    ('n07647870', 'drumstick'),
    ('n07657664', 'plate'),
    ('n07695742', 'pretzel'),
    ('n07711569', 'mashed potato'),
    ('n07715103', 'cauliflower'),
    ('n07720875', 'bell pepper'),
    ('n07749582', 'lemon'),
    ('n07753592', 'banana'),
    ('n07768694', 'pomegranate'),
    ('n07871810', 'meat loaf, meatloaf'),
    ('n07873807', 'pizza, pizza pie'),
    ('n07875152', 'potpie'),
    ('n07920052', 'espresso'),
    ('n07975909', 'bee'),
    ('n08496334', 'apron'),
    ('n08620881', 'pole'),
    ('n08742578', 'Chihuahua'),
    ('n09193705', 'alp'),
    ('n09246464', 'cliff, drop, drop-off'),
    ('n09256479', 'coral reef'),
    ('n09332890', 'lakeside, lakeshore'),
    ('n09428293', 'seashore, coast, seacoast, sea-coast'),
    ('n12267677', 'acorn'),
    ('n12520864', 'broom'),
    ('n13001041', 'mushroom'),
    ('n13652335', 'nail'),
    ('n13652994', 'chain'),
    ('n13719102', 'slug'),
    ('n14991210', 'orange')
 ]
