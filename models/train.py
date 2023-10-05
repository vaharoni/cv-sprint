""" 
The goal of the registry is to ensure that Jupyter notebooks can be run repeatedly without incurring
the cost of re-training. We should be able to replay the entire notebook, or up to a certain point,
without spending any time on training - just by loading states from checkpoints. 

When we perform a training process, the model, optimizer, and learning schedules hold states.
Their state before their training is different than their state after the training.
This means that there are two behaviors we need to allow:
- First run: we need the calling code to build all relevant objects in their initial state.
- Subsequent runs: while the calling code can (and often should) build the objects, we need to load 
  all object states from the checkpoint so we can continue where we left off. 

This can be achieved with something like the following.

Usage:

# my_namespace is used as the root folder
import train
train.set_namespace('my_namespace')

# The model id (model1) is used as the folder where all the model's checkpoints are stored.
model = Model(id='model1', model=build_model(), description='Model description')

with model.checkpoint(id='1.1', description='First cycle') as c:
    # In the first run, c is uninitialized
    # After the first run, c and the model get loaded with all relevant object states
    
    # The setup() method is ignored in all runs except the first.
    # These objects are stored on the model level, and if setup() is not called they are taken from there. 
    # This allows continuing the training process using a second checkpoint from where we left off. 
    # It is required to call setup() during the first checkpoint. 
    c.setup(        
        loss_cls=torch.nn.CrossEntropyLoss,
        optimizer_cls=torch.optim.Adam, 
        optimizer_args=dict(weight_decay=5e-4, lr=0.001),
        loss_fn_cls=torch.nn.CrossEntropyLoss, 
        epoch_scheduler_cls=torch.optim.lr_scheduler.StepLR, 
        epoch_scheduler_args=dict(step_size=40, gamma=0.2)        
    )

    # When the first run concludes, all states are stored as well as the output hist
    # After the first run, hist is loaded from the checkpoint
    hist = c.train(train_dl, test_dl, epochs=160, metrics=[train.metric_accuracy], watch='val_accuracy')

    # Helpful utilities
    c.plot_metrics(hist)

# Revert the model to any checkpoint
model.load_checkpoint(id='1.1')

# Create a new model in memory based on all object states of model
model2 = model.fork(id='model2', description='Going to try something new')
with model2.checkpoint(id=1) as c:
    # Here we are overriding just the optimizer. The class and states of the schedulers and loss functions remain 
    # the same as in model1, but they are distinct objects in memory due to the fork.
    c.setup(
        optimizer_cls=torch.optim.SGD, 
        optimizer_args=dict(weight_decay=5e-4, lr=0.1, momentum=0.9),    
    )
    c.train(train_dl, test_dl, epochs=160, metrics=[train.metric_accuracy])

# Delete one checkpoint
model2.delete_checkpoint(id=1)

# All checkpoints are deleted
model2.delete()

---
setup() args:
    Loss factory:
        loss_cls=torch.nn.CrossEntropyLoss,

    Optimizer factory, option 1:
        optimizer_cls=torch.optim.Adam
        optimizer_args=dict(weight_decay=5e-4, lr=0.001)

    Optimizer factory, option 2:
        optimizer_fn=lambda params: torch.optim.Adam(params, weight_decay=54-4, lr=0.001)

    Epoch scheduler, option 1:
        epoch_scheduler_cls=torch.optim.lr_scheduler.StepLR
        epoch_scheduler_args=dict(step_size=40, gamma=0.2)

    Epoch scheduler, option 2:
        epoch_scheduler_fn=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)

    Step scheduler, option 1:
        step_scheduler_cls=torch.optim.lr_scheduler.StepLR
        step_scheduler_args=dict(step_size=40, gamma=0.2)

    Step scheduler, option 2:
        step_scheduler_fn=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)
"""

from contextlib import contextmanager
from typing import Literal
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

root_folder = Path('..')
model_checkpoint_sep = '-'
noetbook_namespace = None
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def set_root_folder(root):
    global root_folder
    root_folder = Path(root)

def set_namespace(namespace):
    global noetbook_namespace
    noetbook_namespace = namespace

def set_model_checkpoint_sep(sep):
    global model_checkpoint_sep
    model_checkpoint_sep = str(sep)

def _ensure_namespace():
    if not noetbook_namespace:
        raise Exception('Must call train.set_namespace()')
    return True

def _ensure_no_space(id):
    if ' ' in id: 
        raise Exception(f'ID must not contain spaces (received "{id}")')
    return str(id)

_metric_factors = { 'loss': 1 }
def _metric(name, higher_is_better=True):
    """Decorator args:
        name - the metric name. Will be attached to a prefix: train_{name} or val_{name}
        higher_is_better - whether higher or lower value of the metric is desirable.
    """
    def decorator(fn):
        fn.name = name
        fn.factor = -1 if higher_is_better else 1
        _metric_factors[name] = fn.factor
        return fn
    return decorator

# = Training helpers

# Metrics should return: [name, factor, value]. 
# Factor is used to determine whether higher or lower value of the metric is desirable.  
# It should be either 1 (minimize) or -1 (maximize)

@_metric('accuracy')
def metric_accuracy(y_pred, y_true):
    return torch.sum(torch.argmax(y_pred, dim=-1) == y_true).item()

class EarlyStopException(Exception): pass

# = Main class

# This is built with the idea of a callback. The constructor can take anything, while the hooks take a sensible API.
class MetricsManager:
    def __init__(self, 
                 metrics: list[callable], 
                 watch='loss', 
                 train_dataloader: DataLoader | None = None, 
                 val_dataloader: DataLoader | None = None, 
                 tensorboard_dir=None):
        self.history = defaultdict(lambda: [])
        self.metrics = metrics
        self.dataloaders = { 'train': train_dataloader, 'val': val_dataloader }

        os.makedirs(tensorboard_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(tensorboard_dir)

        self.watch = watch
        self.watched_metric_factor = _metric_factors[self.watch]
        self.is_best_watched_metric_epoch = None
        self.best_watched_metric_epoch = None
        self.best_watched_metric = None
        self.global_epoch_start = None

    def log(self, msg):
        self.history['log'].append(msg)
        print(msg)

    def start_training(self, model: 'Model', epochs):
        self.global_epoch_start = model.epoch
        self.epochs = epochs
        self.best_watched_metric_epoch = max(self.global_epoch_start - 1, 0)
        mean_metrics = model.evaluate(self.dataloaders['val'], metrics=self.metrics)
        self.best_watched_metric = mean_metrics[self.watch]

        self.mean_metrics = { f'val_{metric_name}': metric_val for metric_name, metric_val in mean_metrics.items() }
        metrics_str = ", ".join([f'val_{metric_name}={metric_val:.4f}' for metric_name, metric_val in mean_metrics.items()])
        self.log(f'Initial {metrics_str}')

        if self.global_epoch_start == 0:
            inputs, _ = next(iter(self.dataloaders['val']))
            self.tb_writer.add_graph(model.model, inputs.to(device))
            self.tb_writer.flush()

    def start_step(self, phase, local_step):
        pass
        
    def start_epoch(self, local_epoch):
        self.is_best_watched_metric_epoch = False
        self.running_metrics = { 'train': defaultdict(lambda: 0), 'val': defaultdict(lambda: 0) }

    def end_step(self, phase, local_step, total_loss, y_pred, y_true):
        self.running_metrics[phase]['loss'] += total_loss
        for metric in self.metrics:
            metric_val = metric(y_pred, y_true)
            self.running_metrics[phase][metric.name] += metric_val

    def end_epoch(self, local_epoch):
        self.mean_metrics = { f"{phase}_{metric_name}": self.running_metrics[phase][metric_name] / len(self.dataloaders[phase].dataset)
                                for phase in self.running_metrics  
                                    for metric_name in self.running_metrics[phase] }

        # Print loss and metrics
        self.log(f"Epoch {local_epoch + 1}/{self.epochs}: " + ", ".join([f"{metric_name}={self.mean_metrics[metric_name]:.4f}" for metric_name in self.mean_metrics]))

        # Store metrics in history
        self.history['epoch'].append(self.global_epoch_start + local_epoch + 1)
        for full_metric_name in self.mean_metrics:
            self.history[full_metric_name].append(self.mean_metrics[full_metric_name])

        # Report to tensorboard
        if self.tb_writer is not None:
            tb_metrics = defaultdict(lambda: dict())
            for full_metric_name in self.mean_metrics:
                phase, *metric_name = full_metric_name.split('_')
                metric_name = "_".join(metric_name)
                tb_metrics[metric_name][phase] = self.mean_metrics[full_metric_name]
            for metric_name in tb_metrics:
                self.tb_writer.add_scalars(metric_name, tb_metrics[metric_name], self.global_epoch_start + local_epoch + 1)
            self.tb_writer.flush()

        # Check if we exceeded the best watched metric
        if self.mean_metrics[f'val_{self.watch}'] * self.watched_metric_factor < self.best_watched_metric * self.watched_metric_factor:
            self.is_best_watched_metric_epoch = True
            self.best_watched_metric_epoch = self.global_epoch_start + local_epoch
            self.best_watched_metric = self.mean_metrics[f'val_{self.watch}']

        

class Model:
    def __init__(self, id, model: torch.nn.Module, description=None):
        _ensure_namespace()
        self.id = _ensure_no_space(id)
        self.description = description
        self.model = model.to(device)
        self.optimizer: torch.optim.Optimizer | None = None
        self.loss_fn = None
        self.epoch_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.step_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.step = 0
        self.epoch = 0
        self.last_checkpoint = None
        self.print_params(False)

    def tensorboard_path(self) -> Path:
        return root_folder / 'runs' / noetbook_namespace / self.id
    
    def checkpoint_path(self) -> Path:
        return root_folder / 'ckpts' / noetbook_namespace / self.id

    def build_loss(self, loss_cls=None):
        if loss_cls:
            self.loss_fn = loss_cls()

    def build_optimizer(self, optimizer_cls=None, optimizer_args=None, optimizer_fn=None):
        if optimizer_fn:
            self.optimizer = optimizer_fn(self.model.parameters())
        elif optimizer_cls or optimizer_args:
            optimizer_args = optimizer_args if optimizer_args else dict()
            self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)

    def build_epoch_scheduler(self, epoch_scheduler_cls=None, epoch_scheduler_args=None, epoch_scheduler_fn=None):
        if epoch_scheduler_fn:
            self.epoch_scheduler = epoch_scheduler_fn(self.optimizer)
        elif epoch_scheduler_cls or epoch_scheduler_args:
            epoch_scheduler_args = epoch_scheduler_args if epoch_scheduler_args else dict()
            self.epoch_scheduler = epoch_scheduler_cls(self.optimizer, **epoch_scheduler_args)

    def build_step_scheduler(self, step_scheduler_cls=None, step_scheduler_args=None, step_scheduler_fn=None):
        if step_scheduler_fn:
            self.epoch_scheduler = step_scheduler_fn(self.optimizer)
        elif  step_scheduler_cls or step_scheduler_args:
            step_scheduler_args = step_scheduler_args if step_scheduler_args else dict()
            self.epoch_scheduler = step_scheduler_cls(self.optimizer, **step_scheduler_args)

    def set_epoch_and_step(self, global_epoch, global_step):
        self.epoch = global_epoch
        self.step = global_step

    def serialize(self):
        data = {
            'model_id': self.id, 
            'model_description': self.description,
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
        }
        if self.optimizer:
            data['optimizer_state_dict'] = self.optimizer.state_dict()
        if self.epoch_scheduler:
            data['epoch_scheduler_state_dict'] = self.epoch_scheduler.state_dict()
        if self.step_scheduler:
            data['step_scheduler_state_dict'] = self.step_scheduler.state_dict()
        return data
    
    def deserialize(self, data):
        if not self.description:
            self.description = data['model_description']
        self.step = data['step']
        self.epoch = data['epoch']
        self.model.load_state_dict(data['model_state_dict'])
        if data.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(data['optimizer_state_dict'])
        if data.get('epoch_scheduler_state_dict'):
            self.epoch_scheduler.load_state_dict(data['epoch_scheduler_state_dict'])
        if data.get('step_scheduler_state_dict'):
            self.step_scheduler.load_state_dict(data['step_scheduler_state_dict'])

    @contextmanager
    def checkpoint(self, id, description=None):
        cp = CheckpointLifecycle(model=self, id=id, description=description)
        self.last_checkpoint = cp
        yield cp.api

    def load_checkpoint(self, id, from_backup=False):
        cp = CheckpointLifecycle(model=self, id=id)
        self.last_checkpoint = cp
        cp.load(backup=from_backup)
        return self

    def delete_checkpoint(self, id):
        cp = self.last_checkpoint if self.last_checkpoint and self.last_checkpoint.id == id else CheckpointLifecycle(model=self, id=id)
        cp.delete()

    def delete(self):
        shutil.rmtree(self.tensorboard_path(), ignore_errors=True)
        shutil.rmtree(self.checkpoint_path(), ignore_errors=True)

    def train(self, 
              train_dataloader: DataLoader, 
              val_dataloader: DataLoader, 
              watch: str = 'loss', 
              load_best: bool = True,
              epochs: int = 10, 
              patience: int = 0, 
              warmup: int = 1, 
              metrics: list[callable] = [],
              tensorboard_dir: str = None,
              save_checkpoint_fn: callable = None,
              save_hist_fn: callable = None,            # Separate from checkpoint, since we want to record the full history even it we revert to best prior state
              load_best_fn: callable = None):

        dataloaders = { 'train': train_dataloader, 'val': val_dataloader }

        metrics_manager = MetricsManager(metrics=metrics, train_dataloader=train_dataloader, val_dataloader=val_dataloader, watch=watch, tensorboard_dir=tensorboard_dir)
        metrics_manager.start_training(model=self, epochs=epochs)

        global_epoch_start = self.epoch
        global_step_start = self.step
        local_step = 0

        # Creating a checkpoint file prior to training so that the file structure is always consistent with the post-trained model in memory,
        # and so that the load_best_fn() reverts to it in case no improvement is observed during training and load_best is True
        if save_checkpoint_fn:
            save_checkpoint_fn(global_epoch_start, global_step_start, metrics_manager.mean_metrics, make_backup=False, logger=metrics_manager.log)

        steps_per_epoch = len(train_dataloader)
        warmup_scheduler = None

        # TODO: change this to an actual WarmUp scheduler
        if warmup > 0 and warmup > global_epoch_start:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1e-3, total_iters=steps_per_epoch * warmup)

        try: 
            for local_epoch in range(epochs):
                metrics_manager.start_epoch(local_epoch)

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
                            loss: torch.Tensor = self.loss_fn(y_pred, y_true)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()
                                if warmup_scheduler and warmup > global_epoch_start + local_epoch:
                                    warmup_scheduler.step()

                        metrics_manager.end_step(phase=phase, local_step=local_step, total_loss=loss.item() * inputs.shape[0], y_pred=y_pred, y_true=y_true)

                        if phase == 'train':
                            local_step += 1
                            # Advance the step LR scheduler
                            if self.step_scheduler is not None and warmup <= global_epoch_start + local_epoch:
                                self.step_scheduler.step()

                    # Advance the epoch LR scheduler
                    if phase == 'train' and self.epoch_scheduler is not None and warmup <= global_epoch_start + local_epoch:
                        self.epoch_scheduler.step()

                metrics_manager.end_epoch(local_epoch=local_epoch)

                if save_hist_fn:
                    save_hist_fn(metrics_manager.history)

                if metrics_manager.is_best_watched_metric_epoch and save_checkpoint_fn:
                    save_checkpoint_fn(global_epoch_start + local_epoch + 1, global_step_start + local_step, metrics_manager.mean_metrics, make_backup=False, logger=metrics_manager.log)

                if patience > 0 and global_epoch_start + local_epoch - metrics_manager.best_watched_metric_epoch >= patience:
                    raise EarlyStopException
                
        except EarlyStopException:
            pass

        if load_best:
            if load_best_fn:
                load_best_fn(logger=metrics_manager.log)
        else:
            if save_checkpoint_fn and not metrics_manager.is_best_watched_metric_epoch:
                save_checkpoint_fn(global_epoch_start + local_epoch + 1, global_step_start + local_step, metrics_manager.mean_metrics, make_backup=True, logger=metrics_manager.log)

        if save_hist_fn:
            save_hist_fn(metrics_manager.history)
                
        return metrics_manager.history


    def evaluate(self, dataloader: DataLoader, metrics: list[callable] = []):
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
                    metric_val = metric(y_pred, y_true)
                    running_metrics[metric.name] += metric_val
            
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


class CheckpointLifecycle:
    def __init__(self, model: Model, id, description=None):
        self.model = model
        self.id = str(id).zfill(4) if isinstance(id, int) else _ensure_no_space(id)
        self.description = description
        self.persisted = False
        self.metrics = {}
        self.histories = []
        self.api = CheckpointAPI(self)

    def serialize(self):
        data = {
            **self.model.serialize(),
            'checkpoint_id': self.id,
            'checkpoint_description': self.description,
            'checkpoint_metrics': self.metrics
        }
        return data

    def deserialize(self, data):
        self.model.deserialize(data)
        if not self.description:
            self.description = data['checkpoint_description']
        self.metrics = data['checkpoint_metrics']

    def tensorboard_path(self) -> Path:
        return self.model.tensorboard_path() / self.id
    
    def checkpoint_path(self, backup=False) -> Path:
        version = 'best' if backup else 'last'
        return self.model.checkpoint_path() / f'{self.id + model_checkpoint_sep + version}.pt'
    
    def hist_path(self) -> Path:
        return self.model.checkpoint_path() / f'{self.id}.hist'

    # The knowledge of whether a backup exists is only present in the calling code. There is no trace for such information in the data stored.
    def load(self, backup=False, logger=None):
        path = self.checkpoint_path(backup=backup)
        if not path.exists(): 
            return False

        data = torch.load(path)
        self.deserialize(data)
        self.persisted = True

        if logger:
            loaded_str = f"Loaded model {self.model.id} from checkpoint {self.id}. epoch={self.model.epoch}, step={self.model.step}."
            metrics_str = ", ".join([f'{metric_name}={metric_val:.4f}' for metric_name, metric_val in self.metrics.items()])
            logger(" ".join([x for x in [loaded_str, metrics_str] if x]))

        return True

    def save(self, global_epoch, global_step, metrics, make_backup=False, logger=None):
        save_path = self.checkpoint_path(backup=False)
        os.makedirs(save_path.parent, exist_ok=True)

        if self.persisted and make_backup:
            backup_path = self.checkpoint_path(backup=True)
            os.rename(save_path, backup_path)
            logger(f"Created backup for model {self.model.id} checkpoint {self.id}.")

        self.model.set_epoch_and_step(global_epoch, global_step)
        self.metrics = dict(metrics)
        data = self.serialize()
        torch.save(data, save_path)
        self.persisted = True

        if logger:
            saved_str = f"Saved model {self.model.id} checkpoint {self.id}. epoch={global_epoch}, step={global_step}."
            metrics_str = ", ".join([f'{metric_name}={metric_val:.4f}' for metric_name, metric_val in self.metrics.items()])
            logger(" ".join([x for x in [saved_str, metrics_str] if x]))

        return True
    
    def load_history(self):
        hist_path = self.hist_path()
        if not hist_path.exists():
            return False
        self.histories = torch.load(hist_path)
        return True

    def save_history(self):
        hist_path = self.hist_path()
        os.makedirs(hist_path.parent, exist_ok=True)
        torch.save(self.histories, hist_path)
        return True
    
    def set_history(self, index, hist):
        if index < len(self.histories):
            self.histories[index] = dict(hist)
        elif index == len(self.histories):
            self.histories.append(dict(hist))
        else:
            raise Exception(f'Cannot set history of length {len(self.histories)} at index {index}')
        
    def combined_history(self):
        padding = 0
        hist_dict = defaultdict(lambda: [None] * padding)
        for hist in self.histories:
            for key, values in hist.items():
                hist_dict[key] += values
            padding += len(hist['epoch'])
        return dict(hist_dict)

    def delete(self):
        backup_path = self.checkpoint_path(backup=True)
        if backup_path.exists():
            os.remove(backup_path)

        save_path = self.checkpoint_path(backup=False)
        if save_path.exists():
            os.remove(save_path)

        hist_path = self.hist_path()
        if hist_path.exists():
            os.remove(hist_path)

        self.persisted = False

    
class CheckpointAPI:
    def __init__(self, cp_lifecycle: CheckpointLifecycle):
        self._cp_lifecycle = cp_lifecycle
        self._train_method_call_count = 0

    def setup(self, 
              loss_cls=None, 
              optimizer_cls=None, optimizer_args=None, optimizer_fn=None, 
              epoch_scheduler_cls=None, epoch_scheduler_args=None, epoch_scheduler_fn=None,
              step_scheduler_cls=None, step_scheduler_args=None, step_scheduler_fn=None):
        """args:
            Loss factory:
                loss_cls=torch.nn.CrossEntropyLoss,

            Optimizer factory, option 1:
                optimizer_cls=torch.optim.Adam
                optimizer_args=dict(weight_decay=5e-4, lr=0.001)

            Optimizer factory, option 2:
                optimizer_fn=lambda params: torch.optim.Adam(params, weight_decay=54-4, lr=0.001)

            Epoch scheduler, option 1:
                epoch_scheduler_cls=torch.optim.lr_scheduler.StepLR
                epoch_scheduler_args=dict(step_size=40, gamma=0.2)

            Epoch scheduler, option 2:
                epoch_scheduler_fn=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)

            Step scheduler, option 1:
                step_scheduler_cls=torch.optim.lr_scheduler.StepLR
                step_scheduler_args=dict(step_size=40, gamma=0.2)

            Step scheduler, option 2:
                step_scheduler_fn=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.2)
        """         
        self._cp_lifecycle.model.build_loss(loss_cls)
        self._cp_lifecycle.model.build_optimizer(optimizer_cls, optimizer_args, optimizer_fn)
        self._cp_lifecycle.model.build_epoch_scheduler(epoch_scheduler_cls, epoch_scheduler_args, epoch_scheduler_fn)
        self._cp_lifecycle.model.build_step_scheduler(step_scheduler_cls, step_scheduler_args, step_scheduler_fn)
        return True


    def train(self, 
              train_dataloader: DataLoader, val_dataloader: DataLoader, 
              watch: str = 'loss', load_best: bool = True,
              epochs: int = 10, 
              patience: int = 0, 
              warmup: int = 1, 
              metrics: list[callable] = []):
        
        if self._train_method_call_count == 0:
            self._cp_lifecycle.load()
            self._cp_lifecycle.load_history()

        self._train_method_call_count += 1

        if self._train_method_call_count <= len(self._cp_lifecycle.histories):
            hist = self._cp_lifecycle.histories[self._train_method_call_count - 1]
            for msg in hist['log']:
                print(msg)
            return hist

        tensorboard_dir = self._cp_lifecycle.tensorboard_path()

        def save_checkpoint_fn(*args, **opts):
            self._cp_lifecycle.save(*args, **opts)

        def save_hist_fn(hist):
            self._cp_lifecycle.set_history(self._train_method_call_count - 1, hist)
            self._cp_lifecycle.save_history()

        def load_best_fn(logger=None):
            self._cp_lifecycle.load(backup=False, logger=logger)

        return self._cp_lifecycle.model.train(train_dataloader=train_dataloader, 
                                              val_dataloader=val_dataloader, 
                                              watch=watch, 
                                              load_best=load_best, 
                                              epochs=epochs, 
                                              patience=patience, 
                                              warmup=warmup, 
                                              metrics=metrics, 
                                              tensorboard_dir=tensorboard_dir, 
                                              save_checkpoint_fn=save_checkpoint_fn,
                                              save_hist_fn=save_hist_fn,
                                              load_best_fn=load_best_fn)

    def evaluate(self, *args, **opts):
        return self._cp_lifecycle.model.evaluate(*args, **opts)
    
    def plot_metrics(self, hist=None):
        hist = hist if hist else self._cp_lifecycle.combined_history()
        plot_metrics(hist)

def plot_metrics(hist):
    x = hist['epoch']
    y = defaultdict(lambda: {})
    for metric_name in hist:
        if metric_name in ['epoch', 'log']:
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
