The main goal of this framework is to allow re-running training scripts up to an arbitrary point without incurring the cost of training the model from scratch. This is achieved by saving checkpoint files that behave as cache for re-runs. 

To define a model, initialize a Model class:
```python
# model_builder() should return an nn.Module object
model = train.Model('model-id', model_builder(), 'Model description')
```

Each checkpoint stores in a few files the:
- model parameters
- optimizer state
- epoch scheduler state - a scheduler whose step() function is called per training epoch
- step scheduler state - a scheduler whose step() function is called per training step
- results from various cachable operations

There are two main APIs for checkpoints - the training API and the cache API. 

## The Training API

The training API is invoked using a `with` block:
```python
with model.checkpoint(id=1, description='Adam for 20 + 20 epochs') as cp:
    cp.setup(
        loss_cls=torch.nn.CrossEntropyLoss, 
        optimizer_cls=torch.optim.Adam,
        optimizer_args=dict(lr=5e-4, weight_decay=5e-4),
        epoch_scheduler_cls=torch.optim.lr_scheduler.StepLR, 
        epoch_scheduler_args=dict(step_size=10, gamma=0.2)
    )
    cp.train(train_dl, test_dl, watch='accuracy', load_best=True, epochs=20, metrics=[train.metric_accuracy])
    cp.train(train_dl, test_dl, watch='accuracy', load_best=True, epochs=20, metrics=[train.metric_accuracy])
    cp.plot_metrics()
```

The training API should only be called once for the same checkpoint. Inside the training block, setup() can optionally be called, but no more than once. The setup() function contains instructions for how to instantiate the main components of the training process. When a component is provided, it is instantiated anew, resetting its state. If a component is not provided, the training loop uses the component and its end state from the previous checkpoint, if it exists. 

The train() function trains the model. It can be called multiple times. It is considered a cachable operation, i.e. the result of each train() function is cached based on the order of the train() calls. After the first execution of the checkpoint trainig block, the states of all components is stored, the results of all cachable operations are stored in order, and the state of the model parameters is stored. In subsequent executions, the components will be instantiated and their post-training state loaded, the model parameters will be loaded, and the execution of all cachable operations such as train() will be skipped and their output fetched from cache.

If we ran the example above, then added a third train() call and reran the block, the first two train() calls would retrun immediately and the third train call would be executed as usual. However, if we ran the example above, then changed the setup() call to use SGD instead of Adam and reran the block, we would get an error since the saved state of the Adam optimizer contains parameters SGD does not accept. We would have to delete the checkpoint first using `model.delete_checkpoint(id=1)` and rerun the block.

## The Cache API

The cache API allows running cachable operations under a checkpoint, caching their results in its bundle of files.
```python
model.checkpoint(id=1).find_lr(train_dataloader)
model.checkpoint(id=1).evaluate(test_dataloader)
```

If the same cachable operation is called multiple times, the order of its invocations matter. For each cachable operation, cached results are fetched in the same order. 

As a side note, equivalent functions can be called directly on the model object to avoid caching their results.

## Model and Checkpoint Management

Example of usage:
```python
with model.checkpoint(1) as cp:
    # ... train model

with model.checkpoint(2) as cp:
    # ... train some more

# Rollback to the post-training state of checkpoint 1:
# Reinstantiate the training components, load their states, and load the model parameters 
model.load_checkpoint(id=1)

# Calling a new training block with a different checkpoint ID effectively performs a fork.
# Now checkpoints 2 and 3 are the result of different training approaches from checkpoint 1.
with model.checkpoint(3) as cp:
    # ... train

# Deletes all files asssociated with checkpoint 1
model.delete_checkpoint(1)

# Deletes the model and all its checkpoints
model.delete()
```


---


State Dictionaries:
```python
model1_state_dict = model1.state_dict()     # state dict is retrieved, but gotcha: will continue to change if model1 is trained
model2.load_state_dict(model1_state_dict)   # model2 state gets set. It is now forked from model1, even it model1_state_dict changes
model1.train(...)
model3.load_state_dict(model1_state_dict)   # model3 is now the same as model1 post-training. model2 is like model1 pre-training
```