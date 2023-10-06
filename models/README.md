The framework assists in organizing the training code around checkpoints.

A checkpoints encapsulates the storage of states for the following objects:
- Model parameters
- Optimizer
- Epoch scheduler
- Step scheduler

Checkpoints also store the results of each call to the train() method.

A checkpoint file on its own is not useful - it must be accompanied by the code that created it, as that code is responsible for instantiating the objects before a checkpoint can load their states.

A `with model.checkpoint()` block should only be defined once per checkpoint ID. It is not possible to monkey-patch an existing checkpoint by defining a second checkpoint block with the same ID. However, the same block can be run multiple times, and it is possible to add additional `train()` calls in each run. Changing the `setup()` arguments is not recommended without first deleting the checkpoint.


Unlike `train` inside a checkpoint block, `evaluate` and `find_lr` are not retrieved from cache. 
This means that running these will not be instanteneous. I may want to cache these.

```python
with model.checkpoint(1) as cp:
    cp.setup(...)
    cp.train(...)
    cp.plot_metrics()

with model.cache(1) as cache:
    cache.evaluate(...)
    cache.find_lr(...)
```


State Dictionaries:
```python
model1_state_dict = model1.state_dict()     # state dict is retrieved, but gotcha: will continue to change if model1 is trained
model2.load_state_dict(model1_state_dict)   # model2 state gets set. It is now forked from model1, even it model1_state_dict changes
model1.train(...)
model3.load_state_dict(model1_state_dict)   # model3 is now the same as model1 post-training. model2 is like model1 pre-training
```