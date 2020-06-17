# Performance enhancement

## Eager execution

This is the default mode

* It is an imperative model.
* It makes tensorflow to look like numpy. 
* It is easier
to debug. 
* You can print results and values.

You can use Python Control Flow: conditions / loops.

It is possible to create tensors from numpy arrays. Also, it is possible to use
numpy methods and function on tensors.

Even when tracing operations, it is possible to use Python Control Flow 
operations (when you use tape).

> Pythonic

## Graph mode

It is a **declarative environment**. It is like working with a compiler.

Benefits:

* Performance: consumption can be optimised
* Portability: it is language independent. You can use it on Python and C++.

It is slightly more complicated to debug.

In order to integrate a function, you can use the `tf.function`. It converts
from Python function to Tensorflow function.

It is possible to use TensorFlow functions eagerly.

For the generated TensorFlow functions, they are statically typed. It generates
the signatures on-demand. You can get the signature of the concrete function 
with: `.get_concrete_function(args...)`

In theory, each function is a graph of operations. The graph is a 
`concrete_function`, whereas the stages of the graph are the `operations`. 
To get the operations: `.get_operations()`. It returns a list of operations.

There are a couple of operations which are added automatically:

* placeholder: which works as the buffer for the input
* identity: which works as the buffer for the output

You can `print` the properties of each operation.

### Tracing

When the argument has a new signature, it will trace all the operations and 
adapts it according to the signature. In some cases, some operations can be
optimised out of the graph, looking for maximising the performance.

The tracing for the same signature is computed just once. It means that 
executing two times a function with the same signature will lead to only one 
trace but two executions.

### AutoGraph

This option will make the input sensible and branches will be traced when 
tracing. It allows to keep the degrees of freedom.

### Variables

In graph mode, it **is not possible** to create new variables. It is only
allowed when running in eager mode.

Notes:

1. It is good practice to put a batch normalisation after convolution and 
before the activation.
2. The bias is meaningless when batching normalisation.

Observations:

- `train_on_batch`: executes the logic (eager mode)
- `train_step`: graph mode

By doing: `tf_train_Step = tf.function(train_step)` and then execute the
tensor flow function, it reduces a lot the overhead: like 10 times!

You can transform a function into tensorflow function by:

- `@tf.function` decorator
- `tf.function()` method

## GPU support

There are not support for Integer Tensors. Most of the support is for Float32.

You can force the placement of a variable:

```python
with tf.device('/CPU:0'):
  x = tf.Variable([3., 2.], [1., 7.])
x.device
```

Some times, the desired behaviour is not performed. To get errors:

```python
tf.config.set_soft_device_placement(False)
```

> It is not possible to run more than one program that uses the GPU. The first
reserves its memory and the others will complain because no mem is available.

To make GPU devices invisible:

`os.environ["CUDA_VISIBLE_DEVICES"]=""`

Also, you can list the indices of the devices.

## `tf.data`

It provides an API for high-performance data management, facilitating
pipelines. Also it can manage large datasets.

### Dataset

It is the main abstraction. It helps to represent large sequence of elements.

The usages:

* Create a source dataset from your data
* Apply transformations to preprocess the data
* Iterate over the dataset

From tensors `.from_tensor_slices`

From all matching files: `.list_files`. You may need more functions and 
mehtods.

> `-1` dimension is a way to say "automatic"

The `Dataset` object is iterable.

### Transformations

* `map`: applies a function on each element
* `filter`: provided a predicate
* `batch`: returns batches
* `shuffle`: shuffles randomly the elements. Imagine you have a huge deck of 
cards. It works on batches
* `repeat`: make the dataset cyclic

## Multi-offloading

### Layer parallelism

You can split the network by layers and assign one computational unit to each
one.

It doesn't have a good performance, since you have communication, especially 
when forwarding and backward for a single instance during training!

### Vertical split

It is possible to have data parallelism: predicting several instance at a time.

Also, it is possible to split a model into two pieces. It may happen that the
network is partially connected and this is a good option. Albeit, if it is
a dense network with strong connections, this may lead to a huge communication
overhead.

Also, the combination of the two are possible (layer and vertical).

* Having more devices allows increasing the batch size

Remarks:

* Learning rate
* Batch size

Both are *tunable* hyper-parameters 

The strategy for multiple devices during training: focus on batches. When
having the results for each batch, you can collectively retrieve the results
and compute the evolution (update) of your weights.

In Tensorflow: `tf.distribute`:

* `MirroredStrategy`: Synchronous training across multiple replicas on one machine.

* `tf.distribute.experimental.CentralStorageStrategy`, which puts all 
variables on a single device on the same machine (and does sync training).

> Lower batch size might lead to better results!
