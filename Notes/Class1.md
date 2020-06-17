# Overview

2018: PyTorch exploded in contributions and research.

Moving from TF 1.0 to PyTorch:

* Tf was complicated to use
* Debugging was a nightmare
* Tf changed their front-end many times

Why is it still used?

* Most of the problems have been addressed by TF 2
* Better documentation
* Cross-platform: TF.lite, TF.js, TFX

> There are no plans to change the Keras API at the moment.

# Developer

## Sequential API

Using `fashion_mnist`:

* 70K images: 60K for training and 10K for testing. 
* 28x28

```python
from sklearn.model_selection import train_test_split
import numpy as np
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalise the data to avoid unexpected behaviours. Also it allows to use
# Floats for more accuracy
x_train, x_test                      = x_train.astype(np.float32)/255., x_test.astype(np.float32)/255.

# Split
x_train, x_val, y_train, y_val       = train_test_split(x_train, y_train, test_size=5000)
```

### Building the layers

To build a sequential model:

```python
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Sequential
MLP = Sequential([
        Flatten(input_shape=[28, 28]),
        Dense(100, activation="relu"),
        Dense(50, activation="relu"),
        Dense(10, activation="softmax")
])
```

It allows you to list the layers. Each element of the sequential list is a 
layer.

> Flatten: it "unrolls" the image

> `Dense(100, activation="relu")`: 100 is the number of outputs!

It is possible to plot the model:

```python
from tensorflow.keras.utils import plot_model
plot_model(MLP, show_shapes=True)
```

For creating (compile the model) and training:

```python
# Generate the model
MLP.compile(loss='sparse_categorical_crossentropy',
            optimizer='sgd')
# Train
MLP.fit(x=x_train, y=y_train, epochs=3, batch_size=16, validation_data=(x_val, y_val))
```

### Getting the values of hidden layers

```
w, b = hidden_layer.get_weights()
print(type(w), w.shape)
display(w.round(2))
print(type(b), b.shape)
display(b.round(2))
```

It is possible for debugging

### Loss

Output: probabilities

- Binary: classification -> binary cross_entropy. It penalties the low probabilities
- Multi-class classfication: categorical cross_entropy.

> `sparse_categorical_crossentropy`

The target are the indexes, since we have lots of zeros. It is based on
One-Hot Enconding under the hood.

### Training metrics

```python
MLP.compile(loss='sparse_categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy']
            )
```

The `metrics=['accuracy']` helps to print the accuracy when training

You can plot this:

```python
plt.figure(figsize=(10,8))
for k, v in history.history.items():
    plt.plot(v, label=k)
plt.legend()
plt.grid()
plt.xlabel('Epoch')
```

> The validation loss is lower than training loss since the net learns a lot
at the first epoch.

For testing, it is possible to validate to see overfitting

```python
test_loss, test_acc = MLP.evaluate(x_test, y_test)
```

### Exporting/Importing the module

```python
# Export
MLP.save(os.path.join(log_dir,"MLP.h5"))
# Import
new_model = tf.keras.models.load_model(os.path.join(log_dir,"MLP.h5"))
```

### Prediction


`predict method`

```
X = x_test[:20]
y_prob = new_model.predict(X)
```

## Datasets:

They are in:

```python
tf.keras.datasets
```

## Functional API

It is possible to create non-sequential networks with different branches.
It is possible to concatenate and split.

For example:

```python
from tensorflow.keras.layers import Input, Dense, Concatenate

input_1 = Input(shape=[4], name='deep_in')
tower1  = Dense(10, activation="relu")(input_1)
tower1  = Dense(10, activation="relu")(tower1)

input_2 = Input(shape=[3], name='wide_in')
tower2  = Dense(20, activation="relu")(input_2)

concat  = Concatenate()([tower1, tower2])
output  = Dense(1, activation=None)(concat)

multi_in = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
```

![](img/class1_functional.png)

> To plot: `plot_model(multi_in, show_shapes=True)`

This is the basis of Inception Network (GoogLeNet) for object classification.

You need to normalise it to 1

![](img/class1_functionaltraining.png)

The losses are way different when there is overfitting.

### Naming

It is recommended to give names:

```python
inputs_d = Input(shape=[x_train.shape[1] - wide_feats], name='deep_in')
```

### Training

Since you have named the inputs, it is possible to pass a dictionary with
the names of the inputs as keys and the lists as input data. Also, it is
needed to pass the ground truth.

```python
multi_input.compile(loss='mse', optimizer='sgd')
history = multi_input.fit(x={'wide_in': x_train[:,:wide_feats], 'deep_in': x_train[:,wide_feats:]},
                          y=y_train,
                          epochs=30,
                          validation_data=((x_val[:,:wide_feats], x_val[:,wide_feats:]), y_val))
```

The same is for prediction.

Also, it may happen that we have multi-output cases, which are addressed in 
the same way like shown briefly above.

### Auxiliary outputs

Example:

```python
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense

input_ = Input(shape=(28, 28, 1))
x = Conv2D(16, 3, strides=(2,2), padding='same', activation="relu")(input_)
x = Conv2D(32, 3, strides=(2,2),padding='same', activation="relu")(x)

aux_out = Flatten()(x)
aux_out = Dense(10, activation='softmax', name="aux_output")(aux_out)

x = Conv2D(64, 3, strides=(2,2), padding='same', activation="relu")(x)
x = Flatten()(x)
main_out = Dense(10, activation='softmax', name="main_output")(x)

multi_out = tf.keras.Model(inputs=input_, outputs=[main_out, aux_out])
```

For each output, it is required to specify the loss function:

```python
multi_out.compile(loss={"aux_output": 'sparse_categorical_crossentropy',
                        "main_output": 'sparse_categorical_crossentropy'},
                  loss_weights={"aux_output": 0.3, "main_output": 0.7},
                  optimizer="RMSProp",
                  metrics=["accuracy"])
```

It is possible to specify the relevance of the outputs:

```python
loss_weights={"aux_output": 0.3, "main_output": 0.7},
```

For training, you can use the dictionary (recommended) or the tuple approach:

```python
multi_out.fit(x=x_train,
              y=(y_train, y_train),
              batch_size=16,
              epochs=1)
```

It is possible to reconstruct the path of the layers by selecting one of the 
outputs:

```python
single_out = tf.keras.Model(inputs=input_, outputs=main_out)
single_out.summary()
```

It may happen that overengineering the model doesn't lead to better results
or any gain.

### Usage

This approach is really useful for multitasking: predicting several stuff 
recycling stages. Also, for auxiliar outputs for regularisation, like it is
the case of the GoogLeNet. This helps to avoid wrong training in some stages.

Besides, it helps to avoid the *vanishing gradients*, which helps to 
continuous learning, without the risk of disconnecting and so on.

### Preprocessing

Be careful with the input data: numerical inestability:

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train, x_test, x_val = [scaler.transform(x_) for x_ in (x_train, x_test, x_val)]
```

### `BatchNormalization`

It is like a dropout.

## Callbacks

Help to perform some actions when training. It is basically a tool to 
personalise the training process.

It also helps to set checkpoints, when a training goes wrong, you can stop
the training and load the checkpoint. Also, it allows to change the learning 
rate at runtime. Furthermore, you can detect overfitting.

### Checkpoint

To create the callback:

```python
# Define the path of the file that stores the checkpoint
log_dir = os.path.join("logs","slides")
os.makedirs(log_dir, exist_ok=True)
ckpt_file = os.path.join(log_dir,"MLP_ckpt.h5")

# Initialise the callback by passing the path 
cb_model_ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_file)
```

To deploy it:

```python
MLP.fit(x=x_train, y=y_train, epochs=2,
        validation_data=(x_val, y_val), 
        callbacks=[cb_model_ckpt])
```

The checkpoints can be tuned: frequency, conditions (if it improves).

### Early stopping

To create the callback:

```python
cb_early_stop = tf.keras.callbacks.EarlyStopping(
                    # Stop training when `val_loss` is no longer improving
                    monitor='val_loss',
                    # "no longer improving" -> "for at least 2 epochs"
                    patience=3,
                    # restore the weights from the epoch with the best value monitored
                    restore_best_weights=True)
```

* You can specify the monitor: watch the variable of interest. 
* Patience: how many epochs persists.

Also, enabling the switch `restore_best_weights` will restore the weights to a default or to the best.

The process of attaching it is similar to the last one.

> Look at the documentation for more details.

### Learning rate scheduler

You can change the learning rate dinamically according to the number of epoch.

```python
def step_decay(epoch):
    if epoch < 10:
        return 1e-3
    elif epoch < 20:
        return 1e-4
    else:
        return 1e-5

cb_lr_sched = tf.keras.callbacks.LearningRateScheduler(step_decay)
```

> If you start tuning the learning rate, it allows to get new valleys of the 
loss function. The explanation is still a curious phenomenon.

### TensorBoard

It is useful to have the matrix and see the changes.

For visualising:

```bash
tensorboard --logdir="logs/"
```

### `MaxPooling2D` and convolutional layers

Downsamples the output by taking the maximum value within a window.

> padding='same': defines that the output will have the same dimension as the 
input.

> the number of channels is defined by the `n_filters`.

Example of overfitting:

![](img/class1_overfitting.png)

### Dropout regularisation

It is common to place it before the output

### Batch normalisation

> Collects the mean and std. The output is rescaled, its substracted by the mean and divided by std. It happens at the output

It happens batch-wise. It happens on the over all the training set.

They are quite recommended after convolutional layers and before the activation
function (ReLU, eLU, SELU,...)

## Programming style

### Symbolic APIs

> Symbolic is declarative!

The sequential: is symbolic -> which layers to use.

> Drawback: the errors spot at runtime, which may be difficult to debug.

### Imperative APIs

Layer and Model subclassing:

You have a parent model and you can inherite a child model (model subclassing).

A layer class:

* Takes an input tensor or more
* Computation by `call()` defined either in the constructor or the `build()`.

The layer encapsulates: stores the states (weights) and transforms the inputs
into the outputs.

For example, for a `Dense`:

> Keep the `super()` in that order.

```python
class Linear(tf.keras.layers.Layer):    
    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

Look at it. This is the clue for device offloading of the layers.

The `build()` is called once during the first call of the layer. There is a 
verification when calling for the first time, and it checks if the module has 
been already build.

It is used as usual:

```python
from tensorflow.keras.layers import Softmax, ReLU
new_MLP = tf.keras.Sequential([Input((28,28)), Flatten(),
                               Linear(100), ReLU(), Linear(50), ReLU(),
                               Linear(10), Softmax()])
new_MLP(x_train[:1])
```

This approach is also used for creating blocks of layers.

For example:

```python
class MLPBlock(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Here, we define the layers needed
        self.linear_1 = Linear(100)
        self.linear_2 = Linear(50)
        self.linear_3 = Linear(10)

    def call(self, inputs):
        # This is the functional way of building the model
        x = self.linear_1(inputs)
        x = tf.nn.relu(x) # This adds the activation
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)

# The model is now compacted. 
new_MLP = tf.keras.Sequential([Input((28,28)), Flatten(), MLPBlock(), Softmax()])
```

The compated models are called in the same fashion as a normal layer.

> The layers are recursively composable! It means that it goes from the
back to the top.

Also, you can create custom models. They are built in a similar way to the 
block of layers.

* Exposes a `.fit()`, `.evaluate()`, `.predict()`
* Exposes the list of layers via `.layers`
* Exposes saving and serialisation APIs

It is not a good practice composition of models.

### Automatic differentiation

It is composed by two steps:

* Forward
* Backward

> The gradient is computed on the loss function as a function of the weights.

TensorFlow use reverse mode autodiff. It is like a FIFO of operations
that is pushed when going forward and popped when going backwards.

TO use the autodiff, TensorFlow provides the `tf.GradientTape`.

Example:

```python
A, B = tf.Variable(1.), tf.Variable(2.)
f = lambda x, y: 3 * x + x * y + 1

with tf.GradientTape() as tape:
    Z = f(A, B)

gradients = tape.gradient(Z, [A,B])

print('Z:', Z)
print('dZ/dA:', gradients[0])
print('dZ/dB:', gradients[1])
```

Output

``` 
Z: tf.Tensor(6.0, shape=(), dtype=float32)
dZ/dA: tf.Tensor(5.0, shape=(), dtype=float32)
dZ/dB: tf.Tensor(1.0, shape=(), dtype=float32)
```

The requirement:

The variables must be declared as tf. Remember it is like a FIFO or
automatically garbage collected.

### Custom training step

We can specify our training procedure

1. Compute the loss function of a mini-batch
2. Collect the gradients
3. Update the weights

After that, it is possible to define the complete training, which will be 
composed by two inner loops:

1. The epochs loop: outtest
2. The batch loop: innest

The innest loop will call the training step.

Defining the required API for tensorboard helps to collect statistics ->
look at the slides.