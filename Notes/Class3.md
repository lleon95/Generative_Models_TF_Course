# Generative modelling

Types:

* Discriminative
* Generative
* Class

It should meet:

* Generate examples which looks like the training observations.

Applications:

* Denoising
* Painting
* Semi-supervised learning
* Synthetic data generation
* Compression

Examples: GauGAN by Nvidia

The difficulty of using GANs is that they are not easy to evaluate. There are 
not clear metrics to evaluate them and most of them are subjective.

## Representation Learning

It is used for detecting supporting features which represent better the data.
It is used for compression and autoencoders.

High-dimensional -> Just-right-dimensional.

### Autoencoders

They compresses the data.

* Encoder: compresses the data
* Decoder: decompresses the data and tries to reconstruct the original data.

It tries to learn the identity function under some constraints.

**Architecture**

It is a sandwich-like architecture. In the middle, we have the so-called 
"bottleneck".

Hyper-parameters to tune:

* Number of hidden layers
* Type of layers: dense, convolutional, lstm
* Loss function: MSE, MAE, RMSE

** Image reconstruction**

You may consider:

* Use **binary crossentropy** as loss function, considering it is a regression 
problem.
* *Convolution* layers in the *encoder* and *transpose convolution* in the *decoder*.


**Example:**

Autoencoders

```python
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

encoder = Sequential([Dense(2, input_shape=[3])])
decoder = Sequential([Dense(3, input_shape=[2])])
autoencoder = Sequential([encoder, decoder])

autoencoder.compile(loss="mse", optimizer=SGD(1.))
autoencoder.fit(x=data, y=data, epochs=6, verbose=0)

coded_data = encoder.predict(data)
```

Notice: `autoencoder.fit(x=data, y=data, epochs=6, verbose=0)`

The input data and output data are the same!

The output should be similar to the PCA for this example. PCA is a way also 
to compress and detect support features.

For image reconstruction: its architecture is quite similar to the
multilayer perceptron seen before.

```python
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.nn import leaky_relu

def get_dense_autoencoder(latent_dim=30, activation=leaky_relu):
    encoder = Sequential([
        Flatten(input_shape=[28, 28]),
        Dense(100, activation=activation),
        Dense(latent_dim, activation=activation),
    ])
    decoder = Sequential([
        Dense(100, activation=activation, input_shape=[latent_dim]),
        Dense(28 * 28, activation="sigmoid"),
        Reshape([28, 28])
    ])
    return Sequential([encoder, decoder])
```

Look at the encoder/decoder. They are symmetric in the sense.

The images in the example are quite good for 1 layer!

Convolutional layers want a 3D input. It means we need to reshape 2D input to
make explicit the channel and make it pseudo-3D.

`reset_seed()`: helps to preserve some predictability to the random.

### Variational Autoencoders

They are based on probability

* Model generates new instances
* Non-deterministic

From the point of view of the implementation, the are only two parts that need 
to be changed to make an autoencoder variational:

1. The encoder's output.
2. The loss function.

* Encoder's output: Inputs are mapped into a probability distribution. It 
usually is a multivariate Gaussian.

Multivariate Gaussias form a parametric family of distributions, with a mean 
vector and a covariance vector matrix.

In $\mathbb{R}^{\text{latent-dim}}$

For numerical stability, the std is replaced by: $\gamma = log(\sigma^2)$.

The encoder first maps the input to two vectors of mean and modified covariance
($\gamma$) and samples it output from:

$$
N(z|\mu(x), \gamma(x))
$$

Where $N$: Kullback-Liebler

The ultimate goal of variational autoencoders is to make the codings 
distributions more continuous, differently to the conventional autoencoders, 
which may be discontinuous.

Look at the gif:

![](img/VAE.gif)

The variational autoencoders don't only learn how to reconstruct the inputs
but also make some smooth reconstructions.

> The best representation is when the intrinsic dimension is achieved (this
will allow to have the best representation).

> **Symmetry** is a guideline for autoencoders, but it is not mandatory. Some
other topologies can be followed, especially when dealing with non-dense
layer.

For convolutional autoencoders: learning rate of 0.1 is OK.

The number of filters will double as the layers increases:

1 -> 8 -> 16 ...

The latent dimension will have dimension * number of filters.

The greater the number of filter, better is the reconstruction, but the 
latent space increases in its dimensions.

> The Autoencoders are not Generative Models: they are not probabilistics!

Another practice: Use encoders with convolutional layers but ended with a dense
layer to adapt to the desired mapping dimensions, which is now a more tunable 
Hyper-Parameter.

The `Lambda` layer allows to perform a custom operation given by a function.

**Exercise**

Use <= 3 conv layers for VAE.

Third exercise is optional.

## GAN (Generative Adversarial Networks)

There are two networks:

* Generator generates samples intended to resemble the ones in the training 
set. Produces data of the same format of the training data. Similarly to the 
decoder it maps a latent space to the original domain.

* Discriminator inspects a sample and tells if it is real or fake. It is
usually a binary classifier.

### Training

**Discriminator**

We use 50-50 (real-fake) images. We introduce noise to the fake ones.

**Generator**

Freezing the discriminator, we generate the first batch, which will be purely
noise since it is randomly initialise. The generator will take the results 
from the discriminators as a loss parameter and the generator goal is to
reach the 1's from the discrminator.

> In both: generator and discriminator, the binary crossentropy is used, since
what we want is to classify or minimise the number of fakes. So, the generator
receives the feedback from the discriminator. It is like a feed-backed 
topology.

The training:

- Generate fake data by the generator
- Train step to the discriminator
- Turn off the discriminator training
- Train the generator by generating a new data.

Drawback: We don't have any codings on the latent space. We are producing from
the noise the images!

For that reason, the representation is different.

Variational autoencoders are more blurry (more uncertain) about what they are
generating, whereas the GAN seems sharper.

GANs are also used for anomaly detection. Farther from the  center of the 
distribution, more likely to be an anomaly.

### Challenges

1. Oscillating loss
2. Uninformative loss
3. Really sensitive to the hyperparams
4. Mode Collapse: when the generator gets specialised to a mode or class. It
happens with long training. -Tradeoff with training-

Deep:

```
Some guidlines:

- Use batch normalisation in the generator (excluding the last layer)
- Use leaky relu activations where possible (not in the last layers!)
- Replace transpose convolution by upsampling + convolutions (why? check [this article](https://distill.pub/2016/deconv-checkerboard/))
- Use the hyperbolic tangent as last activation function in the generator (This requires rescaling the data between -1 and 1)
```

## Exercises:

- Conv Variational
- Auxiliar
