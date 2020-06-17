import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def reset_seed():
    tf.random.set_seed(0)
    np.random.seed(0)

def display_compare(*args, Cols=10, names=None):
    Rows = len(args) + 1
    fig = plt.figure(figsize=(2*Cols, 2*Rows))
    for i, *replicas in zip(range(Cols), *args):
        for j, img in enumerate(replicas):
            ax = fig.add_subplot(Rows, Cols, j*Cols+i+1)
            ax.imshow(img), ax.axis('off')
            if names: ax.set_title(names[j])
    plt.show