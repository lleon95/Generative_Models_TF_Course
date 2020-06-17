import numpy as np
import matplotlib.pyplot as plt


def display_fashion_mnist(x,y):
    labels_map = np.array(['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
    display_gallery(x, labels_map[y])

def display_gallery(images, labels, n_rows=2):
    fig = plt.figure(figsize=(16,2*n_rows))
    for i, (x, y) in enumerate(zip(images, labels)):
        ax = fig.add_subplot(n_rows, np.ceil(len(images)/n_rows), i + 1)
        plt.imshow(x)
        plt.axis('off')
        ax.set_title(y)
    plt.show()