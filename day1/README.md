# Day 1: Overview of TF2

## Topics:

- Libraries and frameworks for Deep Learning
- Build a model with tf.keras (the easy way)
- Customise models, metrics and training loops
- Visualise statistics with TensorBoard
- DL programming paradigms: Symbolic vs. Imperative Programs
- Backpropagation and AutoDiff

## Install TensorFlow

Tensorflow can be easily installed using `conda` or `pip`:

- `pip` (make sure `python3` is installed)
```
python3 -m venv ~/envs/tf2
source ~/envs/tf2/bin/activate
pip install tensorflow==2.2.0
pip install scikit-learn matplotlib jupyter pydot graphviz
```

- `conda` (usually the latest TensorFlow versions are not immediately available).
```
conda create -n tf2 python=3.7
conda activate tf2
conda install tensorflow=2.0
conda install sklearn matplotlib jupyter pydot graphviz
```