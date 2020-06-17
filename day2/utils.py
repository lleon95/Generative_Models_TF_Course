import tensorflow as tf
from time import perf_counter

class StepTiming(tf.keras.callbacks.Callback):
    """Timer Callback
    
    The time spent in each batch is recorded in a tf.summary.
    
    # Arguments
        log_dir: Path to save the tf.summary.
        init_step: define starting step (default 0).
        step_mean_only: Report only the step mean time per epoch. 
    """
    def __init__(self, log_dir, init_step=0, step_mean_only=True):
        self.file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
        self.step = init_step
        self.mean_only = step_mean_only
        
    def on_epoch_begin(self, epoch, logs={}):
        self.logs = []
        self.epoch_start = perf_counter()
        
    def on_epoch_end(self, epoch, logs={}):
        with self.file_writer.as_default():
            tf.summary.scalar('Mean step time', data=tf.reduce_mean(self.logs), step=epoch)
            tf.summary.scalar('Epoch time', data=perf_counter()-self.epoch_start, step=epoch)
            if not self.mean_only:
                for i, t in enumerate(self.logs):
                    tf.summary.scalar('Step time', data=t, step=self.step+i)
        self.step += len(self.logs)
        self.logs = []

    def on_batch_begin(self, batch, logs={}):
        self.batch_start = perf_counter()
        
    def on_batch_end(self, batch, logs={}):
        self.logs.append(perf_counter() - self.batch_start)

def exponential_decay_fn(lr0=1e-3, decay_rate=10):      
    def exponential_decay(epoch):
        return lr0 * 0.1**(epoch/decay_rate)
    return exponential_decay

def step_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr