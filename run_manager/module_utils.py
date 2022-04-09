import os
import time
import yaml
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from typing import GeneratorType

timestamp = lambda: time.strftime("%m_%d_%y_%H-%M-%S", time.strptime(time.asctime()))


class add_lr_to_history_obj(callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if 'lr' not in self.model.history.history.keys():
            self.model.history.history['lr'] = list()
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr)
        self.model.history.history['lr'].append(lr)

def load_and_build_model(FLAGS,
                         batched_x_shape,
                         batched_y_shape):
    if FLAGS.get('src_model_from_file', False):
        path = FLAGS['src_model_path']
        spec = importlib.util.spec_from_file_location("model", path)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        return foo.build_model(FLAGS, batched_x_shape, batched_y_shape)
    src = 'models.{}'.format(FLAGS.get('model_str', './model.py'))
    src += '(FLAGS, batched_x_shape=batched_x_shape, batched_y_shape=batched_y_shape)'
    model = eval(src)
    try:
        print(model.summary())
    except:
        pass
    return model

def restore_checkpoint(FLAGS, 
                       x_shape, y_shape,
                       checkpoint_dir='./model_ckpt', 
                       checkpoint_name='cp.ckpt'):
    """ Restore model from a checkpoint """
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    model = load_and_build_model(FLAGS, x_shape, y_shape)
    model.load_weights(checkpoint_path)
    return model

def inside_docker():
    path = '/proc/self/cgroup'
    x = (
        os.path.exists('/.dockerenv') or \
        os.path.isfile(path) and \
        any('docker' in line for line in open(path))
    )
    return any(list(x)) if isinstance(x, GeneratorType) else x

def import_flags(path=None):
    if path is not None:
        try:
            with open(path, 'r') as f:
                return yaml.load(f, Loader=yaml.SafeLoader)
        except:
            pass
    possible_dirs = ['./', './config', './data', './fr_train']
    for directory in possible_dirs:
        path = os.path.join(directory, 'flags.yaml')    
        FLAGS_FILE = os.path.abspath(path)
        if os.path.exists(FLAGS_FILE):
            with open(FLAGS_FILE, 'r') as f:
                return yaml.load(f, Loader=yaml.SafeLoader)
    raise ValueError("No flags file found.")


def plot_metrics(history, acc='accuracy', loss='loss', 
                 val_acc='val_accuracy', val_loss='val_loss', show=False,
                 save_png=True, outpath='training_curves_' + timestamp()):
    all_keys = [acc, loss, val_acc, val_loss]
    keys = list(history.history)
    idx = np.asarray([k in keys for k in all_keys])
    np.asarray(all_keys)[idx]
    keys = list(history.history)
    epochs = range(len(history.history[keys[0]]))
    plt.figure(figsize=(8,8))
    if acc in keys:
        acc  = history.history[acc]
        plt.subplot(211)
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
    if val_acc in keys:
        val_acc = history.history[val_acc]
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    if loss in keys:
        loss = history.history[loss]
        plt.subplot(212)
        plt.plot(epochs, loss, 'bo', label='Training Loss')
    if val_loss in keys:
        val_loss = history.history[val_loss]
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    if save_png:
        plt.savefig(outpath)
    if show:
        plt.show()