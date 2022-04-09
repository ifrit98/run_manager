import os
import sys
import time
import pickle
import logging
import tensorflow as tf
import matplotlib.pyplot as plt

from .eval import evaluate_model
from .module_utils import plot_metrics, import_flags
from .module_utils import add_lr_to_history_obj, load_and_build_model
from .stream_logger import StreamToLogger


def train(FLAGS=None):
    if FLAGS.get('redirect_stdout', True):
        # Setup stream logger
        logger = logging.getLogger('train')
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler('train_stdout.log')
        fh.setLevel(logging.DEBUG)

        fmt = '%(name)s - %(message)s'
        formatter = logging.Formatter(fmt)
        fh.setFormatter(formatter)

        logger.addHandler(fh)
        sys.stdout = StreamToLogger(logger, logging.DEBUG)

    if FLAGS is None:
        try:
            FLAGS = import_flags()
        except:
            raise ValueError("No flags file found.")

    start = time.time()

    print("Loading training and validation datasets...")
    ds, val_ds = training_datasets(FLAGS)

    if isinstance(ds, list):
        batched_x_shape = ds[0].shape
        batched_y_shape = ds[1].shape
    else:
        for x in ds: break
        batched_x_shape = x[0].shape
        batched_y_shape = x[1].shape
        del x

    print("Loading model...")
    model = load_and_build_model(FLAGS, batched_x_shape, batched_y_shape)

    if FLAGS.get('plot_model', True):
        from tensorflow.keras.utils import plot_model
        tf.keras.utils.plot_model(model, show_shapes=True)
        print("Model block diagram saved to {}/model.png".format(os.getcwd().upper()))
    print("Model created")

    if FLAGS.get('dry_run', False):
        callbacks = []
    else:
        checkpoint_path = FLAGS.get('checkpoint_path', 'model_ckpt/cp.ckpt')
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        checkpt_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            save_weights_only=True,
            monitor=FLAGS.get('monitor', 'val_loss'),
            verbose=FLAGS.get('verbose', True)
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=FLAGS.get('monitor', 'val_loss'),
            factor=FLAGS.get('lr_factor', 0.5),
            patience=FLAGS.get('lr_patience', 3), 
            min_lr=FLAGS.get('min_learning_rate', 1e-6)
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=FLAGS.get('stopping_patience', 5)
        )

        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir='tensorboard',
            histogram_freq=FLAGS.get('histogram_freq', 3),
            profile_batch=FLAGS.get('profile_batch', 0)
        )

        callbacks = [reduce_lr,  checkpt_cb, early_stop, tensorboard_cb]

    print("Callbacks loaded:")
    for cb in callbacks:
        print(cb)

    # TRAINING
    print("Begining training")
    steps_pe = 10 if FLAGS.get('dry_run', False) else FLAGS.get('steps_per_epoch', None)
    epochs = 10 if FLAGS.get('dry_run', False) else FLAGS.get('epochs', 100)
    history = model.fit(
            x=ds[0], y=ds[1],
            validation_data=tuple(val_ds),
            epochs=epochs,
            callbacks=callbacks,
            verbose=FLAGS.get('fit_verbose', 1),
            steps_per_epoch=steps_pe
        ) if isinstance(ds, list) else model.fit(
            ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=FLAGS.get('fit_verbose', 1),
            steps_per_epoch=steps_pe
        )

    print("Training complete!")
    print("History:\n", history.history)

    # Save model, plots, and history objects
    model.save(FLAGS.get('saved_model_path', 'trained_model'))
    print("Model saved")
    print('\nTraining took {} seconds:'.format(int(time.time() - start)))

    hist_path = FLAGS.get('history_path', 'history/model_history')
    hist_dir = os.path.dirname(hist_path)
    if not os.path.exists(hist_dir):
        os.mkdir(hist_dir)
    with open(hist_path, "wb") as f:
        pickle.dump(history.history, f)

    print("Plotting training curves...")
    plot_metrics(history)

    print("Evaluating model...")
    metadata = evaluate_model(model, FLAGS)
    return history, metadata
