import os
import time
import yaml
import glob

from types import GeneratorType
from pprint import pformat, pprint
import matplotlib.pyplot as plt

from stream_logger import file_logger
# from train import train

import importlib

def load_module(path):
        spec = importlib.util.spec_from_file_location(path)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        return foo

def execute_code(path):
    with open(path) as f:
        return exec(f.read())

timestamp = lambda: time.strftime("%m_%d_%y_%H-%M-%S", time.strptime(time.asctime()))

#Defaults for legible figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams["image.cmap"] = 'jet'


HPARAM_DEFAULTS = {
    "epochs": 10,
    "steps_per_epoch": 100,
    "units": 128,
    "batch_size": 32,
    "init_learning_rate": 1e-3,
    "max_learning_rate": 1,
    "min_learning_rate": 1e-6,
    "buffer_size": 100,
    "verbose": False
}

_globals = {
    'runs_dir': None,
    'run_dir': {
        'path': None,
        # 'FLAGS': None
    }
}

HISTORY = None
LOGGER = None


def inside_docker():
    path = '/proc/self/cgroup'
    x = (
        os.path.exists('/.dockerenv') or \
        os.path.isfile(path) and \
        any('docker' in line for line in open(path))
    )
    return any(list(x)) if isinstance(x, GeneratorType) else x

def clear_run():
    """
    Clears `_globals` dictionary by setting all to `None`
    """
    _globals['runs_dir'] = None
    _globals['run_dir']['path'] = None
    # _globals['run_dir']['FLAGS'] = None

def unique_run_dir(runs_dir = None, format_="%m_%d_%y_%H-%M-%S"):
    """Returns a unique run directory filepath to house an experiment"""
    runs_dir = runs_dir or _globals['runs_dir']
    run_dir = time.strftime(format_, time.strptime(time.asctime()))
    return os.path.join(runs_dir, run_dir)

def is_run_active():
    return _globals['run_dir']['path'] is None

def run_dir():
    return _globals['run_dir']['path'] if is_run_active() else os.getcwd()


def do_training_run(run_dir,
                    meta_file = 'metadata.json',
                    logfile='stdout.log'):
    """
    Perform the training run with current run_dir. Sets cwd to run_dir, and then 
    executes training.  Logs created by redirecting stdout to `logfile`.

    Clears cache and returns to original working directory before returning.

    Args: 
        run_dir: String path to current run directory.
        meta_file: json filepath to metadata output file for dumping `globals`. 
          Defaults to `metadata.json`
        logfile: String filepath to desired logfile. Defaults to stdout.log
    
    Returns:
        None
    """
    global LOGGER

    _globals['start_time'] = timestamp()
    LOGGER.info('Start time: {}'.format(_globals['start_time']))
    LOGGER.info("Using run directory: {}".format(run_dir))
    owd = os.getcwd()
    os.chdir(run_dir)

    LOGGER.info("Executing training...")
    history, eval_metrics = train()

    epochs = _globals['run_dir']['FLAGS'].get('epochs', 0)

    LOGGER.info("History:\n{}".format(pformat(history)))

    # Record end time
    _globals['end_time'] = timestamp()
    LOGGER.info('End time: {}'.format(_globals['end_time']))

    # Save and log results
    import json
    with open(meta_file, 'w') as f:
      json.dump(_globals,  f)

    LOGGER.info('_globals\n: {}'.format(pformat(_globals)))

    clear_run()
    os.chdir(owd)



def initialize_run(run_dir=None,
                   logger_name='init_log',
                   FLAGS=None):
    """
    Initializes training run variables.

    Args:
        run_dir: String path to current run directory.
        flags: flags object, as a python dictionary from yaml file.

    """
    if _globals['runs_dir'] is None:
        _globals['runs_dir'] = os.path.abspath('/training/data/runs')
    
    print("Runs_dir:", _globals['runs_dir'])
    if not os.path.exists(_globals['runs_dir']):
        os.mkdir(_globals['runs_dir'])

    if run_dir is None:
        run_dir = unique_run_dir()
    print("run_dir:", run_dir)

    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    global LOGGER
    LOGGER = file_logger(os.path.join(run_dir, logger_name))

    # Copy py files over
    pyfiles = glob.glob("*.py")
    for pyfile in pyfiles:
        dest = os.path.join(run_dir, pyfile)
        cmd = "cp ./{} {}".format(pyfile, dest)
        os.system(cmd)

    _globals['run_dir']['path'] = run_dir

    LOGGER.info("Initialized run_dir {}".format(run_dir))
    LOGGER.info("Globals:\n{}".format(pformat(_globals)))

    return run_dir


def training_run(FLAGS=None,
                 run_dir=None,
                 runs_dir=None,
                 encoding='utf-8'):
    """Initialize and perform a training run given `file_` training script.
    Args:
        file_ -- training python script
        FLAGS -- FLAGS object or dictionary if already loaded
        run_dir -- the unique run directory to place experiment metadata
        runs_dir -- high level runs directory that houses all runs. Defaults to `~/runs`
        exclude -- comma separated list of files or directories to exlucde from rsync
    """
    global LOGGER

    clear_run()

    if FLAGS is None:
        pass
        # FLAGS = import_flags()

    default_path = '/training/data/runs'
    path_a = os.path.abspath(default_path if runs_dir is None else runs_dir)
    path_b = os.path.expanduser('~/runs')
    runs_dir = path_a if inside_docker() else path_b
    _globals['runs_dir'] = runs_dir

    run_dir = initialize_run(run_dir=run_dir, FLAGS=FLAGS)

    do_training_run(run_dir)
    
    LOGGER.info("Training run completed: {}".format(run_dir))
    raise ValueError("0 <- Training completed successfully.  IGNORE VALUE ERROR")


