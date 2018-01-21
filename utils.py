import os
import pickle
import random
from collections import defaultdict

import librosa
import numpy as np
import tensorboard_logger
import torch

import config


def save_pickle(data, filename, base_directory, verbose=False):
    if verbose:
        print(f'Saving {filename}')
    with open(os.path.join(base_directory, filename), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename, base_directory):
    with open(os.path.join(base_directory, filename), 'rb') as f:
        return pickle.load(f)


def save_predictions(data, filename):
    save_pickle(data, filename, base_directory=config.PREDICTIONS_PATH)


def load_predictions(filename):
    return load_pickle(filename, base_directory=config.PREDICTIONS_PATH)


def make_path_func(base_dir):
    def path_func(filename):
        return os.path.join(base_dir, filename)
    return path_func


def save_checkpoint(state, filename, verbose=False):
    if verbose:
        print(f'Saving {filename}')
    filepath = os.path.join(config.SAVED_MODELS_PATH, filename)
    torch.save(state, filepath)
    return filepath


def load_checkpoint(filename, verbose=True):
    path = os.path.join(config.SAVED_MODELS_PATH, filename)
    if verbose:
        print(f'Loading {path}')
    return torch.load(path)


class MetricMonitor:

    def __init__(self, batch_size=None):
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {'sum': 0, 'count': 0, 'avg': 0})

    def update(self, metric_name, value, n=None, multiply_by_n=True):
        if n is None:
            n = self.batch_size
        metric = self.metrics[metric_name]
        if multiply_by_n:
            metric['sum'] += value * n
        else:
            metric['sum'] += value
        metric['count'] += n
        metric['avg'] = metric['sum'] / metric['count']

    def get_avg(self, metric_name):
        return self.metrics[metric_name]['avg']

    def get_metric_values(self):
        return [
            (metric, values['avg']) for metric, values in self.metrics.items()
        ]

    def __str__(self):
        return ' | '.join(
            f'{metric_name} {metric["avg"]:.6f}'
            for metric_name, metric in self.metrics.items()
        )


class TensorboardClient:

    def __init__(self, experiment_name):
        tensorboard_logger.configure(
            os.path.join(config.TENSORBOARD_LOGS_DIR, experiment_name))

    def log_value(self, mode, key, value, step):
        name = f'{mode}/{key}'
        tensorboard_logger.log_value(name, value, step)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def calculate_accuracy(outputs, targets):
    _, predictions = outputs.topk(1, 1, True, True)
    predictions = predictions.t()
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))
    correct_k = correct[0].view(-1).float().sum(0)
    return correct_k.data.cpu()[0]


def load_wav(filepath):
    return librosa.core.load(filepath, sr=config.AUDIO_SAMPLING_RATE)[0]
