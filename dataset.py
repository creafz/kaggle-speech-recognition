from collections import Counter
import os

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from torch.utils.data import Dataset

import config
from utils import load_wav


labels = [
    'yes',
    'no',
    'up',
    'down',
    'left',
    'right',
    'on',
    'off',
    'stop',
    'go',
    'silence',
    'unknown',
]
idx_to_label = dict(enumerate(labels))
label_to_idx = {name: idx for idx, name in idx_to_label.items()}
labels_set = set(labels)


def load_noise_waves():
    noise_waves = []
    noise_directory = os.path.join(config.TRAIN_DIR_PATH, '_background_noise_')
    for filepath in sorted(os.listdir(noise_directory)):
        if not filepath.endswith('.wav'):
            continue
        wave = load_wav(os.path.join(noise_directory, filepath))
        noise_waves.append(wave)
    return noise_waves


class TrainValidDataset(Dataset):

    @staticmethod
    def _prepare_data():
        data = []
        for directory in sorted(os.listdir(config.TRAIN_DIR_PATH)):
            if directory == '_background_noise_':
                continue
            if directory in labels_set:
                label = directory
            else:
                label = 'unknown'
            label_idx = label_to_idx[label]
            directory_path = os.path.join(config.TRAIN_DIR_PATH, directory)
            filenames = sorted(os.listdir(directory_path))
            for filename in filenames:
                if not filename.endswith('.wav'):
                    continue
                user_id = filename.split('_')[0]
                data.append([
                    os.path.join(directory_path, filename),
                    label_idx,
                    user_id,
                ])
        return data

    @staticmethod
    def _get_dataset_index(data, folds, fold_num, mode):
        data = shuffle(data, random_state=config.SHUFFLE_SEED)
        group_kfold = GroupKFold(n_splits=folds)
        groups = [user_id for _, _, user_id in data]
        train_index, valid_index = (
            list(group_kfold.split(data, groups=groups))[fold_num]
        )
        dataset_index = train_index if mode == 'train' else valid_index
        return dataset_index

    def get_item_weights(self):
        label_idxs = []
        for i in self.dataset_index:
            if i == 'silence':
                label_idx = label_to_idx['silence']
            else:
                _, label_idx, _ = self.data[i]
            label_idxs.append(label_idx)
        label_weights = {
            idx: 1 / count for idx, count in Counter(label_idxs).items()
        }
        item_weights = np.array([label_weights[idx] for idx in label_idxs])
        return item_weights

    def __init__(self, transform=None, mode='train', folds=5, fold_num=0):
        assert mode in {'train', 'valid'}
        self.transform = transform
        self.mode = mode
        data = self._prepare_data()
        dataset_index = self._get_dataset_index(data, folds, fold_num, mode)
        self.dataset_index = [*dataset_index, 'silence']
        self.data = data
        self.silence_wave = np.zeros(config.AUDIO_LENGTH)
        self.noise_waves = load_noise_waves()

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, index):
        i = self.dataset_index[index]
        if i == 'silence':
            label = label_to_idx['silence']
            wave = self.silence_wave
        else:
            filepath, label, user_id = self.data[i]
            wave = load_wav(filepath)
        if self.transform:
            wave = self.transform(wave)
        return wave, label


class TestDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        self.files = list(sorted(os.listdir(config.TEST_DIR_PATH)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]
        filepath = os.path.join(config.TEST_DIR_PATH, filename)
        wave = load_wav(filepath)
        if self.transform:
            wave = self.transform(wave)
        return wave, filepath
