"""
Audio transform functions are based on functions from
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/46982
https://kaggle2.blob.core.windows.net/forum-message-attachments/265667/8192/audio_processing_tf.py
"""

import random

import librosa
import numpy as np
import torch

import config


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, wave):
        for t in self.transforms:
            wave = t(wave)
        return wave


class CustomTransform:

    def __init__(self):
        self.prob = 1

    def with_prob(self, prob):
        self.prob = prob
        return self

    def __call__(self, wave):
        if self.prob < 1 and np.random.rand() > self.prob:
            return wave

        return self.transform(wave)

    def transform(self, wave):
        raise NotImplementedError()


class PadToLength(CustomTransform):

    def __init__(self, length):
        super().__init__()
        self.length = length

    def transform(self, wave):
        return np.pad(wave, (self.length - wave.shape[0], 0), mode='constant')


class RandomPadToLength(CustomTransform):

    def __init__(self, length):
        super().__init__()
        self.length = length

    def transform(self, wave):
        wave_length = wave.shape[0]
        if wave_length >= self.length:
            return wave

        left_pad = np.random.randint(0, self.length - wave_length)
        right_pad = self.length - left_pad - wave_length
        return np.pad(wave, (left_pad, right_pad), mode='constant')


class ExpandDims(CustomTransform):

    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def transform(self, wave):
        return np.expand_dims(wave, axis=self.axis)


class Noise(CustomTransform):

    def __init__(self, length, noise_waves, noise_limit=0.2):
        super().__init__()
        self.noise_waves = noise_waves
        self.noise_limit = noise_limit
        self.length = length

    def _random_crop(self, wave):
        wave_length = wave.shape[0]
        start_idx = np.random.randint(0, wave_length - self.length)
        return wave[start_idx: start_idx + self.length]

    def transform(self, wave):
        noise_wave = random.choice(self.noise_waves)
        noise_wave = self._random_crop(noise_wave)
        alpha = np.random.random() * self.noise_limit
        wave = alpha * noise_wave + wave
        wave = np.clip(wave, -1, 1)
        return wave


class Pad(CustomTransform):

    def __init__(self, *pad_params):
        super().__init__()
        self.pad_params = pad_params

    def transform(self, wave):
        return np.lib.pad(wave, *self.pad_params)


class RandomShift(CustomTransform):

    def __init__(self, shift_limit=0.2):
        super().__init__()
        self.shift_limit = shift_limit

    def transform(self, wave):
        wave_length = len(wave)
        shift_limit = self.shift_limit*wave_length
        shift = random.randint(-shift_limit, shift_limit)
        t0 = -min(0, shift)
        t1 =  max(0, shift)
        wave = np.pad(wave, (t0, t1), 'constant')
        wave = wave[:-t0] if t0 else wave[t1:]
        return wave


class MelSpectrogram(CustomTransform):
    def __init__(self, n_mels, hop_length):
        super().__init__()
        self.n_mels = n_mels
        self.hop_length = hop_length

    def transform(self, wave):
        spectrogram = librosa.feature.melspectrogram(
            wave,
            sr=config.AUDIO_SAMPLING_RATE,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=480,
            fmin=20,
            fmax=4000,
        )
        spectrogram = librosa.power_to_db(spectrogram)
        spectrogram = spectrogram.astype(np.float32)
        return spectrogram


class Mfcc(CustomTransform):

    def __init__(self, n_mels, hop_length):
        super().__init__()
        self.n_mels = n_mels
        self.hop_length = hop_length

    def transform(self, wave):
        spectrogram = librosa.feature.melspectrogram(
            wave,
            sr=config.AUDIO_SAMPLING_RATE,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=480,
            fmin=20,
            fmax=4000
        )
        idx = [spectrogram > 0]
        spectrogram[idx] = np.log(spectrogram[idx])
        dct_filters = librosa.filters.dct(n_filters=128, n_input=128)
        mfcc = [
            np.matmul(dct_filters, x) for
            x in np.split(spectrogram, spectrogram.shape[1], axis=1)
        ]
        mfcc = np.hstack(mfcc)
        mfcc = mfcc.astype(np.float32)
        return mfcc


class ToTensor(CustomTransform):

    def __init__(self, tensor_type=torch.FloatTensor):
        super().__init__()
        self.tensor_type = tensor_type

    def transform(self, wave):
        return torch.from_numpy(wave).type(self.tensor_type)
