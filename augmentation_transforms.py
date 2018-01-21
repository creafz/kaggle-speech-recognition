import config
import custom_transforms as t
from dataset import load_noise_waves


augmentations = {
    'mel': t.MelSpectrogram(n_mels=128, hop_length=126),
    'mfcc': t.Mfcc(n_mels=128, hop_length=126),
}


def make_augmentation_transforms(augmentation, mode):
    if mode == 'train':
        transforms = [
            t.RandomPadToLength(length=config.AUDIO_LENGTH),
            t.Noise(
                length=config.AUDIO_LENGTH,
                noise_waves=load_noise_waves(),
                noise_limit=0.2,
            ).with_prob(0.5),
            t.RandomShift(shift_limit=0.2).with_prob(0.5),
        ]
    else:
        transforms = [t.PadToLength(length=config.AUDIO_LENGTH)]
    transforms.append(augmentations[augmentation])
    transforms += [
        t.Pad(((0, 0), (0, 1)), 'constant'),
        t.ExpandDims(),
        t.ToTensor(),
    ]
    return t.Compose(transforms)
