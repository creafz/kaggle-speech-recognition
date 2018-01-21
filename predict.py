import click
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from click import option as opt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import dataset
from augmentation_transforms import make_augmentation_transforms
from model import make_nasnet_model
from utils import load_checkpoint, save_predictions


cudnn.benchmark = config.CUDNN_BENCHMARK


@click.command()
@opt('--experiment-name', type=str, required=True)
@opt('--dropout-p', default=0.5)
@opt('--batch-size', default=256)
@opt('--num-workers', default=config.NUM_WORKERS)
@opt('--augmentation', type=str, required=True)
@opt('--folds', default=5)
def main(
    experiment_name,
    dropout_p,
    batch_size,
    num_workers,
    augmentation,
    folds,
):
    transforms = make_augmentation_transforms(augmentation, mode='test')
    test_dataset = dataset.TestDataset(transform=transforms)
    model = make_nasnet_model(
        num_classes=config.NUM_CLASSES,
        dropout_p=dropout_p,
    )
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )
    test_predictions = np.zeros((folds, len(test_dataset), config.NUM_CLASSES))
    for fold_num in range(folds):
        checkpoint = load_checkpoint(
            f'{experiment_name}_{fold_num}_{folds}_best.pth'
        )
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda().eval()
        with torch.no_grad():
            for batch_index, (waves, _) in enumerate(tqdm(test_data_loader)):
                waves = Variable(waves).cuda()
                logits = model(waves)
                probs = F.softmax(logits, dim=1)
                numpy_probs = probs.cpu().data.numpy()
                start_index = batch_index * batch_size
                end_index = start_index + numpy_probs.shape[0]
                test_predictions[fold_num, start_index: end_index] = numpy_probs
    save_predictions(test_predictions, f'{experiment_name}.pkl')


if __name__ == '__main__':
    main()
