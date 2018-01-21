from collections import defaultdict

import click
from click import option as opt
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from augmentation_transforms import make_augmentation_transforms
import config
import dataset
from model import make_nasnet_model
from optimizer import get_optimizer
from utils import (
    save_checkpoint,
    load_checkpoint,
    MetricMonitor,
    set_seed,
    calculate_accuracy,
    TensorboardClient
)


cudnn.benchmark = config.CUDNN_BENCHMARK


def forward_pass(
    images,
    targets,
    model,
    loss_fn,
    epoch,
    stream,
    monitor,
    mode='train',
):
    images = Variable(images).cuda(async=True)
    targets = Variable(targets).cuda(async=True)
    outputs = model(images)
    accuracy = calculate_accuracy(outputs, targets)
    monitor.update('accuracy', accuracy, multiply_by_n=False)
    loss = loss_fn(outputs, targets)
    monitor.update('loss', loss.data[0])
    stream.set_description(f'epoch: {epoch} | {mode}: {monitor}')
    return loss, outputs


def train(
    train_data_loader,
    model,
    optimizer,
    iter_size,
    loss_fn,
    epoch,
    tensorboard_client,
    grad_max_norm,
):
    model.train()
    train_monitor = MetricMonitor(batch_size=train_data_loader.batch_size)
    stream = tqdm(train_data_loader)
    for i, (images, targets) in enumerate(stream, start=1):
        loss, _ = forward_pass(
            images,
            targets,
            model,
            loss_fn,
            epoch,
            stream,
            train_monitor,
            mode='train',
        )
        loss.backward()
        if grad_max_norm is not None:
            torch.nn.utils.clip_grad_norm(model.parameters(), grad_max_norm)
        if i % iter_size == 0 or i == len(train_data_loader):
            optimizer.step()
            optimizer.zero_grad()

    tensorboard_client.log_value(
        'train',
        'lr',
        optimizer.param_groups[0]['lr'],
        epoch,
    )
    for metric, value in train_monitor.get_metric_values():
        tensorboard_client.log_value('train', metric, value, epoch)


def validate(valid_data_loader, model, loss_fn, epoch, tensorboard_client):
    model.eval()
    valid_monitor = MetricMonitor(batch_size=valid_data_loader.batch_size)
    stream = tqdm(valid_data_loader)
    with torch.no_grad():
        for images, targets in stream:
            _, outputs = forward_pass(
                images,
                targets,
                model,
                loss_fn,
                epoch,
                stream,
                valid_monitor,
                mode='valid',
            )
    for metric, value in valid_monitor.get_metric_values():
        tensorboard_client.log_value('valid', metric, value, epoch)
    return valid_monitor


def train_and_validate(
    train_data_loader,
    valid_data_loader,
    model,
    optimizer,
    iter_size,
    scheduler,
    loss_fn,
    epochs,
    experiment_name,
    tensorboard_client,
    start_epoch,
    best_val_loss,
    max_epochs_without_improvement,
    grad_max_norm,
):
    if best_val_loss is None:
        best_val_loss = float('+inf')

    epochs_without_improvement = 0
    best_checkpoint = None
    for epoch in range(start_epoch, epochs + 1):
        train(
            train_data_loader,
            model,
            optimizer,
            iter_size,
            loss_fn,
            epoch,
            tensorboard_client,
            grad_max_norm,
        )
        val_monitor = validate(
            valid_data_loader,
            model,
            loss_fn,
            epoch,
            tensorboard_client,
        )
        val_loss = val_monitor.get_avg('loss')
        if val_loss < best_val_loss:
            print('Best model so far!')
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            save_checkpoint(
                 best_checkpoint,
                 f'{experiment_name}_best.pth',
                 verbose=True,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement > max_epochs_without_improvement:
            print(
                f'{epochs_without_improvement} epochs without improvement. '
                f'Training is finished.'
            )
            break

        scheduler.step(val_loss)

    return best_checkpoint


@click.command()
@opt('--batch-size', default=32)
@opt('--optimizer-name', type=click.Choice(['adam', 'sgd']), default='sgd')
@opt('--lr', default=0.001)
@opt('--epochs', default=300)
@opt('--iter-size', default=1)
@opt('--experiment-name', type=str)
@opt('--folds', default=5)
@opt('--fold-num', default=0)
@opt('--load-best-model', is_flag=True)
@opt('--load-best-model-optimizer', is_flag=True)
@opt('--start-epoch', default=1)
@opt('--seed', default=config.SEED)
@opt('--dropout-p', default=0.5)
@opt('--num-workers', default=config.NUM_WORKERS)
@opt('--max-epochs-without-improvement', default=9)
@opt('--grad-max-norm', type=float)
@opt('--augmentation', type=str, required=True)
def main(
    batch_size,
    optimizer_name,
    lr,
    epochs,
    iter_size,
    experiment_name,
    folds,
    fold_num,
    load_best_model,
    load_best_model_optimizer,
    start_epoch,
    seed,
    dropout_p,
    num_workers,
    max_epochs_without_improvement,
    grad_max_norm,
    augmentation,
):
    set_seed(seed)
    transform_train = make_augmentation_transforms(augmentation, mode='train')
    transform_valid = make_augmentation_transforms(augmentation, mode='valid')

    if experiment_name is None:
        experiment_name = f'{config.PROJECT_NAME}'

    full_experiment_name = f'{experiment_name}_{fold_num}_{folds}'
    print(full_experiment_name)

    model = make_nasnet_model(
        num_classes=config.NUM_CLASSES,
        dropout_p=dropout_p,
    )
    best_val_loss = None
    model = model.cuda()
    optimizer = get_optimizer(optimizer_name, lr, model)
    if load_best_model:
        checkpoint_filename = f'{full_experiment_name}_temp.pth'
        print(f'Loading checkpoint {checkpoint_filename}')
        checkpoint = load_checkpoint(checkpoint_filename)
        model.load_state_dict(checkpoint['state_dict'])
        best_val_loss = checkpoint.get('val_loss')

        if load_best_model_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer.state = defaultdict(dict, optimizer.state)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    tensorboard_client = TensorboardClient(full_experiment_name)
    loss_fn = nn.CrossEntropyLoss().cuda()
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=3,
        verbose=True,
        threshold=1e-5,
        min_lr=0,
        mode='min',
    )

    dataset_args = {
        'folds': folds,
        'fold_num': fold_num,
    }

    train_dataset = dataset.TrainValidDataset(
        mode='train',
        transform=transform_train,
        **dataset_args,
    )
    valid_dataset = dataset.TrainValidDataset(
        mode='valid',
        transform=transform_valid,
        **dataset_args,
    )

    data_loader_args = {
        'pin_memory': True,
        'num_workers': num_workers,
    }

    train_data_weights = (
        torch.from_numpy(train_dataset.get_item_weights()).double()
    )
    train_sampler = WeightedRandomSampler(
        train_data_weights,
        len(train_dataset),
    )

    train_data_loader = DataLoader(
        **data_loader_args,
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
    )
    valid_data_loader = DataLoader(
        **data_loader_args,
        dataset=valid_dataset,
        batch_size=batch_size,
    )

    train_and_validate(
        train_data_loader,
        valid_data_loader,
        model,
        optimizer,
        iter_size,
        scheduler,
        loss_fn,
        epochs,
        full_experiment_name,
        tensorboard_client,
        start_epoch,
        best_val_loss,
        max_epochs_without_improvement,
        grad_max_norm,
    )


if __name__ == '__main__':
    main()
