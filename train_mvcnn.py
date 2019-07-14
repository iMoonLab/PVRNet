# -*- coding: utf-8 -*-
import config
from utils import meter
from torch import nn
from torch import optim
from models import MVCNN
from torch.utils.data import DataLoader
from datasets import *


def train(train_loader, net, criterion, optimizer, epoch):
    """
    train for one epoch on the training set
    """
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    losses = meter.AverageValueMeter()
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)
    # training mode
    net.train()

    for i, (views, labels) in enumerate(train_loader):
        batch_time.reset()
        views = views.to(device=config.device)
        labels = labels.to(device=config.device)

        preds = net(views)  # bz x C x H x W
        loss = criterion(preds, labels)

        prec.add(preds.detach(), labels.detach())
        losses.add(loss.item())  # batchsize

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % config.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Batch Time {batch_time.value():.3f}\t'
                  f'Epoch Time {data_time.value():.3f}\t'
                  f'Loss {losses.value()[0]:.4f} \t'
                  f'Prec@1 {prec.value(1):.3f}\t')

    print(f'prec at epoch {epoch}: {prec.value(1)} ')


def validate(val_loader, net, epoch):
    """
    validation for one epoch on the val set
    """
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)

    # testing mode
    net.eval()

    for i, (views, labels) in enumerate(val_loader):
        batch_time.reset()
        # bz x 12 x 3 x 224 x 224
        views = views.to(device=config.device)
        labels = labels.to(device=config.device)

        preds = net(views)  # bz x C x H x W

        prec.add(preds.data, labels.data)

        if i % config.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(val_loader)}]\t'
                  f'Batch Time {batch_time.value():.3f}\t'
                  f'Epoch Time {data_time.value():.3f}\t'
                  f'Prec@1 {prec.value(1):.3f}\t')

    print(f'mean class accuracy at epoch {epoch}: {prec.value(1)} ')
    return prec.value(1)


def save_record(epoch, prec1, net: nn.Module):
    state_dict = net.state_dict()
    torch.save(state_dict, osp.join(config.view_net.ckpt_record_folder, f'epoch{epoch}_{prec1:.2f}.pth'))


def save_ckpt(epoch, best_prec1, net, optimizer, training_conf=config.view_net):
    ckpt = dict(
        epoch=epoch,
        best_prec1=best_prec1,
        model=net.module.state_dict(),
        optimizer=optimizer.state_dict(),
        training_conf=training_conf
    )
    torch.save(ckpt, config.view_net.ckpt_file)


def main():
    print('Training Process\nInitializing...\n')
    config.init_env()

    train_dataset = data_pth.view_data(config.view_net.data_root,
                                       status=STATUS_TRAIN,
                                       base_model_name=config.base_model_name)
    val_dataset = data_pth.view_data(config.view_net.data_root,
                                     status=STATUS_TEST,
                                     base_model_name=config.base_model_name)

    train_loader = DataLoader(train_dataset, batch_size=config.view_net.train.batch_sz,
                              num_workers=config.num_workers,shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size=config.view_net.train.batch_sz,
                            num_workers=config.num_workers,shuffle=True)

    best_prec1 = 0
    resume_epoch = 0
    # create model
    net = MVCNN()
    net = net.to(device=config.device)
    net = nn.DataParallel(net)
    optimizer = optim.SGD(net.parameters(), config.view_net.train.lr,
                          momentum=config.view_net.train.momentum,
                          weight_decay=config.view_net.train.weight_decay)
    # optimizer = optim.Adam(net.parameters(), config.view_net.train.lr,
    #                        weight_decay=config.view_net.train.weight_decay)

    if config.view_net.train.resume:
        print(f'loading pretrained model from {config.view_net.ckpt_file}')
        checkpoint = torch.load(config.view_net.ckpt_file)
        net.module.load_state_dict({k[7:]: v for k, v in checkpoint['model'].items()})
        # net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_prec1 = checkpoint['best_prec1']
        if config.view_net.train.resume_epoch is not None:
            resume_epoch = config.view_net.train.resume_epoch
        else:
            resume_epoch = checkpoint['epoch'] + 1

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=config.device)

    # for p in net.module.feature.parameters():
    #     p.requires_grad = False

    for epoch in range(resume_epoch, config.view_net.train.max_epoch):
        if epoch >= 5:
            for p in net.parameters():
                p.requires_grad = True
        lr_scheduler.step(epoch=epoch)

        train(train_loader, net, criterion, optimizer, epoch)

        with torch.no_grad():
            prec1 = validate(val_loader, net, epoch)

        # save checkpoints
        if best_prec1 < prec1:
            best_prec1 = prec1
            save_ckpt(epoch, best_prec1, net, optimizer)

        save_record(epoch, prec1, net.module)
        print('curr accuracy: ', prec1)
        print('best accuracy: ', best_prec1)

    print('Train Finished!')


if __name__ == '__main__':
    main()

