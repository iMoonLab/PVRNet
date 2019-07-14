# -*- coding: utf-8 -*-
import config
import torch
import os.path as osp
from utils import meter
from torch import nn
from torch import optim
from models import DGCNN
from torch.utils.data import DataLoader
from datasets import data_pth, STATUS_TRAIN, STATUS_TEST

def validate(val_loader, net, epoch):
    """
    validation for one epoch on the val set
    """
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)

    # testing mode
    net.eval()

    for i, (pcs, labels) in enumerate(val_loader):
        batch_time.reset()
        # bz x 12 x 3 x 224 x 224
        pcs = pcs.to(device=config.device)
        labels = labels.to(device=config.device)

        preds = net(pcs)  # bz x C x H x W

        prec.add(preds.data, labels.data)

        if i % config.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(val_loader)}]\t'
                  f'Batch Time {batch_time.value():.3f}\t'
                  f'Epoch Time {data_time.value():.3f}\t'
                  f'Prec@1 {prec.value(1):.3f}\t')

    print(f'mean class accuracy at epoch {epoch}: {prec.value(1)} ')
    return prec.value(1)




def main():
    print('Training Process\nInitializing...\n')
    config.init_env()

    val_dataset = data_pth.pc_data(config.pc_net.data_root, status=STATUS_TEST)

    val_loader = DataLoader(val_dataset, batch_size=config.pc_net.validation.batch_sz,
                            num_workers=config.num_workers,shuffle=True)

    # create model
    net = DGCNN(n_neighbor=config.pc_net.n_neighbor,num_classes=config.pc_net.num_classes)
    net = torch.nn.DataParallel(net)
    net = net.to(device=config.device)
    optimizer = optim.Adam(net.parameters(), config.pc_net.train.lr,
                          weight_decay=config.pc_net.train.weight_decay)

    print(f'loading pretrained model from {config.pc_net.ckpt_load_file}')
    checkpoint = torch.load(config.pc_net.ckpt_load_file)
    net.module.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_prec1 = checkpoint['best_prec1']
    resume_epoch = checkpoint['epoch']

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=config.device)

    # for p in net.module.feature.parameters():
    #     p.requires_grad = False

    with torch.no_grad():
        prec1 = validate(val_loader, net, resume_epoch)

    print('curr accuracy: ', prec1)
    print('best accuracy: ', best_prec1)



if __name__ == '__main__':
    main()

