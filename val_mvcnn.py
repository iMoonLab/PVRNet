# -*- coding: utf-8 -*-
import config
from utils import meter
from torch import nn
from models import MVCNN
from torch.utils.data import DataLoader
from datasets import *


def validate(val_loader, net):
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
            print(f'[{i}/{len(val_loader)}]\t'
                  f'Batch Time {batch_time.value():.3f}\t'
                  f'Epoch Time {data_time.value():.3f}\t'
                  f'Prec@1 {prec.value(1):.3f}\t')

    print(f'mean class accuracy: {prec.value(1)} ')
    return prec.value(1)


def main():
    print('Training Process\nInitializing...\n')
    config.init_env()

    val_dataset = data_pth.view_data(config.view_net.data_root,
                                     status=STATUS_TEST,
                                     base_model_name=config.base_model_name)

    val_loader = DataLoader(val_dataset, batch_size=config.view_net.train.batch_sz,
                            num_workers=config.num_workers,shuffle=False)


    # create model
    net = MVCNN()
    net = net.to(device=config.device)
    net = nn.DataParallel(net)

    print(f'loading pretrained model from {config.view_net.ckpt_load_file}')
    checkpoint = torch.load(config.view_net.ckpt_load_file)
    net.module.load_state_dict(checkpoint['model'])
    best_prec1 = checkpoint['best_prec1']

    with torch.no_grad():
        prec1 = validate(val_loader, net)

    print('curr accuracy: ', prec1)
    print('best accuracy: ', best_prec1)

    print('Train Finished!')


if __name__ == '__main__':
    main()

