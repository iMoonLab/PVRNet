# -*- coding: utf-8 -*-
import config
from utils import meter
from torch import nn
from torch import optim
from models import PVRNet
from torch.utils.data import DataLoader
from datasets import *
import argparse


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

    for i, (views, pcs, labels) in enumerate(train_loader):
        batch_time.reset()
        views = views.to(device=config.device)
        pcs = pcs.to(device=config.device)
        labels = labels.to(device=config.device)

        preds = net(pcs, views)  # bz x C x H x W
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
    retrieval_map = meter.RetrievalMAPMeter()

    # testing mode
    net.eval()

    total_seen_class = [0 for _ in range(40)]
    total_right_class = [0 for _ in range(40)]

    for i, (views, pcs, labels) in enumerate(val_loader):
        batch_time.reset()

        views = views.to(device=config.device)
        pcs = pcs.to(device=config.device)
        labels = labels.to(device=config.device)

        preds, fts = net(pcs, views, get_fea=True)  # bz x C x H x W
        # prec.add(preds.data, labels.data)

        prec.add(preds.data, labels.data)
        retrieval_map.add(fts.detach()/torch.norm(fts.detach(), 2, 1, True), labels.detach())
        for j in range(views.size(0)):
            total_seen_class[labels.data[j]] += 1
            total_right_class[labels.data[j]] += (np.argmax(preds.data,1)[j] == labels.cpu()[j])

        if i % config.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(val_loader)}]\t'
                  f'Batch Time {batch_time.value():.3f}\t'
                  f'Epoch Time {data_time.value():.3f}\t'
                  f'Prec@1 {prec.value(1):.3f}\t')

    mAP = retrieval_map.mAP()
    print(f' instance accuracy at epoch {epoch}: {prec.value(1)} ')
    print(f' mean class accuracy at epoch {epoch}: {(np.mean(np.array(total_right_class)/np.array(total_seen_class,dtype=np.float)))} ')
    print(f' map at epoch {epoch}: {mAP} ')
    return prec.value(1), mAP


def save_ckpt(epoch, epoch_pc, epoch_all, best_prec1, net, optimizer_pc, optimizer_all, training_conf=config.pv_net):
    ckpt = dict(
        epoch=epoch,
        epoch_pc=epoch_pc,
        epoch_all=epoch_all,
        best_prec1=best_prec1,
        model=net.module.state_dict(),
        optimizer_pc=optimizer_pc.state_dict(),
        optimizer_all=optimizer_all.state_dict(),
        training_conf=training_conf
    )
    torch.save(ckpt, config.pv_net.ckpt_file)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Main",
    )
    parser.add_argument("-batch_size", '-b', type=int, default=32, help="Batch size")
    parser.add_argument('-gpu', '-g', type=str, default=None, help='GPUS used')
    parser.add_argument(
        "-epochs", '-e', type=int, default=None, help="Number of epochs to train for"
    )
    return parser.parse_args()

def main():
    print('Training Process\nInitializing...\n')
    config.init_env()
    args = parse_args()

    total_batch_sz = config.pv_net.train.batch_sz * len(config.available_gpus.split(','))
    total_epoch = config.pv_net.train.max_epoch

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        total_batch_sz= config.pv_net.train.batch_sz * len(args.gpu.split(','))
    if args.epochs is not None:
        total_epoch = args.epochs


    train_dataset = pc_view_data(config.pv_net.pc_root,
                                 config.pv_net.view_root,
                                 status=STATUS_TRAIN,
                                 base_model_name=config.base_model_name)
    val_dataset = pc_view_data(config.pv_net.pc_root,
                               config.pv_net.view_root,
                               status=STATUS_TEST,
                               base_model_name=config.base_model_name)

    train_loader = DataLoader(train_dataset, batch_size=total_batch_sz,
                              num_workers=config.num_workers,shuffle = True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=total_batch_sz,
                            num_workers=config.num_workers,shuffle=True)

    best_prec1 = 0
    best_map = 0
    resume_epoch = 0

    epoch_pc_view = 0
    epoch_pc = 0

    # create model
    net = PVRNet()
    net = net.to(device=config.device)
    net = nn.DataParallel(net)

    # optimizer
    fc_param = [{'params': v} for k, v in net.named_parameters() if 'fusion' in k]
    if config.pv_net.train.optim == 'Adam':
        optimizer_fc = optim.Adam(fc_param, config.pv_net.train.fc_lr,
                                  weight_decay=config.pv_net.train.weight_decay)

        optimizer_all = optim.Adam(net.parameters(), config.pv_net.train.all_lr,
                                   weight_decay=config.pv_net.train.weight_decay)
    elif config.pv_net.train.optim == 'SGD':
        optimizer_fc = optim.SGD(fc_param, config.pv_net.train.fc_lr,
                                 momentum=config.pv_net.train.momentum,
                                 weight_decay=config.pv_net.train.weight_decay)

        optimizer_all = optim.SGD(net.parameters(), config.pv_net.train.all_lr,
                                  momentum=config.pv_net.train.momentum,
                                  weight_decay=config.pv_net.train.weight_decay)
    else:
        raise NotImplementedError
    print(f'use {config.pv_net.train.optim} optimizer')
    print(f'Sclae:{net.module.n_scale} ')


    if config.pv_net.train.resume:
        print(f'loading pretrained model from {config.pv_net.ckpt_file}')
        checkpoint = torch.load(config.pv_net.ckpt_file)
        state_dict = checkpoint['model']
        net.module.load_state_dict(checkpoint['model'])
        optimizer_fc.load_state_dict(checkpoint['optimizer_pc'])
        optimizer_all.load_state_dict(checkpoint['optimizer_all'])
        best_prec1 = checkpoint['best_prec1']
        epoch_pc_view = checkpoint['epoch_all']
        epoch_pc = checkpoint['epoch_pc']
        if config.pv_net.train.resume_epoch is not None:
            resume_epoch = config.pv_net.train.resume_epoch
        else:
            resume_epoch = max(checkpoint['epoch_pc'], checkpoint['epoch_all'])

    if config.pv_net.train.iter_train == False:
        print ('No iter')
        lr_scheduler_fc = torch.optim.lr_scheduler.StepLR(optimizer_fc, 5, 0.3)
        lr_scheduler_all = torch.optim.lr_scheduler.StepLR(optimizer_all, 5, 0.3)
    else:
        print ('iter')
        lr_scheduler_fc = torch.optim.lr_scheduler.StepLR(optimizer_fc, 6, 0.3)
        lr_scheduler_all = torch.optim.lr_scheduler.StepLR(optimizer_all, 6, 0.3)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=config.device)

    for epoch in range(resume_epoch, total_epoch):


        if config.pv_net.train.iter_train == True:
            if epoch < 12:
                lr_scheduler_fc.step(epoch=epoch_pc)
                print(lr_scheduler_fc.get_lr())

                if (epoch_pc + 1) % 3 == 0:
                    print ('train score block')
                    for m in net.module.parameters():
                        m.reqires_grad = False
                    net.module.fusion_conv1.requires_grad = True
                else:
                    print ('train all fc block')
                    for m in net.module.parameters():
                        m.reqires_grad = True

                train(train_loader, net, criterion, optimizer_fc, epoch)
                epoch_pc += 1

            else:
                lr_scheduler_all.step(epoch=epoch_pc_view)
                print(lr_scheduler_all.get_lr())

                if (epoch_pc_view + 1) % 3 == 0:
                    print('train score block')
                    for m in net.module.parameters():
                        m.reqires_grad = False
                    net.module.fusion_conv1.requires_grad = True
                else:
                    print('train all block')
                    for m in net.module.parameters():
                        m.reqires_grad = True

                train(train_loader, net, criterion, optimizer_all, epoch)
                epoch_pc_view += 1


        else:
            if epoch < 10:
                lr_scheduler_fc.step(epoch=epoch_pc)
                print(lr_scheduler_fc.get_lr())
                train(train_loader, net, criterion, optimizer_fc, epoch)
                epoch_pc += 1

            else:
                lr_scheduler_all.step(epoch=epoch_pc_view)
                print(lr_scheduler_all.get_lr())
                train(train_loader, net, criterion, optimizer_all, epoch)
                epoch_pc_view += 1


        with torch.no_grad():
            prec1, retrieval_map = validate(val_loader, net, epoch)

        # save checkpoints
        if best_prec1 < prec1:
            best_prec1 = prec1
            save_ckpt(epoch, epoch_pc, epoch_pc_view, best_prec1, net, optimizer_fc, optimizer_all)
        if best_map < retrieval_map:
            best_map = retrieval_map

        print('curr accuracy: ', prec1)
        print('best accuracy: ', best_prec1)
        print('best map: ', best_map)

    print('Train Finished!')


if __name__ == '__main__':
    main()

