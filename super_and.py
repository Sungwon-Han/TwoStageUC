'''
Impletment for super-AND
'''
import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import time
from datetime import datetime
import numpy as np

import datasets
import models
from lib.non_parametric_classifier import NonParametricClassifier
from lib.ans_discovery import ANsDiscovery
from lib.criterion import Criterion_SAND, UELoss
from lib.protocols import kNN
from lib.utils import AverageMeter
from lib.normalize import Normalize
from lib.LinearAverage import LinearAverage


def config():
    global args
    parser = argparse.ArgumentParser(description='config for super-AND')

    parser.add_argument('--dataset', default='cifar10', type=str, help='available dataset: cifar10, cifar100 (dafault: cifar10)')
    parser.add_argument('--network', default='resnet18', type=str, help='available network: resnet18, resnet101 (default: resnet18)')

    parser.add_argument('--low_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--npc_t', default=0.1, type=float, metavar='T', help='temperature parameter for softmax')
    parser.add_argument('--npc_m', default=0.5, type=float, help='momentum for non-parametric updates')
    parser.add_argument('--ANs_select_rate', default=0.25, type=float, help='ANs select rate at each round')
    parser.add_argument('--ANs_size', default=1, type=int, help='ANs size discarding the anchor')
    parser.add_argument('--lr', default=0.03, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--epochs', default=200, type=int, help='max epoch per round. (default: 200)')
    parser.add_argument('--rounds', default=5, type=int, help='max iteration, including initialisation one. ''(default: 5)')

    parser.add_argument('--batch_t', default=0.1, type=float, metavar='T', help='temperature parameter for softmax')
    parser.add_argument('--batch_m', default=1, type=float, metavar='N', help='m for negative sum')
    parser.add_argument('--batch_size', default=128, type=int, metavar='B', help='training batch size')

    parser.add_argument('--model_dir', default='checkpoint/', type=str, help='model save path')
    parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--test_only', action='store_true', help='test only')

    parser.add_argument('--seed', default=1567010775, type=int, help='random seed')

    args = parser.parse_args()
    return args


def preprocess(args):
    if args.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == 'cifar100':
        mean = (0.5071, 0.4866, 0.4409)
        std = (0.2009, 0.1984, 0.2023)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if args.dataset == 'cifar10':
        trainset = datasets.CIFAR10_(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
        testset = datasets.CIFAR10_(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, drop_last=True)
    elif args.dataset == 'cifar100':
        trainset = datasets.CIFAR100_(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
        testset = datasets.CIFAR100_(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, drop_last=True)
        
    return trainset, trainloader, testset, testloader


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if epoch >= 80:
        lr = args.lr * (0.1 ** ((epoch - 80) // 40))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(round, epoch, net, trainloader, optimizer, npc, criterion, criterion2, ANs_discovery, device):
    # tracking variables
    train_loss = AverageMeter()
    i_loss = AverageMeter()
    e_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    c_loss = AverageMeter()
    # switch the model to train mode
    net.train()
    # adjust learning rate
    adjust_learning_rate(optimizer, epoch)  
    optimizer.zero_grad()

    for batch_idx, (inputs1, inputs2, _, indexes) in enumerate(trainloader):
        inputs1, inputs2, indexes = inputs1.to(device), inputs2.to(device), indexes.to(device)
        inputs = torch.cat((inputs1,inputs2), 0)

        features = net(inputs) # (256, 128)
        outputs = npc(features, indexes) # (256, 50000)

        # AND + Augmentation Loss
        loss_i = criterion(outputs, indexes, ANs_discovery)
        
        # UELoss
        val = 0.2 * (epoch // 80)
        loss_e = 0
        if val > 0:
            outputs_e = outputs.clone()
            for i in range(0, len(indexes)):
                outputs_e[i, indexes[i]] = -10
                outputs_e[i + inputs.shape[0] // 2, indexes[i]] = -10 
            loss_e = criterion2(outputs_e) / inputs.size(0)
            e_loss.update(loss_e.item(), inputs.size(0))        

        loss = loss_i + val*loss_e
        loss.backward()
        train_loss.update(loss.item(), inputs.size(0))
        i_loss.update(loss_i.item(), inputs.size(0))


        optimizer.step()
        optimizer.zero_grad()
        if batch_idx % 80 == 0:
            print('Round: {round} Epoch: [{epoch}][{elps_iters}/{tot_iters}] '
                  'Train loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'I Loss: {i_loss.val:.4f} ({i_loss.avg:.4f}) '
                  'E Loss: {e_loss.val:.4f} ({e_loss.avg:.4f}) '.format(
                  round=round, epoch=epoch, elps_iters=batch_idx,
                  tot_iters=len(trainloader), train_loss=train_loss, i_loss=i_loss, e_loss = e_loss))


def main():
    args = config()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainset, trainloader, testset, testloader = preprocess(args)
    ntrain = len(trainset)
    cheat_labels = torch.tensor(trainset.targets).long().to(args.device)
    net = models.__dict__['ResNet18withSobel'](low_dim=args.low_dim)
    npc = NonParametricClassifier(args.low_dim, ntrain, args.npc_t, args.npc_m)
    ANs_discovery = ANsDiscovery(ntrain, args.ANs_select_rate, args.ANs_size, args.device)
    criterion = Criterion_SAND(args.batch_m, args.batch_t, args.batch_size, args.device)
    criterion2 = UELoss()
    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    if args.device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    
    net.to(args.device)
    npc.to(args.device)
    ANs_discovery.to(args.device)
    criterion.to(args.device)
    criterion2.to(args.device)
    
    if args.test_only or len(args.resume) > 0:
        model_path = args.model_dir + args.resume
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(args.model_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])
        npc.load_state_dict(checkpoint['npc'])
        ANs_discovery = checkpoint['ANs_discovery']
        best_acc = checkpoint['acc']
        start_round = checkpoint['round']
        start_epoch = checkpoint['epoch']

    if args.test_only:
        acc = kNN(net, npc, trainloader, testloader, K=200, sigma=0.1, recompute_memory=False, device=args.device)
        print("accuracy: %.2f\n" % (acc*100))  
        sys.exit(0)

    best_acc = 0
    for r in range(args.rounds):
        if r > 0:
            ANs_discovery.update(r, npc, cheat_labels)

        for epoch in range(args.epochs):
            train(r, epoch, net, trainloader, optimizer, npc, criterion, criterion2, ANs_discovery, args.device)
            acc = kNN(net, npc, trainloader, testloader, K=200, sigma=0.1, recompute_memory=False, device=args.device)
            print("accuracy: %.2f\n" % (acc*100))  

            if acc > best_acc:
                print("state saving...")
                best_acc = acc
            print("best accuracy: %.2f\n" % (best_acc*100))
            
    state = {
            'net': net.state_dict(),
            'npc': npc.state_dict(),
            'ANs_discovery' : ANs_discovery.state_dict(),
            'acc': acc,
            'round': r,
            'epoch': epoch
             }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt_embed.t7')


if __name__ == "__main__":
    main()