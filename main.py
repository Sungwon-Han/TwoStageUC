import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import time
import numpy as np

import datasets
import models
from lib.criterion import Criterion, IID_loss
from lib.utils import AverageMeter
from lib.normalize import Normalize
from lib.protocols import test
import warnings
warnings.filterwarnings("ignore")

def config():
    global args
    parser = argparse.ArgumentParser(description='config for EmbedUL')

    parser.add_argument('--dataset', default='cifar10', type=str, help='available dataset: cifar10, cifar100, cifar20 (dafault: cifar10)')
    parser.add_argument('--low_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--lr', default=0.01, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--epochs', default=300, type=int, help='max epoch per round. (default: 200)')
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
        class_num = 10
    elif args.dataset == 'cifar100':
        mean = (0.5071, 0.4866, 0.4409)
        std = (0.2009, 0.1984, 0.2023)
        class_num = 100
    elif args.dataset == 'cifar20':
        mean = (0.5071, 0.4866, 0.4409)
        std = (0.2009, 0.1984, 0.2023)
        class_num = 20
        

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
        # Train and Test is same dataset but data augmentation is different
        trainset = datasets.CIFAR10_(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        testset = datasets.CIFAR10_(root='./data', train=True, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, drop_last=True)
    elif args.dataset == 'cifar100':
        trainset = datasets.CIFAR100_(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
        testset = datasets.CIFAR100_(root='./data', train=True, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, drop_last=True)
    elif args.dataset == 'cifar20':
        trainset = datasets.CIFAR20_(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
        testset = datasets.CIFAR20_(root='./data', train=True, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, drop_last=True)            
        
    return trainset, trainloader, testset, testloader, class_num


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if epoch >= 80:
        lr = args.lr * (0.1 ** ((epoch - 80) // 40))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(epoch, net, trainloader, optimizer, device, fc, criterion):
    train_loss = AverageMeter()
    mi_loss = AverageMeter()
    ce_loss = AverageMeter()
    net.train()
    fc.train()
    
    # adjust learning rate
    adjust_learning_rate(optimizer, epoch)  
    optimizer.zero_grad()
    
    loss_list = [0, 0, 0, 0, 0]
    for batch_idx, (inputs1, inputs2, _, indexes) in enumerate(trainloader):
        inputs1, inputs2, indexes = inputs1.to(device), inputs2.to(device), indexes.to(device)
        inputs = torch.cat((inputs1,inputs2), 0)
 
        optimizer.zero_grad()
        features = net(inputs)
        # Regulariztion loss
        loss_ce = criterion(features, indexes)
        
        output_list = fc(features)
        
        # Mutual Information Loss
        loss_mi = 0
        for i in range(len(output_list)):
            output1 = output_list[i][:args.batch_size, :]
            output2 = output_list[i][args.batch_size:, :]
            iid = IID_loss(output1, output2)
            loss_mi += iid
            loss_list[i] += iid.item()
            
        loss_mi /= len(output_list)
        total_loss = loss_ce + loss_mi
        total_loss.backward()
        train_loss.update(total_loss.item(), inputs.size(0))
        mi_loss.update(loss_mi.item(), inputs.size(0))
        ce_loss.update(loss_ce.item(), inputs.size(0))
        optimizer.step()
        
        if batch_idx % 80 == 0:
            print('Epoch: [{epoch}][{elps_iters}/{tot_iters}] '
                  'Train loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'MI loss: {mi_loss.val:.4f} ({mi_loss.avg:.4f}) '
                  'CE loss: {ce_loss.val:.4f} ({ce_loss.avg:.4f}) '.format(
                      epoch=epoch, elps_iters=batch_idx,tot_iters=len(trainloader), 
                      train_loss=train_loss, mi_loss=mi_loss, ce_loss=ce_loss))
    print(loss_list)
    return train_loss.avg
            
def main():
    args = config()
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainset, trainloader, testset, testloader, class_num = preprocess(args)
    net = models.__dict__['ResNet18withSobel'](low_dim=args.low_dim)
    fc = models.Multi_head_fc(class_num)
    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    criterion = Criterion(args.batch_m, args.batch_t, args.batch_size, args.device)
    
    if args.device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    
    net.to(args.device)
    fc.to(args.device)
    criterion.to(args.device)
    
    if args.test_only or len(args.resume) > 0:
        model_path = args.model_dir + args.resume
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(args.model_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])
        if args.test_only:
            fc.load_state_dict(checkpoint['fc'])
            
    if args.test_only:
        acc = test(net, testloader, args.device, fc, class_num)
        print("accuracy: {}".format(acc))  
        sys.exit(0)

    best_loss = float('inf')
    best_accuracy = None
    
    for epoch in range(args.epochs):
        loss = train(epoch, net, trainloader, optimizer, args.device, fc, criterion)
        acc_list = test(net, testloader, args.device, fc, class_num)
        print("accuracy: {}\n".format(acc_list))  

        if loss < best_loss:
            print("state saving...")
            state = {
                'net': net.state_dict(),
                'fc': fc.state_dict(),
                'acc_list': acc_list,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_ul.t7')
            best_loss = loss
            best_accuracy = acc_list
        print("best loss: %.2f\n" % (best_loss))
        print("best accuracy: {}".format(best_accuracy))
    print("Final best accuracy: {}".format(best_accuracy))

if __name__ == "__main__":
    main()
