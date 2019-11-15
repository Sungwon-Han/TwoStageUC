import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.normalize import Normalize


class _Head_fc(nn.Module):
    def __init__(self, nfeatures = 128, nclass = 10, T = 0.1):
        super(_Head_fc, self).__init__()
        self.nclass = nclass
        self.l2norm = Normalize(2)
        self.weight = nn.Parameter(torch.empty(nfeatures, nclass))
        self.softmax = nn.Softmax(dim = 1)
        self.T = T
        nn.init.normal_(self.weight)
        
    def forward(self, x):
        predval = torch.mm(x, self.l2norm(self.weight)) / self.T
        return self.softmax(predval)    
    
class Multi_head_fc(nn.Module):
    def __init__(self, nclasses=10, num_sub_heads = 5):
        super(Multi_head_fc, self).__init__()
        self.num_sub_heads = num_sub_heads
        self.heads = nn.ModuleList([_Head_fc(128, nclasses) for _ in range(self.num_sub_heads)])
      
    def forward(self, x):
        results = []
        for i in range(self.num_sub_heads):
            results.append(self.heads[i](x))
        return results    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out        

class ResNetwithSobel(nn.Module):
    def __init__(self, block, num_blocks, low_dim=128):
        super(ResNetwithSobel, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.in_planes = 64
        self.conv1_sobel = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_sobel = nn.BatchNorm2d(64)
        self.layer1_sobel = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2_sobel = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3_sobel = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4_sobel = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512*2*block.expansion, low_dim)
        self.l2norm = Normalize(2)

        grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        grayscale.weight.data.fill_(1.0 / 3.0)
        grayscale.bias.data.zero_()
        sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        sobel_filter.weight.data[0,0].copy_(
            torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        )
        sobel_filter.weight.data[1,0].copy_(
            torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        )
        sobel_filter.bias.data.zero_()
        self.sobel = nn.Sequential(grayscale, sobel_filter)
        for p in self.sobel.parameters():
            p.requires_grad = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Original Input
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        # Sobel Input
        x_sobel = self.sobel(x)
        out_sobel = F.relu(self.bn1_sobel(self.conv1_sobel(x_sobel)))
        out_sobel = self.layer1_sobel(out_sobel)
        out_sobel = self.layer2_sobel(out_sobel)
        out_sobel = self.layer3_sobel(out_sobel)
        out_sobel = self.layer4_sobel(out_sobel)
        out_sobel = F.avg_pool2d(out_sobel, 4)
        out_sobel = out_sobel.view(out_sobel.size(0), -1)

        final_out = torch.cat((out, out_sobel), dim = 1)
        final_out = self.linear(final_out)
        final_out = self.l2norm(final_out)
        return final_out
    
    
def ResNet18withSobel(low_dim=128):
    return ResNetwithSobel(BasicBlock, [2,2,2,2], low_dim)
