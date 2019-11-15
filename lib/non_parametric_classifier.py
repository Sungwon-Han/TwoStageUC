import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch import nn
import math


class NonParametricClassifierOP(Function):
    @staticmethod
    def forward(self, x, y, memory, params):

        T = params[0].item()
        batchSize = x.size(0)

        # inner product
        out = torch.mm(x.data, memory.t()) 
        out.div_(T) # batchSize * N
            
        self.save_for_backward(x, memory, y, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        '''
        Siamese network

        gradOutput: (256, 50000)
        x: (256, 128)

        memory: (50000, 128)
        y: (128)
        '''
        x, memory, y, params = self.saved_tensors
        # concat the targets for siamese network
        y = torch.cat((y, y), 0)
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()
        
        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)

        # update the memory
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        
        return gradInput, None, None, None, None


class NonParametricClassifier(nn.Module):
    """Non-parametric Classifier
    
    Non-parametric Classifier from
    "Unsupervised Feature Learning via Non-Parametric Instance Discrimination"
    
    Extends:
        nn.Module
    """

    def __init__(self, inputSize, outputSize, T, momentum):
        """Non-parametric Classifier initial functin
        
        Initial function for non-parametric classifier
        
        Arguments:
            inputSize {int} -- in-channels dims
            outputSize {int} -- out-channels dims
        
        Keyword Arguments:
            T {int} -- distribution temperate (default: {0.05})
            momentum {int} -- memory update momentum (default: {0.5})
        """
        super(NonParametricClassifier, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize

        self.register_buffer('params',
                        torch.tensor([T, momentum]))
        stdv = 1. / math.sqrt(inputSize/3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize)
                                                .mul_(2*stdv).add_(-stdv))

    def forward(self, x, y):
        out = NonParametricClassifierOP.apply(x, y, self.memory, self.params)
        return out

    def just_calculate(self, x):
        T = self.params[0].item()
        out = torch.mm(x.data, self.memory.t())
        out.div_(T)
        return out   
