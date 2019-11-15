import torch
import torch.nn.functional as F
import torch.nn as nn
from lib.normalize import Normalize
import sys

class UELoss(nn.Module):
    def __init__(self):
        super(UELoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

def IID_loss(x_out, x_tf_out, EPS=sys.float_info.epsilon):
    """
    This code is from IIC github
    """
    
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j) - torch.log(p_j) - torch.log(p_i))
    loss = loss.sum()
    return loss

def compute_joint(x_out, x_tf_out):
    """
    This code is from IIC github
    """

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise
  
    return p_i_j

        

class Criterion(nn.Module):
    def __init__(self, negM, T, batchSize, device):
        super(Criterion, self).__init__()
        self.negM = negM
        self.T = T
        self.diag_mat = 1 - torch.eye(batchSize*2).to(device)
        self.l2norm = Normalize(2)

    def forward(self, x, y):
        batchSize = x.size(0)
        reordered_x = torch.cat((x.narrow(0,batchSize//2,batchSize//2),\
                x.narrow(0,0,batchSize//2)), 0)
        #reordered_x = reordered_x.data
        pos = (x*reordered_x.data).sum(1).div_(self.T).exp_()

        #get all innerproduct, remove diag
        all_prob = torch.mm(x,x.t().data).div_(self.T).exp_()*self.diag_mat
        if self.negM==1:
            all_div = all_prob.sum(1)
        else:
            #remove pos for neg
            all_div = (all_prob.sum(1) - pos)*self.negM + pos

        lnPmt = torch.div(pos, all_div)

        # negative probability
        Pon_div = all_div.repeat(batchSize,1)
        lnPon = torch.div(all_prob, Pon_div.t())
        lnPon = -lnPon.add(-1)
        
        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        lnPmt.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        # negative multiply m
        lnPonsum = lnPonsum * self.negM
        aug_loss = - (lnPmtsum + lnPonsum)/batchSize

        return aug_loss

class Criterion_SAND(nn.Module):
    def __init__(self, negM, T, batchSize, device):
        super(Criterion_SAND, self).__init__()
        self.negM = negM
        self.T = T
        self.diag_mat = 1 - torch.eye(batchSize*2).to(device)
        self.l2norm = Normalize(2)

    def __split(self, y, ANs):
        pos = ANs.position.index_select(0, y.view(-1))
        return (pos >= 0).nonzero().view(-1), (pos < 0).nonzero().view(-1)

    def forward(self, x, y, ANs):
        batchSize = x.size(0)

        '''
        AND loss
        '''
        # split anchor and instance list
        anchor_indexes, instance_indexes = self.__split(y, ANs)
        x1 = x[0:int(batchSize*0.5)]
        x2 = x[int(batchSize*0.5):]
        preds1 = F.softmax(x1, 1)
        preds2 = F.softmax(x2, 1)

        l_inst = 0.
        if instance_indexes.size(0) > 0:
            # compute loss for instance samples
            y_inst = y.index_select(0, instance_indexes)
            x1_inst = preds1.index_select(0, instance_indexes)
            x2_inst = preds2.index_select(0, instance_indexes)
            # p_i = p_{i, i}
            x1_inst = x1_inst.gather(1, y_inst.view(-1, 1))
            x2_inst = x2_inst.gather(1, y_inst.view(-1, 1))
            # NLL: l = -log(p_i)v
            l_inst_x1 = -1 * torch.log(x1_inst).sum(0)
            l_inst_x2 = -1 * torch.log(x2_inst).sum(0)
            l_inst = l_inst_x1 + l_inst_x2
            
        l_ans = 0.
        if anchor_indexes.size(0) > 0:
            # compute loss for anchor samples
            y_ans = y.index_select(0, anchor_indexes)
            y_ans_neighbour = ANs.position.index_select(0, y_ans)
            neighbours = ANs.neighbours.index_select(0, y_ans_neighbour)
            # p_i = \sum_{j \in \Omega_i} p_{i,j}
            x1_ans = preds1.index_select(0, anchor_indexes)
            x2_ans = preds2.index_select(0, anchor_indexes)
            x1_ans_neighbour = x1_ans.gather(1, neighbours).sum(1)
            x2_ans_neighbour = x2_ans.gather(1, neighbours).sum(1)
            x1_ans = x1_ans.gather(1, y_ans.view(-1, 1)).view(-1) + x1_ans_neighbour
            x2_ans = x2_ans.gather(1, y_ans.view(-1, 1)).view(-1) + x2_ans_neighbour
            # NLL: l = -log(p_i)
            l_ans_x1 = -1 * torch.log(x1_ans).sum(0)
            l_ans_x2 = -1 * torch.log(x2_ans).sum(0)
            l_ans = l_ans_x1 + l_ans_x2

        AND_loss = (l_inst + l_ans) / batchSize

        '''
        Augmentation loss
        '''
        # outputs should be features of l2 norm
        x = self.l2norm(x)

        #get positive innerproduct
        reordered_x = torch.cat((x.narrow(0,batchSize//2,batchSize//2),\
                x.narrow(0,0,batchSize//2)), 0)
        #reordered_x = reordered_x.data
        pos = (x*reordered_x.data).sum(1).div_(self.T).exp_()

        #get all innerproduct, remove diag
        all_prob = torch.mm(x,x.t().data).div_(self.T).exp_()*self.diag_mat
        if self.negM==1:
            all_div = all_prob.sum(1)
        else:
            #remove pos for neg
            all_div = (all_prob.sum(1) - pos)*self.negM + pos

        lnPmt = torch.div(pos, all_div)

        # negative probability
        Pon_div = all_div.repeat(batchSize,1)
        lnPon = torch.div(all_prob, Pon_div.t())
        lnPon = -lnPon.add(-1)
        
        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        lnPmt.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        # negative multiply m
        lnPonsum = lnPonsum * self.negM
        aug_loss = - (lnPmtsum + lnPonsum)/batchSize

        return AND_loss + aug_loss


