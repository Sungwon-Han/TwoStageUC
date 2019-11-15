import torch
from lib.utils import AverageMeter
import time
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np


def kNN(net, npc, trainloader, testloader, K=200, sigma=0.1, recompute_memory=False, device='cpu'):
    # set the model to evaluation mode
    net.eval()

    # tracking variables
    total = 0

    trainFeatures = npc.memory
    trainLabels = torch.LongTensor(trainloader.dataset.targets).to(device)

    # recompute features for training samples
    if recompute_memory:
        trainFeatures, trainLabels = traverse(net, trainloader, 
                                    testloader.dataset.transform, device)
    trainFeatures = trainFeatures.t()
    C = trainLabels.max() + 1
    
    # start to evaluate
    top1 = 0.
    top5 = 0.
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C.item()).to(device)
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):

            batchSize = inputs.size(0)
            targets, inputs = targets.to(device), inputs.to(device)

            # forward
            features = net(inputs)

            # cosine similarity
            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C),
                                        yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()

            total += targets.size(0)

    return top1/total

def _hungarian_match(flat_preds, flat_targets, num_samples, class_num):  
    num_k = class_num
    num_correct = np.zeros((num_k, num_k))
  
    for c1 in range(0, num_k):
        for c2 in range(0, num_k):
        # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes
  
    # num_correct is small
    match = linear_assignment(num_samples - num_correct)
  
    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))
  
    return res


def test(net, testloader,device, fc, class_num):
    net.eval()
    fc.eval()
    predicted_all = [[],[],[],[],[]]
    targets_all = [[],[],[],[],[]]
    for batch_idx, (inputs, _, targets, indexes) in enumerate(testloader):
        batchSize = inputs.size(0)
        targets, inputs = targets.to(device), inputs.to(device)
        features = net(inputs)
        output_list = fc(features)
        for i in range(len(output_list)):
            _, predicted = torch.max(output_list[i], 1)
            predicted_all[i].append(predicted)
            targets_all[i].append(targets)
    
    acc_list = []
    for i in range(len(output_list)):
        flat_predict = torch.cat(predicted_all[i]).to(device)
        flat_target = torch.cat(targets_all[i]).to(device)
        num_samples = flat_predict.shape[0]
        match = _hungarian_match(flat_predict, flat_target, num_samples, class_num)
        reordered_preds = torch.zeros(num_samples).to(device)

        for pred_i, target_i in match:
            reordered_preds[flat_predict == pred_i] = int(target_i)
        acc = int((reordered_preds == flat_target.float()).sum()) / float(num_samples) * 100
        acc_list.append(acc)
        
    return acc_list


