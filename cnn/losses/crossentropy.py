import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_ce(input_, target, weight=None):
    
    n, c, d, h, w = input_.size()
    target = target.cpu()
    reqd_dim = d * h * w
    gt = []
    for batch in range(n):
        gt_init = target[batch] * torch.FloatTensor(reqd_dim).fill_(1.)
        gt = torch.cat([torch.Tensor(gt), gt_init], dim=0)
        
    input_ = input_.transpose(1, 2).transpose(2, 3).transpose(3,4).contiguous().view(-1, c)
    
    #print(target)
    #print(gt)
    #target = target.expand(-1,N)
    loss = F.cross_entropy(input_, gt.long().cuda(),reduction='mean')
                                                
    return loss


class categorical_cross_entropy(nn.Module):
    '''
    Class wrapper to loss_ce
    '''
    def __init__(self):
        super(categorical_cross_entropy, self).__init__()

    def forward(self, input_, target):
        return loss_ce(input_, target)
