import math
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


class MSE_Gussian(nn.Module):
    def __init__(self, device):
        super(MSE_Gussian, self).__init__()
        self.device = device
        self.mse = nn.MSELoss()
        self.blurrer = T.GaussianBlur(kernel_size=(9,9), sigma=(0.1, 20.0))

    def forward(self, preds:Tensor, labels:Tensor):
        
        labels_g = self.blurrer(labels)
        return self.mse(preds, labels_g)

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, smooth=1, p=2, ignore_index=0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.ignore_index = ignore_index
        self.reduction=reduction

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
       
        if probas.dim() == 4:
            # 2D segmentation
            B, C, H, W = probas.size()
            probas = probas.contiguous().permute(0, 2, 3, 1).contiguous().view(-1, C) # (B=1)*H*W, C

        if labels.dim() == 4:# B,C,H,W -> B,H,W
            labels_arg = torch.argmax(labels, dim=1)
            labels_arg = labels_arg.view(-1)# B,H,W -> B*H*W
            # assumes output of a sigmoid layer
            B, C, H, W = labels.size()
            labels = labels.view(B, C, H, W).permute(0, 2, 3, 1).contiguous().view(-1, C)# (B=1)*H*W, C
        
        if ignore is None:
            return probas, labels

        valid = (labels_arg != ignore)#label값이 ignore 아닌 픽셀들만 골라서
        mask = valid.nonzero(as_tuple=False).squeeze()
        vprobas = probas[mask]
        vlabels = labels[mask]

        return vprobas.contiguous().view(-1), vlabels.contiguous().view(-1)
    
    def forward(self, predicts, targets):
        loss_total=[]
        for predict, target in zip(predicts, targets):# 배치의 샘플 단위로 손실 값 측정
            predict = predict.unsqueeze(0)#(1, C, H, W)
            target = target.unsqueeze(0)#(1, C, H, W)
            
            predict, target = self.flatten_probas(predict, target, ignore=0)# #(1, C, H, W) -> (K*C)
            predict = predict.unsqueeze(0)#(1, K*C)
            target = target.unsqueeze(0)#(1, K*C)

            num = 2 * (torch.sum(predict * target, dim=1) + self.smooth)
            den = torch.sum(predict ** self.p, dim=1) + torch.sum(target ** self.p, dim=1) + self.smooth

            loss = 1 - num / den
            
            loss_total.append(loss)
 
        if self.reduction == "mean":
            return torch.mean(torch.cat(loss_total))
        elif self.reduction == "sum":
            return torch.sum(torch.cat(loss_total))
        elif self.reduction == "none":
            return torch.cat(loss_total)
        
        return loss

class CategoricalCrossEntropyLoss(nn.Module):
    """Categorical Cross Entropy loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryFocalLoss
    Return:
        same as CrossEntropyLoss
    """
    def __init__(self, weight=None, ignore_index=0, reduction='mean', **kwargs):
        super(CategoricalCrossEntropyLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction=reduction
        self.CEloss = CrossEntropyLoss(reduction=self.reduction,**self.kwargs)

    def forward(self, predicts, targets):
        if self.ignore_index==0:
            targets = targets[:,1:,:]# one-hot
            predicts = predicts[:,1:,:]# predicted prob.

        assert predicts.shape == targets.shape, 'predict & target shape do not match'
        # predict = F.softmax(predict, dim=1)  # prob

        term_true = - torch.log(predicts)
        term_false = - torch.log(1-predicts)
        loss = torch.sum(term_true * targets + term_false * (1-targets), dim=1) #torch.Size([8, 256, 512])

        if self.reduction == "mean":# torch.Size([]) # loss: 4.507612  [    0/ 2975]
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass       
        
        # loss = self.CEloss(predict,torch.argmax(target, dim=1)) # torch.Size([]) # loss: 3.569603  [    0/ 2975]
        return loss

class FocalLoss(nn.Module):
    """Focal loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryFocalLoss
    Return:
        same as BinaryFocalLoss
    """
    def __init__(self,  alpha:float = 0.25, gamma:float = 2, eps = 1e-8, ignore_index=0, reduction:str ='mean'):
        super(FocalLoss, self).__init__()
        self.nclasses = 20 
        self.ignore_index = ignore_index        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 4:
            # 2D segmentation
            B, C, H, W = probas.size()
            probas = probas.contiguous().permute(0, 2, 3, 1).contiguous().view(-1, C) # (B=1)*H*W, C

        if labels.dim() == 4:# B,C,H,W -> B,H,W
            labels_arg = torch.argmax(labels, dim=1)
            labels_arg = labels_arg.view(-1)# B,H,W -> B*H*W

        if labels.dim() == 4:# B,C,H,W -> B*H*W, C
            # assumes output of a sigmoid layer
            B, C, H, W = labels.size()
            labels = labels.view(B, C, H, W).permute(0, 2, 3, 1).contiguous().view(-1, C)# (B=1)*H*W, C
        
        if ignore is None:
            return probas, labels

        valid = (labels_arg != ignore)
        vprobas = probas[valid.nonzero(as_tuple=False).squeeze()] 
        vlabels = labels[valid.nonzero(as_tuple=False).squeeze()] 
        return vprobas, vlabels
    
    def per_class(self, t):
        per_class = torch.zeros([t.shape[0], self.nclasses, t.shape[1], t.shape[2]]).cuda()

        for i in range(self.nclasses):
            per_class[:,i] = torch.where(t==i, 1, 0)
        
        return per_class
    
    def forward(self, predicts: Tensor, targets: Tensor):     # target b h w 
        loss_total=[]
        # loss_total = torch.zeros(predicts.size(0), 1)
        targets = self.per_class(targets) # b h w -> b c h w
        for predict, target in zip(predicts, targets):
        # for i, (predict, target) in enumerate(zip(predicts, targets)):
            predict = predict.unsqueeze(0)
            target = target.unsqueeze(0)
            
            predict, target = self.flatten_probas(predict, target, ignore=0) # (1, C, H, W) -> (K,C)

            term_true =  - self.alpha * ((1 - predict) ** self.gamma) * torch.log(predict+self.eps) # 틀리면 손실 커짐, 맞을수록 작아짐
            term_false = - (1-self.alpha) * (predict**self.gamma) * torch.log(1-predict+self.eps) # 틀리면 손실 커짐, 맞을수록 작아짐

            loss = torch.sum(term_true * target + term_false * (1-target), dim=-1)# (1*K) 

            # predict, target = predict.detach().cpu().numpy(), target.detach().cpu().numpy()
            # term_true = -self.alpha * ((1 - predict)**self.gamma) * np.log(predict + self.eps)
            # term_false = -(1 - self.alpha) * (predict**self.gamma) * np.log(1 - predict + self.eps)
            # loss = np.sum(term_true * target + term_false * (1 - target), axis=-1)
            loss_total.append(loss)
            # loss_total[i] = loss
 
        if self.reduction == "mean":
            return torch.mean(torch.cat(loss_total))
            # return np.mean(np.concatenate(loss_total))
        elif self.reduction == "sum":
            return torch.sum(torch.cat(loss_total))
        elif self.reduction == "none":
            return torch.cat(loss_total)


class FocalLoss_jg(nn.Module):
    """Focal loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryFocalLoss
    Return:
        same as BinaryFocalLoss
    """
    def __init__(self,  alpha:float = 0.25, gamma:float = 2, eps = 1e-8, ignore_index=0, reduction:str ='mean'):
        super(FocalLoss_jg, self).__init__()
        self.ignore_index = ignore_index        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
       
        if probas.dim() == 4:
            # 2D segmentation
            B, C, H, W = probas.size()
            probas = probas.contiguous().permute(0, 2, 3, 1).contiguous().view(-1, C) # (B=1)*H*W, C

        if labels.dim() == 4:# B,C,H,W -> B,H,W
            labels_arg = torch.argmax(labels, dim=1)
            labels_arg = labels_arg.view(-1)# B,H,W -> B*H*W

        if labels.dim() == 4:# B,C,H,W -> B*H*W, C
            # assumes output of a sigmoid layer
            B, C, H, W = labels.size()
            labels = labels.view(B, C, H, W).permute(0, 2, 3, 1).contiguous().view(-1, C)# (B=1)*H*W, C
        
        if ignore is None:
            return probas, labels

        valid = (labels_arg != ignore)#label값이 ignore 아닌 픽셀들만 골라서
        vprobas = probas[valid.nonzero(as_tuple=False).squeeze()] #추려냄
        vlabels = labels[valid.nonzero(as_tuple=False).squeeze()] #마찬가지로 추려냄

        return vprobas, vlabels
    
    def forward(self, predicts: Tensor, targets: Tensor):     
        loss_total=[]
        for predict, target in zip(predicts, targets):# 배치의 샘플 단위로 손실 값 측정
            predict = predict.unsqueeze(0)#(1, C, H, W)
            target = target.unsqueeze(0)#(1, C, H, W)
            
            predict, target = self.flatten_probas(predict, target, ignore=0)# #(1, C, H, W) -> (K,C)

            term_true =  - self.alpha * ((1 - predict) ** self.gamma) * torch.log(predict+self.eps) # 틀리면 손실 커짐, 맞을수록 작아짐
            term_false = - (1-self.alpha) * (predict**self.gamma) * torch.log(1-predict+self.eps) # 틀리면 손실 커짐, 맞을수록 작아짐

            loss = torch.sum(term_true * target + term_false * (1-target), dim=-1)# (1*K) 
            
            loss_total.append(loss)
 
        if self.reduction == "mean":
            return torch.mean(torch.cat(loss_total))
        elif self.reduction == "sum":
            return torch.sum(torch.cat(loss_total))
        elif self.reduction == "none":
            return torch.cat(loss_total)


class FocalLoss2(nn.Module):
    """Focal loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryFocalLoss
    Return:
        same as BinaryFocalLoss
    """
    def __init__(self,  alpha:float = 0.25, gamma:float = 2, eps = 1e-8, ignore_index=0, reduction:str ='mean'):
        super(FocalLoss2, self).__init__()
        self.nclasses = 20 
        self.ignore_index = ignore_index        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 4:
            # 2D segmentation
            B, C, H, W = probas.size()
            probas = probas.contiguous().permute(0, 2, 3, 1).contiguous().view(-1, C) # (B=1)*H*W, C

        if labels.dim() == 4:# B,C,H,W -> B,H,W
            labels_arg = torch.argmax(labels, dim=1)
            labels_arg = labels_arg.view(-1)# B,H,W -> B*H*W

        if labels.dim() == 4:# B,C,H,W -> B*H*W, C
            # assumes output of a sigmoid layer
            B, C, H, W = labels.size()
            labels = labels.view(B, C, H, W).permute(0, 2, 3, 1).contiguous().view(-1, C)# (B=1)*H*W, C
        
        if ignore is None:
            return probas, labels

        valid = (labels_arg != ignore)
        vprobas = probas[valid.nonzero(as_tuple=False).squeeze()] 
        vlabels = labels[valid.nonzero(as_tuple=False).squeeze()] 
        return vprobas, vlabels
    
    def per_class(self, t):
        per_class = torch.zeros([t.shape[0], self.nclasses, t.shape[1], t.shape[2]]).cuda()

        for i in range(self.nclasses):
            per_class[:,i] = torch.where(t==i, 1, 0)
        
        return per_class
    
    def forward(self, predicts: Tensor, targets: Tensor):     # target b h w 
        loss_total=[]
        targets = self.per_class(targets) # b h w -> b c h w
        for predict, target in zip(predicts, targets):
            predict = predict.unsqueeze(0)
            target = target.unsqueeze(0)
            
            predict, target = self.flatten_probas(predict, target, ignore=0) # (1, C, H, W) -> (K,C)

            term_true =  - self.alpha * ((1 - predict) ** self.gamma) * torch.log(predict+self.eps) # 틀리면 손실 커짐, 맞을수록 작아짐
            term_false = - (1-self.alpha) * (predict**self.gamma) * torch.log(1-predict+self.eps) # 틀리면 손실 커짐, 맞을수록 작아짐

            loss = torch.sum(term_true * target + term_false * (1-target), dim=-1)# (1*K) 

            # predict, target = predict.detach().cpu().numpy(), target.detach().cpu().numpy()
            # term_true = -self.alpha * ((1 - predict)**self.gamma) * np.log(predict + self.eps)
            # term_false = -(1 - self.alpha) * (predict**self.gamma) * np.log(1 - predict + self.eps)
            # loss = np.sum(term_true * target + term_false * (1 - target), axis=-1)
            loss_total.append(loss)
 
        if self.reduction == "mean":
            return torch.mean(torch.cat(loss_total))
            # return np.mean(np.concatenate(loss_total))
        elif self.reduction == "sum":
            return torch.sum(torch.cat(loss_total))
        elif self.reduction == "none":
            return torch.cat(loss_total)
        
class Focal_3D_loss(nn.Module):
    def __init__(self, alpha:float=0.25, gamma:float= 2, eps=1e-8, ignore_index=0, reduction:str = 'mean'):
        super(Focal_3D_loss, self).__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction
    
    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 3:# B,C,N -> B*N, C
            # assumes output of a sigmoid layer
            B, C, N = probas.size()
            probas = probas.view(B, C, 1, N).permute(0, 2, 3, 1).contiguous().view(-1, C)

        if labels.dim() == 3 :# B,C,N -> B,N
            # assumes output of a sigmoid layer
            B, C, N = labels.size()
            labels = labels.view(B, C, 1, N).permute(0, 2, 3, 1).contiguous().view(-1, C)

        elif labels.dim()==4:
            B, C, H, W = labels.size()
            labels = labels.permute(0, 2, 3, 1).contiguous().view(-1, C)

        labels_arg = torch.argmax(labels, dim=1)
        labels_arg = labels_arg.view(-1)# B,N -> B*N
        if ignore is None:
            return probas, labels
        
        valid = (labels_arg != ignore)#label값이 ignore 아닌 픽셀들만 골라서

        mask = valid.nonzero(as_tuple=False).squeeze()
        vprobas = probas[mask]
        vlabels = labels[mask]

        return vprobas, vlabels
    
    def forward(self, predicts:Tensor, targets:Tensor):
        loss_total=[]
        for predict, target in zip(predicts, targets):
            predict = predict.unsqueeze(0)
            target = target.unsqueeze(0)
            
            predict, target = self.flatten_probas(predict, target, ignore=0)# (1*K, C) | (1*K, C)

            term_true =  -self.alpha * ((1-predict) ** self.gamma) * torch.log(predict+self.eps) 
            # term_true =  -self.alpha * ((1-predict) ** self.gamma) * torch.log(predict) 
            term_false = -(1-self.alpha) * (predict ** self.gamma) * torch.log(1-predict+self.eps)

            loss = torch.sum(term_true * target + term_false * (1-target), dim=-1)# (1*K) 
            
            loss_total.append(loss)
 
        if self.reduction == "mean":
            return torch.mean(torch.cat(loss_total))
        elif self.reduction == "sum":
            return torch.sum(torch.cat(loss_total))
        elif self.reduction == "none":
            return torch.cat(loss_total)


class FocalLosswithDiceRegularizer_jg(nn.Module):
    """Focal loss with Dice loss as regularizer, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryFocalLoss
    Return:
        same as BinaryFocalLoss
    """
    def __init__(self,  alpha:float = 0.75, gamma:float = 2, eps = 1e-8, smooth=1, p=2,  ignore_index=0, reduction:str ='sum'):
        super(FocalLosswithDiceRegularizer_jg, self).__init__()
        self.ignore_index = ignore_index        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.smooth = smooth
        self.p = p
        self.focal_loss = FocalLoss_jg(alpha = self.alpha, gamma = self.gamma, eps = self.eps, ignore_index = self.ignore_index, reduction=reduction)
        self.dice_regularizer = DiceLoss(smooth=self.smooth, p=self.p, ignore_index = self.ignore_index, reduction=reduction)

    def forward(self, predict: Tensor, target: Tensor):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        predict = F.softmax(predict, dim=1) + self.eps # prob
        f_loss = self.focal_loss(predict, target)
        # d_regularization = self.dice_regularizer(predict*target, target)
        # return f_loss + (8 * d_regularization)
        d_loss = self.dice_regularizer(predict, target)
        return f_loss + d_loss

class FocalLosswithDiceRegularizer(nn.Module):
    """Focal loss with Dice loss as regularizer, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryFocalLoss
    Return:
        same as BinaryFocalLoss
    """
    def __init__(self,  alpha:float = 0.75, gamma:float = 2, eps = 1e-8, smooth=1, p=2,  ignore_index=0, reduction:str ='sum'):
        super(FocalLosswithDiceRegularizer, self).__init__()
        self.ignore_index = ignore_index        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.smooth = smooth
        self.p = p
        self.focal_loss = FocalLoss(alpha = self.alpha, gamma = self.gamma, eps = self.eps, ignore_index = self.ignore_index, reduction=reduction)
        self.dice_regularizer = DiceLoss(smooth=self.smooth, p=self.p, ignore_index = self.ignore_index, reduction=reduction)

    def forward(self, predict: Tensor, target: Tensor):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        # predict = F.softmax(predict, dim=1) + self.eps # prob
        f_loss = self.focal_loss(predict, target)
        # d_regularization = self.dice_regularizer(predict*target, target)
        # return f_loss + (8 * d_regularization)
        d_loss = self.dice_regularizer(predict, target)
        return f_loss + d_loss

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, classes, ignore_index, smoothing=0.1, dim=-1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        if self.ignore_index==0:
            target = target[:,1:,:]# one-hot
            pred = pred[:,1:,:]# predicted prob.

        pred = pred.log_softmax(dim=self.dim)
        target = torch.argmax(target, dim=1)
        # true_dist = pred.data.clone()
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class Lovasz_loss(nn.Module):
    def __init__(self, nclasses=20, reduction:str='mean', ignore_index:int = 0):
        super(Lovasz_loss, self).__init__()
        self.reduction = reduction
        self.ignore_idx = ignore_index 
        self.nclasses = nclasses
    
    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """    
        def isnan(x):
            return x != x

        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            # acc += v
            # 수정했음
            if (v.sum()==0).item():
                acc += v.mean()
            else:
                acc += v
        if n == 1:
            return acc
        return acc / n

    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1] # jaccard index에 가장 큰 불이익을 주는 오류를 최소화
        return jaccard   


    def lovasz_softmax_flat(self, probas, labels, classes='present'):
        """
        Multi-class Lovasz-Softmax loss
        probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = (labels == c).float()  # foreground for class c
            if (classes == 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (Variable(fg) - class_pred).abs()# target - pred
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, Variable(self.lovasz_grad(fg_sorted)))) # 정렬된 오류와 gradient of the Lovasz extension사이의 내적을 통해 최종 loss전달

        return self.mean(losses) 

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 3:# B,C,N -> B*N, C
            # assumes output of a sigmoid layer
            B, C, N = probas.size()
            probas = probas.view(B, C, 1, N).permute(0, 2, 3, 1).contiguous().view(-1, C)
        
        elif probas.dim() == 5:
            # 3D segmentation
            B, C, L, H, W = probas.size()
            probas = probas.contiguous().permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        # if labels.dim() == 3:# B,C,N -> B,N
        #     labels = torch.argmax(labels, dim=1)
        # labels = labels.view(-1)# B,N -> B*N
        if ignore is None:
            return probas, labels
        labels = labels.reshape(-1)
        valid = (labels != ignore) #label값이 ignore 아닌 픽셀들만 골라서
        vprobas = probas[valid.nonzero(as_tuple=False).squeeze()] #추려냄
        vlabels = labels[valid]#마찬가지로 추려냄

        return vprobas, vlabels
        
    def lovasz_softmax(self, probas, labels, classes='present', per_image=True, ignore=0):
        """
        Multi-class Lovasz-Softmax loss
        probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        per_image: compute the loss per image instead of per batch
        ignore: void class labels
        """
        #우리는 per_image == True 이걸로 쓰면 될 듯
        #ignore=0으로 배경과 3D point 아닌 애들을 걸러내기 
        #label이 3차원으로 들어오는지 확인 필요

        if per_image: # mean reduction
            loss = self.mean(self.lovasz_softmax_flat(*self.flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes) for prob, lab in zip(probas, labels))
        else: # sum reduction
            loss = self.lovasz_softmax_flat(*self.flatten_probas(probas, labels, ignore), classes=classes)
        return loss


    def forward(self, uv_out, uv_label):
        lovasz_loss = self.lovasz_softmax(uv_out, uv_label, ignore=self.ignore_idx)
        return lovasz_loss

class FocalLosswithLovaszRegularizer(nn.Module):
    def __init__(self,  alpha:float = 0.75, gamma:float = 2, eps = 1e-8, smooth=1, p=2, reduction:str = 'mean', ignore_index:int = 0):
        super(FocalLosswithLovaszRegularizer, self).__init__()
        self.reduction = reduction
        self.ignore_idx = ignore_index
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Focal_loss = FocalLoss(alpha = self.alpha, gamma = self.gamma, eps = self.eps, ignore_index = self.ignore_idx, reduction=reduction)
        self.Lovasze_loss = Lovasz_loss(reduction=self.reduction, ignore_index=self.ignore_idx)

    def forward(self,pred:Tensor, label:Tensor):
        # assert pred.shape == label.shape, 'predict & target shape do not match'
        # pred = F.softmax(pred, dim=1)
        f_loss = self.Focal_loss(pred, label)
        # lovasz_regularization = self.Lovasze_loss(pred*label, label)
        # return f_loss + (8 * lovasz_regularization)
        lovasz_loss = self.Lovasze_loss(pred, label)
        return f_loss + lovasz_loss
    
class Lovasz_loss_jg(nn.Module):
    def __init__(self, reduction:str='mean', ignore_index:int = 0):
        super(Lovasz_loss_jg, self).__init__()
        self.reduction = reduction
        self.ignore_idx = ignore_index
    
    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """    
        def isnan(x):
            return x != x

        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n

    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1] # jaccard index에 가장 큰 불이익을 주는 오류를 최소화
        return jaccard   


    def lovasz_softmax_flat(self, probas, labels, classes='present'):
        """
        Multi-class Lovasz-Softmax loss
        probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        #ignore 클래스가 제거된 B*N-(num. ignore)개의 픽셀들의 예측값과 라벨이 넘어오고
        #그럼 그게 결국 N개의 3D 포인트의 예측값만 남는거
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.
        C = probas.size(1)#클래스 수
        losses = []
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = (labels == c).float()  # foreground for class c
            if (classes == 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (Variable(fg) - class_pred).abs()# target - pred
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, Variable(self.lovasz_grad(fg_sorted)))) # 정렬된 오류와 gradient of the Lovasz extension사이의 내적을 통해 최종 loss전달
        return self.mean(losses)

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 3:# B,C,N -> B*N, C
            # assumes output of a sigmoid layer
            B, C, N = probas.size()
            probas = probas.view(B, C, 1, N).permute(0, 2, 3, 1).contiguous().view(-1, C)
        
        elif probas.dim() == 5:
            # 3D segmentation
            B, C, L, H, W = probas.size()
            probas = probas.contiguous().permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        # B, C, H, W = probas.size()
        # probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        if labels.dim() == 3:# B,C,N -> B,N
            labels = torch.argmax(labels, dim=1)
        labels = labels.view(-1)# B,N -> B*N
        if ignore is None:
            return probas, labels
        valid = (labels != ignore)#label값이 ignore 아닌 픽셀들만 골라서
        vprobas = probas[valid.nonzero(as_tuple=False).squeeze()] #추려냄
        vlabels = labels[valid]#마찬가지로 추려냄

        return vprobas, vlabels
    
    def lovasz_softmax(self, probas, labels, classes='present', per_image=True, ignore=0):
        """
        Multi-class Lovasz-Softmax loss
        probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        per_image: compute the loss per image instead of per batch
        ignore: void class labels
        """
        #우리는 per_image == True 이걸로 쓰면 될 듯
        #ignore=0으로 배경과 3D point 아닌 애들을 걸러내기 
        #label이 3차원으로 들어오는지 확인 필요

        if per_image:# mean reduction
            loss = self.mean(self.lovasz_softmax_flat(*self.flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                        for prob, lab in zip(probas, labels))
        else:# sum reduction
            loss = self.lovasz_softmax_flat(*self.flatten_probas(probas, labels, ignore), classes=classes)
        return loss

    def forward(self, uv_out, uv_label):
        lovasz_loss = self.lovasz_softmax(uv_out, uv_label, ignore=self.ignore_idx)
        return lovasz_loss
    
class FocalLosswithLovaszRegularizer_jg(nn.Module):
    def __init__(self,  alpha:float = 0.75, gamma:float = 2, eps = 1e-8, smooth=1, p=2, reduction:str = 'mean', ignore_index:int = 0):
        super(FocalLosswithLovaszRegularizer_jg, self).__init__()
        self.reduction = reduction
        self.ignore_idx = ignore_index
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Focal_loss = Focal_3D_loss(alpha = self.alpha, gamma = self.gamma, eps = self.eps, ignore_index = self.ignore_idx, reduction=reduction)
        self.Lovasze_loss = Lovasz_loss_jg(reduction=self.reduction, ignore_index=self.ignore_idx)

    def forward(self,pred:Tensor, label:Tensor):
        # assert pred.shape == label.shape, 'predict & target shape do not match'
        pred = F.softmax(pred, dim=1)
        f_loss = self.Focal_loss(pred, label)
        # lovasz_regularization = self.Lovasze_loss(pred*label, label)
        # return f_loss + (8 * lovasz_regularization)
        lovasz_loss = self.Lovasze_loss(pred, label)
        return f_loss + lovasz_loss

class FocalLosswithLovaszRegularizer2(nn.Module):
    def __init__(self,  alpha:float = 0.75, gamma:float = 2, eps = 1e-8, smooth=1, p=2, reduction:str = 'mean', ignore_index:int = 0):
        super(FocalLosswithLovaszRegularizer2, self).__init__()
        self.reduction = reduction
        self.ignore_idx = ignore_index
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Focal_loss = FocalLoss2(alpha = self.alpha, gamma = self.gamma, eps = self.eps, ignore_index = self.ignore_idx, reduction=reduction)
        self.Lovasze_loss = Lovasz_loss(reduction=self.reduction, ignore_index=self.ignore_idx)

    def forward(self,pred:Tensor, label:Tensor):
        # assert pred.shape == label.shape, 'predict & target shape do not match'
        # pred = F.softmax(pred, dim=1)
        f_loss = self.Focal_loss(pred, label)
        # lovasz_regularization = self.Lovasze_loss(pred*label, label)
        # return f_loss + (8 * lovasz_regularization)
        lovasz_loss = self.Lovasze_loss(pred, label)
        return f_loss + lovasz_loss

class total_loss(nn.Module):
    def __init__(self, reduction:str ='sum', ignore_index:int = 0):
        super(total_loss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.loss2D = FocalLosswithDiceRegularizer_jg(reduction=self.reduction, ignore_index = self.ignore_index)
        self.loss_3D = FocalLosswithLovaszRegularizer_jg(reduction=self.reduction, ignore_index=self.ignore_index)
        # self.KLD_reg_3D = KLDivLoss(reduction='mean')
        self.transform = Rearrange('b e h w  -> b e (h w)')
        self.alpha = 1
        self.klDiv = nn.KLDivLoss(reduction="none")
        
    def forward(self, segment_out, segment_label, uv_out, uv_label):

        # 2D segmentation loss
        loss_2D_result = self.loss2D(segment_out, segment_label)

        # 3D segmentation loss
        loss_3D_result = self.loss_3D(uv_out, uv_label)

        # mae || KLDiv
        # loss_3D_regularizer = self.kl_div_3D(segment_out.detach(), uv_out) * self.alpha 

        # return loss_2D_result + loss_3D_result + loss_3D_regularizer.clamp(min=1e-8)

        # PerceptionAwareLoss 사용할 시
        # loss_per, pcd_guide_weight, img_guide_weight = self.PerceptionAwareLoss(uv_out, segment_out)
        return loss_2D_result + loss_3D_result #+ loss_per*0.5 #0.5는 해당 논문에서 사용하는 alpha값

class FocalLosswithLovaszRegularizer_cpu(nn.Module):
    def __init__(self,  alpha:float = 0.75, gamma:float = 2, eps = 1e-8, smooth=1, p=2, reduction:str = 'mean', ignore_index:int = 0):
        super(FocalLosswithLovaszRegularizer_cpu, self).__init__()
        self.reduction = reduction
        self.ignore_idx = ignore_index
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Focal_loss = FocalLoss_cpu(alpha = self.alpha, gamma = self.gamma, eps = self.eps, ignore_index = self.ignore_idx, reduction=reduction)
        self.Lovasze_loss = Lovasz_loss_cpu(reduction=self.reduction, ignore_index=self.ignore_idx)

    def forward(self,pred:Tensor, label:Tensor):
        pred, label = pred.detach().cpu().numpy(), label.detach().cpu().numpy()
        f_loss = self.Focal_loss(pred, label)
        lovasz_loss = self.Lovasze_loss(pred, label)
        return f_loss + lovasz_loss    

class FocalLoss_cpu(nn.Module):
    """Focal loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryFocalLoss
    Return:
        same as BinaryFocalLoss
    """
    def __init__(self,  alpha:float = 0.25, gamma:float = 2, eps = 1e-8, ignore_index=0, reduction:str ='mean'):
        super(FocalLoss_cpu, self).__init__()
        self.nclasses = 20 
        self.ignore_index = ignore_index        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.ndim == 4:
            # 2D segmentation
            probas = rearrange(probas, 'b c h w -> b h w c').reshape(-1, probas.shape[1])
            # probas = probas.contiguous().permute(0, 2, 3, 1).contiguous().view(-1, C) # (B=1)*H*W, C

        if labels.ndim == 4:# B,C,H,W -> B,H,W
            labels_arg = np.argmax(labels, axis=1)
            labels_arg = labels_arg.reshape(-1)
            # labels_arg = labels_arg.view(-1)# B,H,W -> B*H*W

            # assumes output of a sigmoid layer
            labels = rearrange(labels, 'b c h w -> b h w c').reshape(-1, labels.shape[1])
            # labels = labels.view(B, C, H, W).permute(0, 2, 3, 1).contiguous().view(-1, C)# (B=1)*H*W, C
        
        if ignore is None:
            return probas, labels

        valid = (labels_arg != ignore)
        vprobas = probas[np.squeeze(np.nonzero(valid))]
        vlabels = labels[np.squeeze(np.nonzero(valid))]
        # vprobas = probas[valid.nonzero(as_tuple=False).squeeze()] 
        # vlabels = labels[valid.nonzero(as_tuple=False).squeeze()] 
        return vprobas, vlabels
    
    def per_class(self, t):
        per_class = np.zeros([t.shape[0], self.nclasses, t.shape[1], t.shape[2]])

        for i in range(self.nclasses):
            per_class[:,i] = np.where(t==i, 1, 0)
        
        return per_class
    
    def forward(self, predicts: Tensor, targets: Tensor):     # target b h w 
        loss_total=[]
        targets = self.per_class(targets) # b h w -> b c h w
        for predict, target in zip(predicts, targets):
            predict = np.expand_dims(predict, axis=0)
            target = np.expand_dims(target, axis=0)
            
            predict, target = self.flatten_probas(predict, target, ignore=0) # (1, C, H, W) -> (K,C)

            term_true =  -self.alpha * ((1 - predict) ** self.gamma) * np.log(predict + self.eps) # 틀리면 손실 커짐, 맞을수록 작아짐
            term_false = -(1 - self.alpha) * (predict ** self.gamma) * np.log(1 - predict + self.eps) # 틀리면 손실 커짐, 맞을수록 작아짐

            loss = np.sum(term_true * target + term_false * (1 - target), axis=-1)
            loss_total.append(loss)
 
        if self.reduction == "mean":
            # return torch.mean(torch.cat(loss_total))
            return np.mean(np.concatenate(loss_total))
        elif self.reduction == "sum":
            return torch.sum(torch.cat(loss_total))
        elif self.reduction == "none":
            return torch.cat(loss_total)
       
class Lovasz_loss_cpu(nn.Module):
    def __init__(self, nclasses=20, reduction:str='mean', ignore_index:int = 0):
        super(Lovasz_loss_cpu, self).__init__()
        self.reduction = reduction
        self.ignore_idx = ignore_index 
        self.nclasses = nclasses
    
    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """    
        def isnan(x):
            return x != x

        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n

    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = np.sum(gt_sorted)
        intersection = gts - np.cumsum(gt_sorted.astype(np.float), 0)
        union = gts + np.cumsum((1 - gt_sorted).astype(np.float), 0)
        jaccard = 1. - intersection/union

        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1] # jaccard index에 가장 큰 불이익을 주는 오류를 최소화

        return jaccard   


    def lovasz_softmax_flat(self, probas, labels, classes='present'):
        """
        Multi-class Lovasz-Softmax loss
        probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        C = probas.shape[1]
        losses = []
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = (labels == c).astype(np.float)  # foreground for class c
            if (classes == 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = np.abs(fg - class_pred) 
            errors_sorted = np.sort(errors, 0)[::-1]
            perm = np.argsort(errors, 0)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(np.dot(errors_sorted, self.lovasz_grad(fg_sorted)))
            # losses.append(torch.dot(errors_sorted, Variable(self.lovasz_grad(fg_sorted)))) # 정렬된 오류와 gradient of the Lovasz extension사이의 내적을 통해 최종 loss전달
        return self.mean(losses)

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """

        B, C, H, W = probas.shape
        probas = rearrange(probas, 'b c h w -> b h w c').reshape(-1, probas.shape[1])
        # probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        # if labels.dim() == 3:# B,C,N -> B,N
        #     labels = torch.argmax(labels, dim=1)
        # labels = labels.view(-1)# B,N -> B*N
        if ignore is None:
            return probas, labels
        labels = labels.reshape(-1)
        valid = (labels != ignore)
        vprobas = probas[np.squeeze(np.nonzero(valid))]
        # vprobas = probas[valid.nonzero(as_tuple=False).squeeze()]
        vlabels = labels[valid]

        return vprobas, vlabels
    
    def flatten_probas_cpu(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.shape[1] == 3:# B,C,N -> B*N, C
            # assumes output of a sigmoid layer
            B, C, N = probas.shape()
            probas = probas.view(B, C, 1, N).permute(0, 2, 3, 1).contiguous().view(-1, C)
        
        elif probas.dim() == 5:
            # 3D segmentation
            B, C, L, H, W = probas.size()
            probas = probas.contiguous().permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        # if labels.dim() == 3:# B,C,N -> B,N
        #     labels = torch.argmax(labels, dim=1)
        # labels = labels.view(-1)# B,N -> B*N
        if ignore is None:
            return probas, labels
        labels = labels.reshape(-1)
        valid = (labels != ignore) #label값이 ignore 아닌 픽셀들만 골라서
        vprobas = probas[valid.nonzero(as_tuple=False).squeeze()] #추려냄
        vlabels = labels[valid]#마찬가지로 추려냄

        return vprobas, vlabels
        
    def lovasz_softmax(self, probas, labels, classes='present', per_image=True, ignore=0):
        """
        Multi-class Lovasz-Softmax loss
        probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        per_image: compute the loss per image instead of per batch
        ignore: void class labels
        """

        if per_image: # mean reduction
            loss = self.mean(self.lovasz_softmax_flat(*self.flatten_probas(np.expand_dims(prob, 0), np.expand_dims(lab, 0), ignore), classes=classes) for prob, lab in zip(probas, labels))
        else: # sum reduction
            loss = self.lovasz_softmax_flat(*self.flatten_probas(probas, labels, ignore), classes=classes)
        return loss

    def forward(self, uv_out, uv_label):
        lovasz_loss = self.lovasz_softmax(uv_out, uv_label, ignore=self.ignore_idx)
        return lovasz_loss