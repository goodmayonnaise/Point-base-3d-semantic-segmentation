import math, cv2
import numpy as np 
from scipy.stats import norm
from einops import rearrange

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F



def similarity(e_features, d_features, model_path):
    total_sim, total_e_spars, total_d_spars = [], [], []
    level = 1
    for e_f, d_f, in zip(e_features, d_features): # f5~f1 (level1 ~ level5) size up

        # spars_ef = (e_f<=0).sum()/(e_f.shape[0]*e_f.shape[1]*e_f.shape[2]*e_f.shape[3])
        # spars_df = (d_f<=0).sum()/(d_f.shape[0]*d_f.shape[1]*d_f.shape[2]*d_f.shape[3])
        spars_ef = (torch.sum(e_f, 1)<=0).sum() / (e_f.shape[0]*e_f.shape[2]*e_f.shape[3])
        spars_df = (torch.sum(d_f, 1)<=0).sum() / (d_f.shape[0]*d_f.shape[2]*d_f.shape[3])
        total_e_spars.append(spars_ef)
        total_d_spars.append(spars_df)
      
        e_r, d_r = torch.max(e_f, dim=1).values, torch.max(d_f, dim=1).values
        total_sim.append(torch.cosine_similarity(e_r, d_r, 0).mean())
        e_r = cv2.putText(e_r[0].cpu().detach().numpy(), str(spars_ef.item()), (0, 10), 0, 0.07*level, (255,255,255), bottomLeftOrigin=False)
        d_r = cv2.putText(d_r[0].cpu().detach().numpy(), str(spars_df.item()), (0, 10), 0, 0.07*level, (255,255,255), bottomLeftOrigin=False)
        cv2.imwrite(f'{model_path}/samples/sparse/encode_level{level}.png', e_r*10)
        cv2.imwrite(f'{model_path}/samples/sparse/decode_level{level}.png', d_r*10)
        level += 1
    
    return {'sim':total_sim , 'spars_e': total_e_spars, 'spars_d': total_d_spars}

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                # window.to(device)
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


def psnr(labels, outputs, max_val=1.):
    labels = labels.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    img_diff = outputs - labels
    rmse = math.sqrt(np.mean((img_diff)**2))
    if rmse == 0: # label과 output이 완전히 일치하는 경우
        return 100
    else:
        psnr = 20 * math.log10(max_val/rmse)
        return psnr

def _take_channels(*xs, ignore_channels=None): # xs[0] : outputs xs[1] : labels 
    if ignore_channels is None:
        return xs
    else:
        # ignore_channels를 제외하고 인덱스 다시 정리 
        channels = [channel for channel in range(xs[0].shape[1]) if channel != ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs

def get_confusion_matrix(pr, gt, num_classes, ignore_index):

    mask = (gt != ignore_index)
    pred_label = pr[mask]
    label = gt[mask]

    n = num_classes
    inds = n * label + pred_label

    mat = torch.bincount(inds, minlength=n**2).reshape(num_classes, num_classes)
    return mat

def IntersectionOverUnion(pr, gt, ignore_index=0, num_class = 19):
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_index)
    num_img = len(pr)
    total_matix = torch.zeros((num_class, num_class), dtype=torch.float).cuda()

    for i in range(num_img):
        pred = torch.argmax(pr[i], dim=0, keepdim=True)
        true = torch.argmax(gt[i], dim=0, keepdim=True)

        mat = get_confusion_matrix(pred, true, num_classes=num_class, ignore_index=ignore_index)     
        total_matix += mat.cuda()
    iou = torch.diagonal(total_matix) / (total_matix.sum(axis=1) + total_matix.sum(axis=0) - torch.diagonal(total_matix))
    return torch.nansum(iou)/num_img

# ----------------------------------------------------------------------------------------------------------------------------
def comput_confusion_matrix(pred, true, num_cls=19): # output, target 
    true = rearrange(true, 'b c h -> b h c')
    pred = rearrange(pred, 'b c h -> b h c')
    conf_mat = torch.zeros((true.shape[0], num_cls, num_cls)) # 클래스 19개의 confusion matrix shape 초기 설정, 배치의 각 샘플마다 confusion matrix를 하나씩 쓰도록 함

    for i in range(true.shape[0]): # 각 배치마다 돌면서         
        y_args = torch.argmax(torch.cuda.FloatTensor(true[i][true[i].sum(axis=-1) > 0]), axis=-1)
        y_hat_args = torch.argmax(torch.cuda.FloatTensor(pred[i][true[i].sum(axis=-1) > 0]), axis=-1) # boolean indexing을 이용해서 정답이 존재하는 포인트만 남긴 후 argmax를 통해 예측 값을 얻음
        
        inds = num_cls * y_args + y_hat_args
        conf_mat[i] = torch.bincount(inds, minlength=num_cls**2).reshape(num_cls, num_cls)  
        
    return conf_mat

def iou(pred, target, num_cls, ignore_class=None):
    if ignore_class == 0:  
        pred = pred[:,1:,:]
        target = target[:,1:,:]
    elif ignore_class == 19:
        pred = pred[:,:-1,:]
        target = target[:,:-1,:]
    
    target = rearrange(target, 'b c h w -> b c (h w)')
    pred = rearrange(pred, 'b c h w -> b c (h w)')

    conf_mat = comput_confusion_matrix(pred, target, num_cls=num_cls)
    miou = torch.zeros((conf_mat.shape[0]))
    for i in range(conf_mat.shape[0]):
        sum_over_row = torch.sum(conf_mat[i], axis=0)
        sum_over_col = torch.sum(conf_mat[i], axis=1)
        true_positives = conf_mat[i].diagonal()
        denominator = sum_over_row + sum_over_col - true_positives
        denominator = torch.where(denominator == 0, torch.tensor(float('nan')), denominator) 
        iou_each_class = true_positives / denominator
        miou[i] = torch.nansum(iou_each_class) / torch.sum(iou_each_class >0 )
    return torch.mean(miou)


def pixel_acc(pred, target, ignore_class=None):
    if ignore_class == 0:  
        pred = pred[:,1:,:]
        target = target[:,1:,:]
    elif ignore_class == 19:
        pred = pred[:,:-1,:]
        target = target[:,:-1,:]
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total

def pixel_acc2(pred, target, ignore_class=None):
    if ignore_class == 0:  
        pred = pred[:,1:,:]
        target = target[:,1:,:]
    elif ignore_class == 19:
        pred = pred[:,:-1,:]
        target = target[:,:-1,:]

    target = rearrange(target, 'b c h w -> b (h w) c')
    pred = rearrange(pred, 'b c h w -> b (h w) c ')

    result = torch.zeros((target.shape[0],))  
    
    for i in range(target.shape[0]): 
        y_args = torch.argmax(torch.cuda.FloatTensor(target[i][target[i].sum(axis=-1)>0]), axis=-1)
        y_hat_args = torch.argmax(torch.cuda.FloatTensor(pred[i][target[i].sum(axis=-1)>0]), axis=-1)

        correct = (y_hat_args[i]==y_args[i]).sum()
        total = (y_args[i]==y_args[i]).sum()
        result[i] = correct / total
    return torch.mean(result)


class iouEval:
  def __init__(self, n_classes, device=torch.device("cpu"), ignore=None):
    # classes
    self.n_classes = n_classes

    # What to include and ignore from the means
    self.ignore = torch.tensor(ignore).long()
    self.include = torch.tensor(
        [n for n in range(self.n_classes) if n not in self.ignore]).long()
    print("[IOU EVAL] IGNORE: ", self.ignore)
    print("[IOU EVAL] INCLUDE: ", self.include)

    # get device
    self.device = device
    # reset the class counters
    self.reset()

  def num_classes(self):
    return self.n_classes

  def reset(self):
    self.conf_matrix = torch.zeros(
        (self.n_classes, self.n_classes), device=self.device).long()

  def addBatch(self, x, y):  # x=preds, y=targets
    # to tensor
    # x_row = torch.from_numpy(x).to(self.device).long()
    # y_row = torch.from_numpy(y).to(self.device).long()

    # sizes should be matching
    x_row = x.reshape(-1)  # de-batchify
    y_row = y.reshape(-1)  # de-batchify

    # check
    assert(x_row.shape == y_row.shape)

    # idxs are labels and predictions
    idxs = torch.stack([x_row, y_row], dim=0).long()

    # ones is what I want to add to conf when I
    ones = torch.ones((idxs.shape[-1]), device=self.device).long()

    # make confusion matrix (cols = gt, rows = pred)
    self.conf_matrix = self.conf_matrix.index_put_(
        tuple(idxs), ones, accumulate=True)

  def getStats(self):
    # remove fp from confusion on the ignore classes cols
    conf = self.conf_matrix.clone().double()
    conf[:, self.ignore] = 0

    # get the clean stats
    tp = conf.diag()
    fp = conf.sum(dim=1) - tp
    fn = conf.sum(dim=0) - tp
    return tp, fp, fn

  def getIoU(self):
    tp, fp, fn = self.getStats()
    intersection = tp
    union = tp + fp + fn + 1e-15
    iou = intersection / union
    iou_mean = (intersection[self.include] / union[self.include]).mean()
    return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

  def getacc(self):
    tp, fp, fn = self.getStats()
    total_tp = tp.sum()
    total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
    acc_mean = total_tp / total
    return acc_mean  # returns "acc mean
  

def iou(pred, target, ignore_class = 0):
    pred = F.softmax(pred, dim=1)
    if ignore_class == 0:
        pred = pred[:,1:,:]# 0번 클래스 제거 -> n_class는 19가 됨
        target = target[:,1:,:]
    
    target = rearrange(target, 'b c h w -> b c (h w)')
    pred = rearrange(pred, 'b c h w -> b c (h w)')

    conf_mat = comput_confusion_matrix(pred, target, num_cls = 19)
    miou = torch.zeros((conf_mat.shape[0]))#샘플 개수만큼의 배열을 만들고 b,c ,c 
    for i in range(conf_mat.shape[0]):
        sum_over_row = torch.sum(conf_mat[i], axis=0)
        sum_over_col = torch.sum(conf_mat[i], axis=1)
        true_positives = conf_mat[i].diagonal()
        denominator = sum_over_row + sum_over_col - true_positives
        denominator = torch.where(denominator == 0, torch.nan, denominator)#???
        iou_each_class = true_positives / denominator
        miou[i] = torch.nansum(iou_each_class) / torch.sum(iou_each_class > 0)
    return torch.nanmean(miou)

def pixel_acc(pred, target, ignore_class = 0):
    pred = F.softmax(pred, dim=1)
    if ignore_class == 0:  
        pred = pred[:,1:,:]# 0번 클래스 제거 -> n_class는 19가 됨
        target = target[:,1:,:]

    target = rearrange(target, 'b c h w -> b (h w) c')
    pred = rearrange(pred, 'b c h w -> b (h w) c ')

    result = torch.zeros((target.shape[0],))  
    
    for i in range(target.shape[0]):
        y_args = torch.argmax(torch.Tensor(target[i][target[i].sum(axis=-1) > 0]), axis=-1)
        y_hat_args = torch.argmax(torch.Tensor(pred[i][target[i].sum(axis=-1) >0]), axis=-1)

        correct = (y_hat_args[i] == y_args[i]).sum()
        total   = (y_args[i] == y_args[i]).sum()
        result[i] = correct / total
    return torch.nanmean(result)

def uv_iou(pred, target, ignore_class = 0):
    # pred = F.softmax(pred, dim=1)

    if ignore_class == 0:
        pred = pred[:,1:,:]# 0번 클래스 제거 -> n_class는 19가 됨
        target = target[:,1:,:]#이미 'b c (h w)'로 들어옴 
    
    conf_mat = comput_confusion_matrix(pred,target, num_cls=19)
    miou = torch.zeros((conf_mat.shape[0]))
    class_iou = torch.zeros((conf_mat.shape[0],conf_mat.shape[1]))
    for i in range(conf_mat.shape[0]):
        sum_over_row = torch.sum(conf_mat[i], axis=0)
        sum_over_col = torch.sum(conf_mat[i], axis=1)
        true_positives = conf_mat[i].diagonal()
        denominator = sum_over_row + sum_over_col - true_positives
        denominator = torch.where(denominator == 0, torch.nan, denominator)#???
        iou_each_class = true_positives / denominator
        miou[i] = torch.nansum(iou_each_class) / torch.sum(iou_each_class > 0)
        class_iou[i] = iou_each_class
    class_ious = torch.nansum(class_iou > 0, 0) / len(class_iou)
    # print(class_iou)
    return torch.nanmean(miou), class_ious

class IOUEval:
    def __init__(self, n_classes, device=torch.device("cpu"), ignore=None, is_distributed=False):
        self.n_classes = n_classes
        self.device = device

        # if ignore is larger than n_classes, consider no ignoreIndex
        self.ignore = torch.tensor(ignore).long()
        self.include = torch.tensor(
            [n for n in range(self.n_classes) if n not in self.ignore]).long()
        print("[IOU EVAL] IGNORE: ", self.ignore)
        print("[IOU EVAL] INCLUDE: ", self.include)
        self.is_distributed = is_distributed
        self.reset()

    def num_classes(self):
        return self.n_classes

    def reset(self):
        self.conf_matrix = torch.zeros((self.n_classes, self.n_classes), device=self.device).long()
        self.ones = None
        self.last_scan_size = None  # for when variable scan size is used

    def addBatch(self, x, y):  # x=preds, y=targets
        # if numpy, pass to pytorch
        # to tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.array(x)).long().to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(np.array(y)).long().to(self.device)

        # sizes should be "batch_size x H x W"
        x_row = x.reshape(-1)  # de-batchify
        y_row = y.reshape(-1)  # de-batchify

        # idxs are labels and predictions
        idxs = torch.stack([x_row, y_row], dim=0).long()

        # ones is what I want to add to conf when I
        if self.ones is None or self.last_scan_size != idxs.shape[-1]:
            self.ones = torch.ones((idxs.shape[-1]), device=self.device).long()
            self.last_scan_size = idxs.shape[-1]

        # make confusion matrix (cols = gt, rows = pred)
        self.conf_matrix = self.conf_matrix.index_put_(tuple(idxs), self.ones, accumulate=True)

        # print(self.tp.shape)
        # print(self.fp.shape)
        # print(self.fn.shape)

    def getStats(self):
        # remove fp and fn from confusion on the ignore classes cols and rows
        conf = self.conf_matrix.clone().double()
        if self.is_distributed:
            conf = conf.cuda()
            torch.distributed.barrier()
            torch.distributed.all_reduce(conf)
            conf = conf.to(self.conf_matrix)
        conf[self.ignore] = 0
        conf[:, self.ignore] = 0
        # for i in self.ignore_add:
        #     conf[i] = 0
        #     conf[:, i] = 0

        # get the clean stats
        tp = conf.diag()
        fp = conf.sum(dim=1) - tp
        fn = conf.sum(dim=0) - tp
        return tp, fp, fn

    def getIoU(self):
        self.include = torch.tensor([n for n in range(self.n_classes) if n not in self.ignore]).long()
        tp, fp, fn = self.getStats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        # self.include = [i.item() for i in self.include if i not in self.ignore_add]
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

    def getacc(self):
        print("getacc() will be deprecated, please use getAcc INSTEAD")
        tp, fp, fn = self.getStats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
        acc_mean = total_tp / total
        return acc_mean  # returns "acc mean"

    def getAcc(self):
        tp, fp, fn = self.getStats()
        total = tp+ fp + 1e-15
        acc = tp / total 
        acc_mean = acc[self.include].mean()
        return acc_mean, acc
    
    def getRecall(self):
        tp, fp, fn = self.getStats()
        total = tp + fn + 1e-15
        recall = tp / total
        recall_mean = recall[self.include].mean()
        return recall_mean, recall
    
# if __name__ == "__main__":
#     pred = torch.rand(4,20,327680)
#     true = torch.rand(4,20,327680)
#     # true = torch.randint(2, size=(4,20,256,1280))
#     miou = uv_iou(pred, true)
#     print(miou)
