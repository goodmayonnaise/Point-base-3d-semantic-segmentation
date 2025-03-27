

import torch
from torch import nn
import torch.nn.functional as F
# from model.modules.attention_modules import *

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters): #5 3
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):

        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output
    
class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1,resA2,resA3),dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA


        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB

class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True, first=False):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.first = first
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.conv0 = nn.Conv2d(in_filters, in_filters//4, 1, 1)

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        if self.first:
            self.conv1 = nn.Conv2d(in_filters + out_filters, out_filters, (3,3), padding=1)
        else:
            # self.conv1 = nn.Conv2d(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=1)
            self.conv1 = nn.Conv2d(in_filters//4 + out_filters, out_filters, (3,3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1,1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip): # 768 32 8
        # if self.first :
        #     upA = self.conv0(x)
        if not self.first:
            upA = nn.PixelShuffle(2)(x) # size 1/4 192 64 16 
        if self.drop_out:
            if not self.first:
                upA = self.dropout1(upA)
                upB = torch.cat((upA,skip),dim=1) # 768/4 + 384
        if self.first:
            upB = torch.cat((x, skip), dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB) # upB dim 784
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1,upE2,upE3),dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)

        if self.drop_out:
            upE = self.dropout3(upE)

        return upE

class UpBlock2(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlock2, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_filters//4, out_filters, (3,3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1,1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        upA = nn.PixelShuffle(2)(x) # size 1/4 192 64 16 
        if self.drop_out:
            upA = self.dropout1(upA)

        upE = self.conv1(upA) # upB dim 784
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1,upE2,upE3),dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE

class SalsaNextEncoder(nn.Module): # orginal code
    def __init__(self, nclasses):
        super(SalsaNextEncoder, self).__init__()
        self.nclasses = nclasses

        self.downCntx = ResContextBlock(3, 32)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        # self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

class SalsaNextDecoder(nn.Module):
    def __init__(self, nclasses, embed_dim):
        super(SalsaNextDecoder, self).__init__()
        # self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2) # 256, 128
        # self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2) # 128, 128
        # self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2) # 128, 64
        # self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False) # 64, 32 
        self.upBlock0 = UpBlock(embed_dim, embed_dim, 0.2, first=True)
        self.upBlock1 = UpBlock(embed_dim, embed_dim//2, 0.2)
        self.upBlock2 = UpBlock(embed_dim//2, embed_dim//4, 0.2)
        self.upBlock3 = UpBlock(embed_dim//4, embed_dim//8, 0.2, drop_out=False)
        self.upBlock4 = UpBlock2(embed_dim//4, embed_dim//8, 0.2, drop_out=False) # 64, 32 

        self.logits = nn.Conv2d(embed_dim//8, nclasses, kernel_size=(1, 1))
        
    def forward(self, down5c, down3b, down2b, down1b, down0b):
        up4e = self.upBlock1(down5c,down3b) # 64 16
        up3e = self.upBlock2(up4e, down2b) # 128 32
        up2e = self.upBlock3(up3e, down1b) # 256 64
        up1e = self.upBlock4(up2e, down0b) # 512 128
        logits = self.logits(up1e) # 20 1024 256

        logits = logits
        logits = F.softmax(logits, dim=1) 
        return logits

class SalsaNext(nn.Module):
    def __init__(self, nclasses):
        super(SalsaNext, self).__init__()
        self.nclasses = nclasses

        self.downCntx = ResContextBlock(3, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        # self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False)

        self.logits = nn.Conv2d(32, nclasses, kernel_size=(1, 1))

    def forward(self, x):
        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)

        up4e = self.upBlock1(down5c,down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)
        logits = self.logits(up1e)

        logits = logits
        logits = F.softmax(logits, dim=1)
        return logits

class SalasNextAttenEncoder(nn.Module):
    def __init__(self, embeding_dim, kernel_size=(3, 3)):
        super(SalasNextAttenEncoder, self).__init__()
        self.embeding_dim = embeding_dim
        self.drop_rate = 0.2
        self.rescontext1 = ResContextBlock(3, self.embeding_dim[0])
        self.rescontext2 = ResContextBlock(self.embeding_dim[0], self.embeding_dim[0])
        self.rescontext3 = ResContextBlock(self.embeding_dim[0], self.embeding_dim[0])

        self.resBlock1 = ResBlock(self.embeding_dim[0], self.embeding_dim[1], dropout_rate=self.drop_rate, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(self.embeding_dim[1], self.embeding_dim[2], dropout_rate=self.drop_rate, pooling=True, drop_out=True)
        self.resBlock3 = ResBlock(self.embeding_dim[2], self.embeding_dim[3], dropout_rate=self.drop_rate, pooling=True, drop_out=True)
        self.resBlock4 = ResBlock(self.embeding_dim[3], self.embeding_dim[3], dropout_rate=self.drop_rate, pooling=True, drop_out=True)
        self.resBlock5 = ResBlock(self.embeding_dim[3], self.embeding_dim[3], dropout_rate=self.drop_rate, pooling=False, drop_out=True)

        self.spatten = SpatialAttention()
        self.act = nn.LeakyReLU()

        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)

    def sa(self, x, att = True):
        x_att = self.spatten(x)
        x = x + (x * x_att)
        x = self.act(x)
        
        if att:
            return x, x_att
        else:
            return x
    
    def forward(self, x):
        x = self.rescontext1(x)
        x = self.rescontext2(x)
        x = self.rescontext3(x) # b 32 256 1280

        down0c, down0b = self.resBlock1(x) # b 64 128 640
        down0c, down0c_att = self.sa(down0c, att=True)
        
        down1c, down1b = self.resBlock2(down0c) # b 128 64 320
        down1c, down1c_att = self.sa(down1c, att=True)

        down2c, down2b = self.resBlock3(down1c) # b 256 32 160
        down2c, down2c_att = self.sa(down2c, att=True)

        down3c, down3b = self.resBlock4(down2c) # 256 16 80
        down3c, down3c_att = self.sa(down3c, att=True) 

        down5c = self.resBlock5(down3c) # b 256 16 80
        down5c, down5c_att = self.sa(down5c, att=True)

        f_dict = {'x_0' : down0b, 'x_1' : down1b, 'x_2' : down2b, 'x_3' : down3b, 'x_4' : down5c}
        att_dict = {'x_0' : down0c_att, 'x_1' : down1c_att, 'x_2' : down2c_att, 'x_3' : down3c_att, 'x_4' : down5c_att}
        unpool_f_dict = {'x_0' : down0c, 'x_1' : down1c, 'x_2' : down2c, 'x_3' : down3c, 'x_4' : down5c}
        
        return f_dict, att_dict, unpool_f_dict

class SalasNextAttenDecoder(nn.Module):
    def __init__(self, embeding_dim):
        super(SalasNextAttenDecoder, self).__init__()
        self.embeding_dim = embeding_dim
        self.drop = 0.2
        self.upBlock1 = UpBlock(self.embeding_dim[3], self.embeding_dim[2], self.drop)
        self.upBlock2 = UpBlock(self.embeding_dim[2], self.embeding_dim[2], self.drop)
        self.upBlock3 = UpBlock(self.embeding_dim[2], self.embeding_dim[1], self.drop)
        self.upBLock4 = UpBlock(self.embeding_dim[1], self.embeding_dim[0], self.drop, drop_out=False)
        self.act = nn.LeakyReLU()
        self.spatten = SpatialAttention()
        
    def sa(self, x, att = True):
        x_att = self.spatten(x)
        x = x + (x * x_att)
        x = self.act(x)
        if att:
            return x, x_att
        else:
            return x    

    def forward(self, f_dict):
        up4e = self.upBlock1(f_dict['x_4'], f_dict['x_3']) # b 128 32 160
        up4e, up4e_att = self.sa(up4e, att=True)
        
        up3e = self.upBlock2(up4e, f_dict['x_2'])
        up3e, up3e_att = self.sa(up3e, att=True)
        
        up2e = self.upBlock3(up3e, f_dict['x_1'])
        up2e, up2e_att = self.sa(up2e, att=True)
        
        up1e = self.upBLock4(up2e, f_dict['x_0'])
        up1e, up1e_att = self.sa(up1e, att=True)
        
        att_dict = {'up4' : up4e_att, 'up3' : up3e_att, 'up2' : up2e_att, 'up1' : up1e_att}
        f_dict = {'up4' : up4e, 'up3' : up3e, 'up2' : up2e, 'up1' : up1e}
        
        return up1e, f_dict, att_dict

