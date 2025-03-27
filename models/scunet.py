
import torch
import torch.nn.functional as F
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate=0.2, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        in_filters, out_filters = int(in_filters), int(out_filters)
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

        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
        return resB

class ResBlock1122(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate=0.2, kernel_size=(3, 3), stride=1, drop_out=True):
        super(ResBlock1122, self).__init__()
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

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

        concat = torch.cat((resA1, resA2 ,resA3),dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA

        if self.drop_out:
            resA = self.dropout(resA)
        return resA

class ResBlock1204(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate=0.2, kernel_size=(3, 3), stride=1, drop_out=True):
        super(ResBlock1122, self).__init__()
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

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

        concat = torch.cat((resA1, resA2 ,resA3),dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA

        if self.drop_out:
            resA = self.dropout(resA)
        return resA



class SC_UNET_Encoder(nn.Module):
    def __init__(self, dim=48, mode=None):
        super(SC_UNET_Encoder, self).__init__()
        self.mode = mode
        self.conv_block1 = self.conv_block(3, dim)
        self.conv_block2 = self.conv_block(dim, dim*2)
        self.conv_block3 = self.conv_block(dim*2, dim*4)
        self.conv_block4 = self.conv_block(dim*4, dim*8)
        self.conv_block5 = self.conv_block(dim*8, dim*16)
        if self.mode == 'feature6':
            self.conv_block6 = self.conv_block(dim*16, dim*32)

        self.maxpool = nn.MaxPool2d(2)

    def conv_block(self, in_channel, out_channels):
        layer = nn.Sequential(nn.Conv2d(in_channel, out_channels, 3, 1, padding=1),
                              nn.ELU(),
                              nn.BatchNorm2d(out_channels),
                              nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
                              nn.ELU(),
                              nn.BatchNorm2d(out_channels))
        return layer

    def forward(self, x):
        f1 = self.conv_block1(x) # 48 256 1024

        f2 = self.maxpool(f1) # 48 56 296
        f2 = self.conv_block2(f2) # 96 128 512

        f3 = self.maxpool(f2)
        f3= self.conv_block3(f3) # 192 64 256

        f4 = self.maxpool(f3)
        f4 = self.conv_block4(f4) # 384 32 128

        f5 = self.maxpool(f4)
        f5 = self.conv_block5(f5) # 768 16 64
        
        if self.mode == 'feature6':
            f6 = self.maxpool(f5)
            f6 = self.conv_block6(f6)
            return f6, f5, f4, f3, f2, f1

        return f5, f4, f3, f2, f1 
    
class Encoder(nn.Module):
    def __init__(self, dim=48):
        super(Encoder, self).__init__()
        self.conv_block1 = self.conv_block(3, dim, first=True)
        self.conv_block2 = self.conv_block(dim, dim*2)
        self.conv_block3 = self.conv_block(dim*2, dim*4)
        self.conv_block4 = self.conv_block(dim*4, dim*8)
        self.conv_block5 = self.conv_block(dim*8, dim*16)
        self.conv_block6 = self.conv_block(dim*16, dim*32)

        self.maxpool = nn.MaxPool2d(2)

    def conv_block(self, in_channels, out_channels, first=False):
        if first:
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                                  nn.ELU(),
                                  nn.BatchNorm2d(out_channels))
        else:
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                                  nn.ELU(),
                                  nn.BatchNorm2d(out_channels),
                                  nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
                                  nn.ELU(),
                                  nn.BatchNorm2d(out_channels))
        return layer

    def forward(self, x):
        f1 = self.conv_block1(x) # 48 256 1024

        f2 = self.maxpool(f1) # 48 56 296
        f2 = self.conv_block2(f2) # 96 128 512

        f3 = self.maxpool(f2)
        f3= self.conv_block3(f3) # 192 64 256

        f4 = self.maxpool(f3)
        f4 = self.conv_block4(f4) # 384 32 128

        f5 = self.maxpool(f4)
        f5 = self.conv_block5(f5) # 768 16 64
        
        f6 = self.maxpool(f5)
        f6 = self.conv_block6(f6)
        return f6, f5, f4, f3, f2, f1
        # original scunet + not extract f6 
        # return f5, f4, f3, f2, f1 

class StridedEncoder(nn.Module):
    def __init__(self, dim=48):
        super(StridedEncoder, self).__init__()
        self.conv_block1 = self.conv_block(3, dim, first=True)
        self.conv_block2 = self.conv_block(dim, dim*2)
        self.conv_block3 = self.conv_block(dim*2, dim*4)
        self.conv_block4 = self.conv_block(dim*4, dim*8)
        self.conv_block5 = self.conv_block(dim*8, dim*16)
        self.conv_block6 = self.conv_block(dim*16, dim*32, last=True)

        self.maxpool = nn.MaxPool2d(2)

    def conv_block(self, in_channels, out_channels, first=False, last=False):
        if first :
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                                  nn.ELU())
        elif last :
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                                  nn.ELU(),
                                  nn.BatchNorm2d(out_channels),
                                  nn.Conv2d(out_channels, out_channels, 3, 2, padding=1),
                                  nn.ELU())
        else:
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                                  nn.ELU(),
                                  nn.BatchNorm2d(out_channels),
                                  nn.Conv2d(out_channels, out_channels, 3, 2, padding=1),
                                  nn.ELU(),
                                  nn.BatchNorm2d(out_channels))

        return layer
            
    def forward(self, x):
        f1 = self.conv_block1(x) 

        f2 = self.conv_block2(f1)

        f3 = self.conv_block3(f2)

        f4 = self.conv_block4(f3)

        f5 = self.conv_block5(f4)
        
        f6 = self.conv_block6(f5)

        return f6, f5, f4, f3, f2, f1
    
class FixedStridedEncoder(nn.Module):
    def __init__(self, dim=48):
        super(FixedStridedEncoder, self).__init__()
        self.conv_block1 = self.conv_block(3, dim, first=True)
        self.conv_block2 = self.conv_block(dim, dim*2)
        self.conv_block3 = self.conv_block(dim*2, dim*4)
        self.conv_block4 = self.conv_block(dim*4, dim*8)
        self.conv_block5 = self.conv_block(dim*8, dim*16)
        self.conv_block6 = self.conv_block(dim*16, dim*32, last=True)

        self.maxpool = nn.MaxPool2d(2)

    def conv_block(self, in_channels, out_channels, first=False, last=False):
        if first :
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                                  nn.ELU())
        elif last :
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                                  nn.ELU(),
                                  nn.BatchNorm2d(out_channels),
                                  nn.Conv2d(out_channels, out_channels, 3, 2, padding=1),
                                  nn.ELU())
        else:
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                                  nn.ELU(),
                                  nn.BatchNorm2d(out_channels),
                                  nn.Conv2d(out_channels, out_channels, 3, 2, padding=1),
                                  nn.ELU(),
                                  nn.BatchNorm2d(out_channels))

        return layer
            
    def forward(self, x):
        f1 = self.conv_block1(x) 

        f2 = self.conv_block2(f1)

        f3 = self.conv_block3(f2)

        f4 = self.conv_block4(f3)

        f5 = self.conv_block5(f4)
        
        f6 = self.conv_block6(f5)

        return f6, f5, f4, f3
    
class FixedStridedEncoder_cntx(nn.Module):
    def __init__(self, dim=48):
        super(FixedStridedEncoder_cntx, self).__init__()
        from model.salsanext import ResContextBlock
        self.downCntx = ResContextBlock(3, 16)
        # self.downCntx2 = ResContextBlock(3, 32)
        # self.downCntx3 = ResContextBlock(32, 32)
        self.conv_block1_1 = self.conv_block(16, dim, first=True)
        self.conv_block2 = self.conv_block(dim, dim*2)
        self.conv_block3 = self.conv_block(dim*2, dim*4)
        self.conv_block4 = self.conv_block(dim*4, dim*8)
        self.conv_block5 = self.conv_block(dim*8, dim*16)
        self.conv_block6 = self.conv_block(dim*16, dim*32, last=True)

        self.maxpool = nn.MaxPool2d(2)

    def conv_block(self, in_channels, out_channels, first=False, last=False):
        if first :
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                                  nn.ELU())
        elif last :
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                                  nn.ELU(),
                                  nn.BatchNorm2d(out_channels),
                                  nn.Conv2d(out_channels, out_channels, 3, 2, padding=1),
                                  nn.ELU())
        else:
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                                  nn.ELU(),
                                  nn.BatchNorm2d(out_channels),
                                  nn.Conv2d(out_channels, out_channels, 3, 2, padding=1),
                                  nn.ELU(),
                                  nn.BatchNorm2d(out_channels))

        return layer
            
    def forward(self, x):
        cntx = self.downCntx(x)
        # cntx = self.downCntx2(cntx)
        # cntx = self.downCntx3(cntx)

        f1 = self.conv_block1_1(cntx) 

        f2 = self.conv_block2(f1)

        f3 = self.conv_block3(f2)

        f4 = self.conv_block4(f3)

        f5 = self.conv_block5(f4)
        
        f6 = self.conv_block6(f5)

        return f6, f5, f4, f3

class FixedStridedEncoder_feature5(nn.Module):
    def __init__(self, dim=48):
        super(FixedStridedEncoder_feature5, self).__init__()
        self.conv_block1 = self.conv_block(3, dim, first=True)
        self.conv_block2 = self.conv_block(dim, dim*2)
        self.conv_block3 = self.conv_block(dim*2, dim*4)
        self.conv_block4 = self.conv_block(dim*4, dim*8)
        self.conv_block5 = self.conv_block(dim*8, dim*16, last=True)

        self.maxpool = nn.MaxPool2d(2)

    def conv_block(self, in_channels, out_channels, first=False, last=False):
        if first :
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                                  nn.ELU())
        elif last :
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                                  nn.ELU(),
                                  nn.BatchNorm2d(out_channels),
                                  nn.Conv2d(out_channels, out_channels, 3, 2, padding=1),
                                  nn.ELU())
        else:
            layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                                  nn.ELU(),
                                  nn.BatchNorm2d(out_channels),
                                  nn.Conv2d(out_channels, out_channels, 3, 2, padding=1),
                                  nn.ELU(),
                                  nn.BatchNorm2d(out_channels))

        return layer
            
    def forward(self, x):
        f1 = self.conv_block1(x) 

        f2 = self.conv_block2(f1)

        f3 = self.conv_block3(f2)

        f4 = self.conv_block4(f3)

        f5 = self.conv_block5(f4)

        return f5, f4, f3 



class SC_UNET_Decoder(nn.Module):
    def __init__(self, dim=768, mode=None, mode2=None): # dim 768*2 = 1536
        super(SC_UNET_Decoder, self).__init__()
        self.mode = mode
        self.mode2 = mode2

        if self.mode == 'feature6' :
            self.conv_block0 = self.conv_block(dim, dim, first=True)
            
            self.up0 = self.deconv(dim, dim/2)

            self.conv_block1 = self.conv_block(dim, dim/2)
            self.up1 = self.deconv(dim/2, dim/2/2)

            if self.mode2 == 'level1':
                self.conv_block2 = self.conv_block(dim/2/2, dim/2/2/2)
                self.up2 = self.deconv(dim/2/2/2, dim/2/2/2)
            else:
                self.conv_block2 = self.conv_block(dim/2, dim/2/2)
                self.up2 = self.deconv(dim/2/2, dim/2/2/2)

            if self.mode2 == "level123":
                self.conv_block3 = self.conv_block(dim/2/2, dim/2/2/2)
                self.up3 = self.deconv(dim/2/2/2, dim/2/2/2/2)

            else:
                self.conv_block3 = self.conv_block(dim/2/2/2, dim/2/2/2/2)
                self.up3 = self.deconv(dim/2/2/2/2, dim/2/2/2/2)

            self.conv_block4 = self.conv_block(dim/2/2/2/2, dim/2/2/2/2/2)
            self.up4 = self.deconv(dim/2/2/2/2/2, dim/2/2/2/2/2)
            
            self.conv_block5 = self.conv_block(dim/2/2/2/2/2, dim/2/2/2/2/2/2)
            self.up5 = self.deconv(dim/2/2/2/2/2/2, dim/2/2/2/2/2/2)

            self.conv = nn.Conv2d(int(dim/2/2/2/2/2/2), 3, 1, 1)

        else :
            self.conv_block1 = self.conv_block(dim, dim, first=True)
            
            self.up1 = self.deconv(dim, dim/2)

            self.conv_block2 = self.conv_block(dim, dim/2)
            self.up2 = self.deconv(dim/2, dim/2/2)

            self.conv_block3 = self.conv_block(dim/2, dim/2/2)
            self.up3 = self.deconv(dim/2/2, dim/2/2/2)

            if self.mode == 'level4' :
                self.conv_block4 = self.conv_block(dim/2/2, dim/2/2/2)
            else:
                self.conv_block4 = self.conv_block(dim/2/2/2, dim/2/2/2)
                
            self.up4 = self.deconv(dim/2/2/2, dim/2/2/2/2)
            
            if self.mode == 'level5':
                self.conv_block5 = self.conv_block(dim/2/2/2, dim/2/2/2/2)
            else:
                self.conv_block5 = self.conv_block(dim/2/2/2/2, dim/2/2/2/2)
            self.conv = nn.Conv2d(int(dim/2/2/2/2), 3, 1, 1)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def deconv(self, in_channels, out_channels):
        in_channels, out_channels = int(in_channels), int(out_channels)
        return nn.ConvTranspose2d(in_channels, out_channels, 4, 2, padding=1)

    def conv_block(self, in_channels, out_channels, first=False):
        in_channels, out_channels = int(in_channels), int(out_channels)
        if first :
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                nn.ELU(),
                nn.BatchNorm2d(out_channels))
        else:
            layer = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                                nn.ELU(),
                                nn.BatchNorm2d(out_channels),
                                nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
                                nn.ELU(),
                                nn.BatchNorm2d(out_channels))
        return layer

    def forward(self, features):

        if self.mode == 'feature6':
            f6, f5, f4 = features[0], features[1], features[2]
            d6 = self.conv_block0(f6)

            d5 = self.up0(d6)

            d4 = torch.cat((d5, f5), dim=1)
            d4 = self.conv_block1(d4)
            d4 = self.up1(d4)
            
            if self.mode2 in [None, 'level123']:
                d3 = torch.cat((d4, f4), dim=1)
                d3 = self.conv_block2(d3)
            elif self.mode2 == 'level1':
                d3 = self.conv_block2(d4)    
            d3 = self.up2(d3)

            if self.mode2 == 'level123':
                f3 = features[3]
                d2 = torch.cat((d3, f3), dim=1)
                d2 = self.conv_block3(d2)
            else: 
                d2 = self.conv_block3(d3)

            d2 = self.up3(d2) # 128 512 

            after_convd2 = self.conv_block4(d2) # after conv d2
            # res_d1 = torch.cat((after_convd2, d2), dim=1)
            # res_d1 = self.res_up4(res_d1)    
            d1 = self.up4(after_convd2) # 256 1024 

            # after_convd1 = self.conv_block5(res_d1)
            after_convd1 = self.conv_block5(d1)
            out = self.conv(after_convd1)
            out = self.sigmoid(out)

            return d6, d5, d4, d3, d2, d1, out

        elif self.mode == 'level5':

            f5, f4, f3, f1 = features[0], features[1], features[2], features[3]
            d5 = self.conv_block1(f5)       # 768 16 64
            
            d4 = self.up1(d5)               # 384 32 128

            d3 = torch.cat((d4, f4), dim=1)  # 768  32  128
            d3 = self.conv_block2(d3)         # 8n
            d3 = self.up2(d3)               # 
            
            d2 = torch.cat((d3, f3), dim=1) # 8n 
            d2 = self.conv_block3(d2)         # 4n 192 #
            d2 = self.up3(d2)                 # 2n 

            d1 = self.conv_block4(d2)         # 2n
            d1 = self.up4(d1)                 # n 
            
            out = torch.cat((d1, f1), dim=1)
            out = self.conv_block5(out)         # n 
            out = self.conv(out)
            out = self.sigmoid(out)
            # out = self.softmax(out)
            

        else: # origin 
            f5, f4, f3 = features[0], features[1], features[2]
            d5 = self.conv_block1(f5)       # 768 16 64
            
            d4 = self.up1(d5)               # 384 32 128

            d3 = torch.cat((d4, f4), dim=1)  # 768  32  128
            d3 = self.conv_block2(d3)         # 8n
            d3 = self.up2(d3)               # 
            
            d2 = torch.cat((d3, f3), dim=1) # 8n 
            d2 = self.conv_block3(d2)         # 4n 192 #
            d2 = self.up3(d2)                 # 2n 

            d1 = self.conv_block4(d2)         # 2n
            d1 = self.up4(d1)                 # n 

            out = self.conv_block5(d1)         # n 
            out = self.conv(out)
            out = self.sigmoid(out)
            # out = self.softmax(out)
            
        return d5, d4, d3, d2, d1, out 

class Decoder(nn.Module): #  mode='feature6', mode2='level123',
    def __init__(self, dim=768, relu=False): # dim 768*2 = 1536
        super(Decoder, self).__init__()
        self.conv_block0 = ResBlock1122(dim, dim, drop_out=False)
        self.up0 = nn.ConvTranspose2d(dim, dim//2, 4, 2, padding=1)

        self.conv_block1 = ResBlock1122(dim, dim//2)
        self.up1 = nn.ConvTranspose2d(dim//2, dim//4, 4, 2, padding=1)

        self.conv_block2 = ResBlock1122(dim//2, dim//4)
        self.up2 = nn.ConvTranspose2d(dim//4, dim//8, 4, 2, padding=1)

        self.conv_block3 = ResBlock1122(dim//4, dim//8)
        self.up3 = nn.ConvTranspose2d(dim//8, dim//16, 4, 2, padding=1)
        self.conv_out_half = nn.Conv2d(dim//16, 3, 1, 1)

        self.conv_block4 = ResBlock1122(dim//16, dim//32)
        self.up4 = nn.ConvTranspose2d(dim//32, dim//32, 4, 2, padding=1)
        
        self.conv_block5 = ResBlock1122(dim//32, dim//64)
        self.up5 = nn.ConvTranspose2d(dim//64, dim//64, 4, 2, padding=1)

        self.conv_out = nn.Conv2d(dim//64, 3, 1, 1)

        if relu:
            self.act = nn.ReLU()
        else:
            self.act = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                              nn.ELU(),
                              nn.BatchNorm2d(out_channels),
                              nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
                              nn.ELU(),
                              nn.BatchNorm2d(out_channels))
        return layer

    def forward(self, f6, f5, f4, f3):
        
        d6 = self.conv_block0(f6) # 1536 8 39
        d5 = self.up0(d6)         # 768 16 78 

        d4 = torch.cat((d5, f5), dim=1) # 1536 16 78
        d4 = self.conv_block1(d4) # 768 16 78
        d4 = self.up1(d4)         # 384 32 156

        d3 = torch.cat((d4, f4), dim=1) # 768 32 156
        d3 = self.conv_block2(d3) # 384 32 156
        d3 = self.up2(d3)         # 192 64 312
        
        d2 = torch.cat((d3, f3), dim=1)
        d2 = self.conv_block3(d2)
        d2 = self.up3(d2)

        out_half = self.conv_out_half(d2)
        out_half = self.act(out_half)
        
        after_convd2 = self.conv_block4(d2)
        
        d1 = self.up4(after_convd2)
        
        after_convd1 = self.conv_block5(d1)
        out = self.conv_out(after_convd1)
        out = self.act(out)
        
        return d6, d5, d4, d3, d2, d1, out, out_half

class Fixed_Decoder(nn.Module): #  mode='feature6', mode2='level123',
    def __init__(self, dim=768, relu=False): # dim 768*2 = 1536
        super(Fixed_Decoder, self).__init__()
        self.conv_block0 = ResBlock1122(dim, dim, drop_out=False)
        self.up0 = nn.ConvTranspose2d(dim, dim//2, 4, 2, padding=1)

        self.conv_block1 = ResBlock1122(dim, dim//2)
        self.up1 = nn.ConvTranspose2d(dim//2, dim//4, 4, 2, padding=1)

        self.conv_block2 = ResBlock1122(dim//2, dim//4)
        self.up2 = nn.ConvTranspose2d(dim//4, dim//8, 4, 2, padding=1)

        self.conv_block3 = ResBlock1122(dim//4, dim//8)
        self.up3 = nn.ConvTranspose2d(dim//8, dim//16, 4, 2, padding=1)

        self.conv_block4 = ResBlock1122(dim//16, dim//32)
        self.up4 = nn.ConvTranspose2d(dim//32, dim//32, 4, 2, padding=1)
        
        self.conv_block5 = ResBlock1122(dim//32, dim//64)
        self.up5 = nn.ConvTranspose2d(dim//64, dim//64, 4, 2, padding=1)

        self.conv_out = nn.Conv2d(dim//64, 3, 1, 1)

        if relu:
            self.act = nn.ReLU()
        else:
            self.act = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                              nn.ELU(),
                              nn.BatchNorm2d(out_channels),
                              nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
                              nn.ELU(),
                              nn.BatchNorm2d(out_channels))
        return layer

    def forward(self, f6, f5, f4, f3):
        
        d6 = self.conv_block0(f6) # 1536 8 39
        d5 = self.up0(d6)         # 768 16 78 

        d4 = torch.cat((d5, f5), dim=1) # 1536 16 78
        d4 = self.conv_block1(d4) # 768 16 78
        d4 = self.up1(d4)         # 384 32 156

        d3 = torch.cat((d4, f4), dim=1) # 768 32 156
        d3 = self.conv_block2(d3) # 384 32 156
        d3 = self.up2(d3)         # 192 64 312
        
        d2 = torch.cat((d3, f3), dim=1)
        d2 = self.conv_block3(d2)
        d2 = self.up3(d2)

        after_convd2 = self.conv_block4(d2)
        
        d1 = self.up4(after_convd2)
        
        after_convd1 = self.conv_block5(d1)
        out = self.conv_out(after_convd1)
        out = self.act(out)
        
        return out

class Fixed_Decoder_feature5(nn.Module): #  mode='feature6', mode2='level123',
    def __init__(self, dim=768, relu=False): # dim 768*2 = 1536
        super(Fixed_Decoder_feature5, self).__init__()
        self.conv_block0 = ResBlock1122(dim, dim, drop_out=False)
        self.up0 = nn.ConvTranspose2d(dim, dim//2, 4, 2, padding=1)

        self.conv_block1 = ResBlock1122(dim, dim//2)
        self.up1 = nn.ConvTranspose2d(dim//2, dim//4, 4, 2, padding=1)

        self.conv_block2 = ResBlock1122(dim//2, dim//4)
        self.up2 = nn.ConvTranspose2d(dim//4, dim//8, 4, 2, padding=1)

        self.conv_block3 = ResBlock1122(dim//8, dim//16)
        self.up3 = nn.ConvTranspose2d(dim//16, dim//32, 4, 2, padding=1)

        self.conv_rgb = nn.Conv2d(dim//32, 3, 1, 1)
        self.conv_seg = nn.Conv2d(dim//32, 20, 1, 1)

        if relu:
            self.act = nn.ReLU()
        else:
            self.act = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                              nn.ELU(),
                              nn.BatchNorm2d(out_channels),
                              nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
                              nn.ELU(),
                              nn.BatchNorm2d(out_channels))
        return layer

    def forward(self, f5, f4, f3):
        
        d5 = self.conv_block0(f5) # 1536 8 39
        d4 = self.up0(d5)         # 768 16 78 

        d3 = torch.cat((d4, f4), dim=1) # 1536 16 78
        d3 = self.conv_block1(d3) # 768 16 78
        d3 = self.up1(d3)         # 384 32 156

        d2 = torch.cat((d3, f3), dim=1) # 768 32 156
        d2 = self.conv_block2(d2) # 384 32 156
        d2 = self.up2(d2)         # 192 64 312
        
        d1 = self.conv_block3(d2)
        d1 = self.up3(d1)

        out_rgb = self.conv_rgb(d1)
        out_rgb = self.act(out_rgb)

        out_seg = self.conv_seg(d1)
        out_seg = F.softmax(out_seg, dim=1)
        
        return out_rgb, out_seg

class FixedDecoder_seg(nn.Module): #  mode='feature6', mode2='level123',
    def __init__(self, dim=768, relu=False): # dim 768*2 = 1536
        super(FixedDecoder_seg, self).__init__()
        self.conv_block0 = ResBlock1122(dim, dim, drop_out=False)
        self.up0 = nn.ConvTranspose2d(dim, dim//2, 4, 2, padding=1)

        self.conv_block1 = ResBlock1122(dim, dim//2)
        self.up1 = nn.ConvTranspose2d(dim//2, dim//4, 4, 2, padding=1)

        self.conv_block2 = ResBlock1122(dim//2, dim//4)
        self.up2 = nn.ConvTranspose2d(dim//4, dim//8, 4, 2, padding=1)

        self.conv_block3 = ResBlock1122(dim//4, dim//8)
        self.up3 = nn.ConvTranspose2d(dim//8, dim//16, 4, 2, padding=1)

        self.conv_block4 = ResBlock1122(dim//16, dim//32)
        self.up4 = nn.ConvTranspose2d(dim//32, dim//32, 4, 2, padding=1)
        
        self.conv_block5 = ResBlock1122(dim//32, dim//64)
        self.up5 = nn.ConvTranspose2d(dim//64, dim//64, 4, 2, padding=1)

        self.conv_out = nn.Conv2d(dim//64, 3, 1, 1)
        self.conv_seg = nn.Conv2d(dim//64, 20, 1, 1)

        if relu:
            self.act = nn.ReLU()
        else:
            self.act = nn.Sigmoid()

    def forward(self, f6, f5, f4, f3):
        
        d6 = self.conv_block0(f6) # 1536 8 39
        d5 = self.up0(d6)         # 768 16 78 

        d4 = torch.cat((d5, f5), dim=1) # 1536 16 78
        d4 = self.conv_block1(d4) # 768 16 78
        d4 = self.up1(d4)         # 384 32 156

        d3 = torch.cat((d4, f4), dim=1) # 768 32 156
        d3 = self.conv_block2(d3) # 384 32 156
        d3 = self.up2(d3)         # 192 64 312
        
        d2 = torch.cat((d3, f3), dim=1)
        d2 = self.conv_block3(d2)
        d2 = self.up3(d2)

        d2 = self.conv_block4(d2)
        
        d1 = self.up4(d2)
        d1 = self.conv_block5(d1)

        out = self.conv_out(d1)
        out = self.act(out)

        out_seg = self.conv_seg(d1)
        out_seg = F.softmax(out_seg, dim=1)
        
        return out, out_seg 

class FixedDecoder_seg_deform(nn.Module): #  mode='feature6', mode2='level123',
    def __init__(self, dim=768, relu=False): # dim 768*2 = 1536
        super(FixedDecoder_seg_deform, self).__init__()
        self.conv_block0 = ResBlock1122(dim, dim, drop_out=False)
        self.up0 = nn.ConvTranspose2d(dim, dim//2, 4, 2, padding=1)

        self.conv_block1 = ResBlock1122(dim, dim//2)
        self.up1 = nn.ConvTranspose2d(dim//2, dim//4, 4, 2, padding=1)

        self.conv_block2 = ResBlock1122(dim//2, dim//4)
        self.up2 = nn.ConvTranspose2d(dim//4, dim//8, 4, 2, padding=1)

        self.conv_block3 = ResBlock1122(dim//4, dim//8)
        self.up3 = nn.ConvTranspose2d(dim//8, dim//16, 4, 2, padding=1)

        self.conv_block4 = ResBlock1122(dim//16, dim//32)
        self.up4 = nn.ConvTranspose2d(dim//32, dim//32, 4, 2, padding=1)
        
        self.conv_block5 = ResBlock1122(dim//32, dim//64)
        self.up5 = nn.ConvTranspose2d(dim//64, dim//64, 4, 2, padding=1)

        from model.deform_conv import ConvOffset2D
        self.deform = ConvOffset2D(dim//64)

        self.conv_out = nn.Conv2d(dim//64, 3, 1, 1)
        self.conv_seg = nn.Conv2d(dim//64, 20, 1, 1)

        if relu:
            self.act = nn.ReLU()
        else:
            self.act = nn.Sigmoid()

    def forward(self, f6, f5, f4, f3):
        
        d6 = self.conv_block0(f6) # 1536 8 39
        d5 = self.up0(d6)         # 768 16 78 

        d4 = torch.cat((d5, f5), dim=1) # 1536 16 78
        d4 = self.conv_block1(d4) # 768 16 78
        d4 = self.up1(d4)         # 384 32 156

        d3 = torch.cat((d4, f4), dim=1) # 768 32 156
        d3 = self.conv_block2(d3) # 384 32 156
        d3 = self.up2(d3)         # 192 64 312
        
        d2 = torch.cat((d3, f3), dim=1)
        d2 = self.conv_block3(d2)
        d2 = self.up3(d2)

        d2 = self.conv_block4(d2)
        
        d1 = self.up4(d2)
        d1 = self.conv_block5(d1) # b 24 256 1248

        # out = self.conv_out(d1)
        out= self.deform(d1)
        out = self.conv_out(out)
        out = self.act(out)

        out_seg = self.conv_seg(d1)
        out_seg = F.softmax(out_seg, dim=1)
        
        return out, out_seg 

class FixedDecoder_seg_deform_branch(nn.Module): #  mode='feature6', mode2='level123',
    def __init__(self, dim=768, relu=False): # dim 768*2 = 1536
        super(FixedDecoder_seg_deform_branch, self).__init__()
        self.conv_block0 = ResBlock1122(dim, dim, drop_out=False)
        self.up0 = nn.ConvTranspose2d(dim, dim//2, 4, 2, padding=1)

        self.conv_block1 = ResBlock1122(dim, dim//2)
        self.up1 = nn.ConvTranspose2d(dim//2, dim//4, 4, 2, padding=1)

        self.conv_block2 = ResBlock1122(dim//2, dim//4)
        self.up2 = nn.ConvTranspose2d(dim//4, dim//8, 4, 2, padding=1)

        self.conv_block3 = ResBlock1122(dim//4, dim//8)
        self.up3 = nn.ConvTranspose2d(dim//8, dim//16, 4, 2, padding=1)

        self.conv_block4 = ResBlock1122(dim//16, dim//32)
        self.up4 = nn.ConvTranspose2d(dim//32, dim//32, 4, 2, padding=1)
        
        self.conv_block5 = ResBlock1122(dim//32, dim//64)
        self.up5 = nn.ConvTranspose2d(dim//64, dim//64, 4, 2, padding=1)

        from model.deform_conv import ConvOffset2D
        self.deform = ConvOffset2D(dim//64)

        self.conv_out = nn.Conv2d(dim//64, 3, 1, 1)
        self.conv_seg = nn.Conv2d(dim//64, 20, 1, 1)

        if relu:
            self.act = nn.ReLU()
        else:
            self.act = nn.Sigmoid()

    def forward(self, f6, f5, f4, f3):
        
        d6 = self.conv_block0(f6) # 1536 8 39
        d5 = self.up0(d6)         # 768 16 78 

        d4 = torch.cat((d5, f5), dim=1) # 1536 16 78
        d4 = self.conv_block1(d4) # 768 16 78
        d4 = self.up1(d4)         # 384 32 156

        d3 = torch.cat((d4, f4), dim=1) # 768 32 156
        d3 = self.conv_block2(d3) # 384 32 156
        d3 = self.up2(d3)         # 192 64 312
        
        d2 = torch.cat((d3, f3), dim=1)
        d2 = self.conv_block3(d2)
        d2 = self.up3(d2)

        d2 = self.conv_block4(d2)
        
        d1 = self.up4(d2)
        d1 = self.conv_block5(d1) # b 24 256 1248

        # out = self.conv_out(d1)
        out= self.deform(d1)
        out = self.conv_out(out)
        out = self.act(out)

        out_seg = self.conv_seg(d1)
        out_seg = F.softmax(out_seg, dim=1)
        
        return out, out_seg 

class PSDecoder(nn.Module): #  mode='feature6', mode2='level123',
    def __init__(self, dim=768, relu=False): # dim 768*2 = 1536
        super(PSDecoder, self).__init__()
        self.upblock1 = self.upblock(dim, dim//2)
        self.upblock2 = self.upblock(dim, dim//4)
        self.upblock3 = self.upblock(dim//2, dim//8)
        self.upblock4 = self.upblock(dim//4, dim//16)
        self.upblock5 = self.upblock(dim//16, dim//32)

        self.conv_half = nn.Conv2d(dim//16, 3, 1, 1)
        self.conv_org = nn.Conv2d(dim//32, 3, 1, 1)
        self.act = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                              nn.ELU(),
                              nn.BatchNorm2d(out_channels),
                              nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
                              nn.ELU(),
                              nn.BatchNorm2d(out_channels))
        return layer

    def upblock(self, indim, outdim):
        return nn.Sequential(nn.PixelShuffle(2), 
                             nn.Conv2d(indim//4, outdim, 3, 1, 1),
                             nn.ReLU()
                             )
    def upblock(self, indim, outdim):
        return nn.Sequential(nn.PixelShuffle(2), 
                             nn.Conv2d(indim//4, outdim, 3, 1, 1),
                             nn.ELU(), 
                             nn.BatchNorm2d(outdim),

                             nn.Conv2d(outdim, outdim, 3, 1, 1),
                             nn.ELU(), 
                             nn.BatchNorm2d(outdim)
                             )

    def forward(self, f6, f5, f4, f3):
        d5 = self.upblock1(f6)

        d4 = torch.cat([d5, f5], dim=1)
        d4 = self.upblock2(d4)

        d3 = torch.cat([d4, f4], dim=1)
        d3 = self.upblock3(d3)

        d2 = torch.cat([d3, f3], dim=1)
        d2 = self.upblock4(d2)
        out_half = self.conv_half(d2)
        out_half = self.act(out_half)

        d1 = self.upblock5(d2)
        out = self.conv_org(d1)
        out = self.act(out)

        return out, out_half

class PSDecoder_segout(nn.Module): #  mode='feature6', mode2='level123',
    def __init__(self, dim=768, relu=False): # dim 768*2 = 1536
        super(PSDecoder_segout, self).__init__()
        self.upblock1 = self.upblock(dim, dim//2)
        self.upblock2 = self.upblock(dim, dim//4)
        self.upblock3 = self.upblock(dim//2, dim//8)
        self.upblock4 = self.upblock(dim//4, dim//16)
        self.upblock5 = self.upblock(dim//16, dim//32)

        self.conv_half = nn.Conv2d(dim//16, 3, 1, 1)
        self.conv_org = nn.Conv2d(dim//32, 3, 1, 1)
        self.conv_seg = nn.Conv2d(dim//32, 20, 1, 1)
        self.act = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                              nn.ELU(),
                              nn.BatchNorm2d(out_channels),
                              nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
                              nn.ELU(),
                              nn.BatchNorm2d(out_channels))
        return layer

    def upblock(self, indim, outdim):
        return nn.Sequential(nn.PixelShuffle(2), 
                             nn.Conv2d(indim//4, outdim, 3, 1, 1),
                             nn.ReLU()
                             )
    def upblock(self, indim, outdim):
        return nn.Sequential(nn.PixelShuffle(2), 
                             nn.Conv2d(indim//4, outdim, 3, 1, 1),
                             nn.ELU(), 
                             nn.BatchNorm2d(outdim),

                             nn.Conv2d(outdim, outdim, 3, 1, 1),
                             nn.ELU(), 
                             nn.BatchNorm2d(outdim)
                             )

    def forward(self, f6, f5, f4, f3):
        d5 = self.upblock1(f6)

        d4 = torch.cat([d5, f5], dim=1)
        d4 = self.upblock2(d4)

        d3 = torch.cat([d4, f4], dim=1)
        d3 = self.upblock3(d3)

        d2 = torch.cat([d3, f3], dim=1)
        d2 = self.upblock4(d2)
        out_half = self.conv_half(d2)
        out_half = self.act(out_half)

        d1 = self.upblock5(d2)
        out = self.conv_org(d1)
        out = self.act(out)

        out_seg = self.conv_seg(d1)
        out_seg = F.softmax(out_seg, dim=1)

        return out, out_half, out_seg
        
class FixedPSDecoder_segout(nn.Module): #  mode='feature6', mode2='level123',
    def __init__(self, dim=768, relu=False): # dim 768*2 = 1536
        super(FixedPSDecoder_segout, self).__init__()
        self.upblock1 = self.upblock(dim, dim//2)
        self.upblock2 = self.upblock(dim, dim//4)
        self.upblock3 = self.upblock(dim//2, dim//8)
        self.upblock4 = self.upblock(dim//4, dim//16)
        self.upblock5 = self.upblock(dim//16, dim//32)

        self.conv_half = nn.Conv2d(dim//16, 3, 1, 1)
        self.conv_org = nn.Conv2d(dim//32, 3, 1, 1)
        self.conv_seg = nn.Conv2d(dim//32, 20, 1, 1)
        self.act = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                              nn.ELU(),
                              nn.BatchNorm2d(out_channels),
                              nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
                              nn.ELU(),
                              nn.BatchNorm2d(out_channels))
        return layer

    def upblock(self, indim, outdim):
        return nn.Sequential(nn.PixelShuffle(2), 
                             nn.Conv2d(indim//4, outdim, 3, 1, 1),
                             nn.ReLU()
                             )
    def upblock(self, indim, outdim):
        return nn.Sequential(nn.PixelShuffle(2), 
                             nn.Conv2d(indim//4, outdim, 3, 1, 1),
                             nn.ELU(), 
                             nn.BatchNorm2d(outdim),

                             nn.Conv2d(outdim, outdim, 3, 1, 1),
                             nn.ELU(), 
                             nn.BatchNorm2d(outdim)
                             )

    def forward(self, f6, f5, f4, f3):
        d5 = self.upblock1(f6)

        d4 = torch.cat([d5, f5], dim=1)
        d4 = self.upblock2(d4)

        d3 = torch.cat([d4, f4], dim=1)
        d3 = self.upblock3(d3)

        d2 = torch.cat([d3, f3], dim=1)
        d2 = self.upblock4(d2)

        d1 = self.upblock5(d2)
        out = self.conv_org(d1)
        out = self.act(out)

        out_seg = self.conv_seg(d1)
        out_seg = F.softmax(out_seg, dim=1)

        return out, out_seg
        
class SC_UNET_ResBlock_Decoder(nn.Module):
    def __init__(self, dim=768, mode=None, mode2=None): # dim 768*2 = 1536
        super(SC_UNET_ResBlock_Decoder, self).__init__()
        self.mode = mode
        self.mode2 = mode2

        if self.mode == 'feature6' :
            self.conv_block0 = ResBlock(dim, dim, pooling=False, drop_out=False)
            
            self.up0 = self.deconv(dim, dim/2)

            self.conv_block1 = ResBlock(dim, dim/2, pooling=False)
            self.up1 = self.deconv(dim/2, dim/2/2)

            if self.mode2 == 'level1':
                self.conv_block2 = ResBlock(dim/2/2, dim/2/2/2, pooling=False)
                self.up2 = self.deconv(dim/2/2/2, dim/2/2/2)
            else:
                self.conv_block2 = ResBlock(dim/2, dim/2/2, pooling=False)
                self.up2 = self.deconv(dim/2/2, dim/2/2/2)

            if self.mode2 == "level123":
                self.conv_block3 = ResBlock(dim/2/2, dim/2/2/2, pooling=False)
                self.up3 = self.deconv(dim/2/2/2, dim/2/2/2/2)

            else:
                self.conv_block3 = ResBlock(dim/2/2/2, dim/2/2/2/2, pooling=False)
                self.up3 = self.deconv(dim/2/2/2/2, dim/2/2/2/2)

            self.conv_block4 = ResBlock(dim/2/2/2/2, dim/2/2/2/2/2, pooling=False)
            self.up4 = self.deconv(dim/2/2/2/2/2, dim/2/2/2/2/2)
            
            self.conv_block5 = ResBlock(dim/2/2/2/2/2, dim/2/2/2/2/2/2, pooling=False)
            self.up5 = self.deconv(dim/2/2/2/2/2/2, dim/2/2/2/2/2/2)

            self.conv = nn.Conv2d(int(dim/2/2/2/2/2/2), 3, 1, 1)

        else :
            self.conv_block1 = self.conv_block(dim, dim, first=True)
            
            self.up1 = self.deconv(dim, dim/2)

            self.conv_block2 = self.conv_block(dim, dim/2)
            self.up2 = self.deconv(dim/2, dim/2/2)

            self.conv_block3 = self.conv_block(dim/2, dim/2/2)
            self.up3 = self.deconv(dim/2/2, dim/2/2/2)

            if self.mode == 'level4' :
                self.conv_block4 = self.conv_block(dim/2/2, dim/2/2/2)
            else:
                self.conv_block4 = self.conv_block(dim/2/2/2, dim/2/2/2)
                
            self.up4 = self.deconv(dim/2/2/2, dim/2/2/2/2)
            
            if self.mode == 'level5':
                self.conv_block5 = self.conv_block(dim/2/2/2, dim/2/2/2/2)
            else:
                self.conv_block5 = self.conv_block(dim/2/2/2/2, dim/2/2/2/2)
            self.conv = nn.Conv2d(int(dim/2/2/2/2), 3, 1, 1)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def deconv(self, in_channels, out_channels):
        in_channels, out_channels = int(in_channels), int(out_channels)
        return nn.ConvTranspose2d(in_channels, out_channels, 4, 2, padding=1)

    def conv_block(self, in_channels, out_channels, first=False):
        in_channels, out_channels = int(in_channels), int(out_channels)
        if first :
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                nn.ELU(),
                nn.BatchNorm2d(out_channels))
        else:
            layer = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                                nn.ELU(),
                                nn.BatchNorm2d(out_channels),
                                nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
                                nn.ELU(),
                                nn.BatchNorm2d(out_channels))
        return layer

    def forward(self, features):
        f6, f5, f4 = features[0], features[1], features[2]
        d6 = self.conv_block0(f6) 

        d5 = self.up0(d6) 

        d4 = torch.cat((d5, f5), dim=1)
        d4 = self.conv_block1(d4)
        d4 = self.up1(d4)
        
        if self.mode2 in [None, 'level123']:
            d3 = torch.cat((d4, f4), dim=1)
            d3 = self.conv_block2(d3)
        elif self.mode2 == 'level1':
            d3 = self.conv_block2(d4)    
        d3 = self.up2(d3)

        if self.mode2 == 'level123':
            f3 = features[3]
            d2 = torch.cat((d3, f3), dim=1)
            d2 = self.conv_block3(d2)
        else: 
            d2 = self.conv_block3(d3)

        d2 = self.up3(d2) # 128 512 

        after_convd2 = self.conv_block4(d2) # after conv d2
        # res_d1 = torch.cat((after_convd2, d2), dim=1)
        # res_d1 = self.res_up4(res_d1)    
        d1 = self.up4(after_convd2) # 256 1024 

        # after_convd1 = self.conv_block5(res_d1)
        after_convd1 = self.conv_block5(d1)
        out = self.conv(after_convd1)
        out = self.sigmoid(out)

        # cv2.imwrite('d2.png', d2.max(1).values.cpu().detach().numpy()[0]*10)
        # cv2.imwrite('after_convd2.png', after_convd2.max(1).values.cpu().detach().numpy()[0]*10)
        # cv2.imwrite('d1.png', d1.max(1).values.cpu().detach().numpy()[0]*10)
        # cv2.imwrite('after_convd1.png', after_convd1.max(1).values.cpu().detach().numpy()[0]*10)
        return d6, d5, d4, d3, d2, d1, out

# if __name__ == "__main__":
#     input = torch.rand([1, 1, 112, 592])
#     encoder = SC_UNET_Encoder(dim=48)
#     decoder = SC_UNET_Decoder(dim=768)

#     f5, f4_1, f3_1 = encoder(input)
#     logits = decoder(f5, f4_1, f3_1)
    
#     from torchinfo import summary
#     print(summary(encoder, (1, 1, 112, 592)))

#     from torchviz import make_dot
#     make_dot(encoder(input.cuda())).render('sc_unet_test', format='png')
