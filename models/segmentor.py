
import torch 
from torch import nn

from adapters import SalsaNextAdapter_1024 as SalsaNextAdapter
from adapters import PatchEmbed_1023 as PatchEmbed
from salsanext import SalsaNextEncoder_1024 as SalsaNextEncdoer
from salsanext import SalsaNextDecoder_1024 as SalsaNextDecoder
from models.modules.adapter_modules1011 import deform_inputs_1024

import torch.nn.functional as F

''' 
injector -> resblock -> extractor
resblock embed_dim에 모두 따르는 것이 너무 channel이 커서 학습 시간이 오래걸린다고 판단됨
injector, extractor의 차원 수를 유동적으로 변경할 수 있으면 channel 수 조절 가능
그거 도전 
'''

class EncoderDecoder(nn.Module):
    def __init__(self, nclasses, img_size, embed_dim=768):
        super(EncoderDecoder, self).__init__()
        self.img_size = img_size
        self.adapter = SalsaNextAdapter(embed_dim=embed_dim)
        self.backbone = SalsaNextEncdoer(nclasses, dim=embed_dim)
        self.patch_embed1 = PatchEmbed(img_size=self.img_size, patch_size=1, in_chans=embed_dim//8, embed_dim=embed_dim//8)
        self.patch_embed2 = PatchEmbed(img_size=self.img_size, patch_size=1, in_chans=embed_dim//4, embed_dim=embed_dim//4)
        self.patch_embed3 = PatchEmbed(img_size=self.img_size, patch_size=1, in_chans=embed_dim//2, embed_dim=embed_dim//2)
        self.patch_embed4 = PatchEmbed(img_size=self.img_size, patch_size=1, in_chans=embed_dim, embed_dim=embed_dim)

        self.conv1x1_f1 = nn.Conv2d(embed_dim//2, embed_dim//4, 1, 1)
        self.conv1x1_f3 = nn.Conv2d(embed_dim//2, embed_dim, 1, 1)
        self.conv1x1_f4 = nn.Conv2d(embed_dim//2, embed_dim, 1, 1)

        self.decode_head = SalsaNextDecoder(nclasses, embed_dim)

    def intercationblock(self, x, c, deform1, deform2, H, W, idx=0):
        layer = self.adapter.interactions[idx-1]
        if idx == 1 :
            block = self.backbone.resBlock1
        if idx == 2 :
            block = self.backbone.resBlock2
        if idx == 3 :
            block = self.backbone.resBlock3
        if idx == 4 :
            block = self.backbone.resBlock4

        x, c = layer(x, c, block, deform1, deform2, H, W, idx)
        
        return x, c

    def forward(self, salsanext_input, adapter_input):

        # adapter spm 
        c1, c2, c3, c4 = self.adapter.spm(adapter_input)
        c2, c3, c4 = self.adapter._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)  

        # salsanext context module
        downCntx = self.backbone.downCntx(salsanext_input)
        downCntx = self.backbone.downCntx2(downCntx)
        downCntx = self.backbone.downCntx3(downCntx) 

        deform1, deform2 = deform_inputs_1024(downCntx) 
        x1, H, W = self.patch_embed1(downCntx) # 256 1248
        bs, n, dim = x1.shape # 319488 32

        # adapter 1
        x1, c_f1 = self.intercationblock(x1, c, deform1, deform2, H, W, idx=1) # output 1/2 
        f1 = x1.transpose(1,2).view(bs, dim*2, H//2, W//2).contiguous()


        deform1, deform2 = deform_inputs_1024(f1, 2)
        x2, H, W = self.patch_embed2(f1) # 128 624
        bs, n, dim = x2.shape # 79872 64
        x2, c_f2 = self.intercationblock(x2, c_f1, deform1, deform2, H, W, idx=2)
        f2 = x2.transpose(1,2).view(bs, dim*2, H//2, W//2).contiguous()

        deform1, deform2 = deform_inputs_1024(f2, 4)
        x3, H, W = self.patch_embed3(f2)
        bs, n, dim = x3.shape 
        x3, c_f3 = self.intercationblock(x3, c_f2, deform1, deform2, H, W, idx=3) 
        f3 = x3.transpose(1, 2).view(bs, dim*2, H//2, W//2).contiguous()

        deform1, deform2 = deform_inputs_1024(f3, 8)
        x4, H, W = self.patch_embed4(f3)
        bs, n, dim = x4.shape 
        x4, c_f4 = self.intercationblock(x4, c_f3, deform1, deform2, H, W, idx=4) 
        f4 = x4.transpose(1, 2).view(bs, dim, H//2, W//2).contiguous()

        c2 = c_f4[:,:c2.size(1),:]
        c3 = c_f4[:,c2.size(1):c2.size(1)+c3.size(1),:]
        c4 = c_f4[:,c2.size(1)+c3.size(1):,:]

        c2 = c2.transpose(1, 2).view(bs, dim//2, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim//2, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim//2, H // 2, W // 2).contiguous()
        c1 = self.adapter.up(c2) + c1
        
        f1 = f1 + self.conv1x1_f1(c1)
        f2 = f2 + c2
        f3 = f3 + self.conv1x1_f3(c3)
        f4 = f4 + self.conv1x1_f4(c4)

        f0 = self.adapter.norm0(downCntx)
        f1 = self.adapter.norm1(f1) # 64 128 624
        f2 = self.adapter.norm2(f2) # 128 64 312
        f3 = self.adapter.norm3(f3) # 256 32 156
        f4 = self.adapter.norm4(f4) # 256 12

        # start decoder 
        up4e = self.decode_head.upBlock1(f4, f3) # 128 32 156
        up3e = self.decode_head.upBlock2(up4e, f2) # 128 64 314
        up2e = self.decode_head.upBlock3(up3e, f1)
        up1e = self.decode_head.upBlock4(up2e, f0)

        logits = self.decode_head.logits(up1e)
        logits = F.softmax(logits, dim=1)

        return logits