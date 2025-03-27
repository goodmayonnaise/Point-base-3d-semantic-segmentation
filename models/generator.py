
from models.scunet import FixedStridedEncoder, FixedDecoder_seg

from torch import nn

class EncoderDecoder(nn.Module):
    def __init__(self, dim=48, relu=False):
        super(EncoderDecoder, self).__init__()

        self.encoder = FixedStridedEncoder(dim=dim)
        self.decoder = FixedDecoder_seg(dim=dim*32, relu=relu)

    def forward(self, x):
        f6, f5, f4, f3 = self.encoder(x)
        out, out_seg = self.decoder(f6, f5, f4, f3)

        return out, out_seg
        
