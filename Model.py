import torch.nn as nn
from timesformer_pytorch import TimeSformer

class GPTS(nn.Module):                   # TMSFM
    def __init__(self):
        super(GPTS, self).__init__()
        self.model = TimeSformer(        # attribute of TimeSformer 
            dim=512,                     # model dimensionality
            image_size=9,                # size of input images
            patch_size=3,                # size of patches used for processing input
            num_frames=3,                # num of frames in input sequence
            num_classes=1,               # num of output classes
            depth=12,                    # num of layers in model 
            heads=8,                     # num of attention heads in model
            dim_head=64,                 # dimensionality of attention head
            attn_dropout=0,              # dropout rate applied to attention mechanism
            ff_dropout=0                 # dropout rate applied to feed-forward layers
        )
    
    def forward(self, x):                # model architecture
        x = self.model(x)               
        return x