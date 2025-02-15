import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

import cliport.utils.utils as utils
from cliport.models.resnet import IdentityBlock, ConvBlock
from cliport.models.core.unet import Up
from cliport.models.core import fusion
from cliport.models.core.clip import tokenize
from cliport.models.clip_lingunet_lat import CLIPLingUNetLat


class CLIPLingUNet(CLIPLingUNetLat):
    """ CLIP RN50 with U-Net skip connections """

    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super().__init__(input_shape, output_dim, cfg, device, preprocess)

    def _build_decoder(self):
        # language
        self.lang_fuser1 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 2).to(self.device)
        self.lang_fuser2 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 4).to(self.device)
        self.lang_fuser3 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 8).to(self.device)

        self.proj_input_dim = 512 if 'word' in self.lang_fusion_type else 1024
        self.lang_proj1 = nn.Linear(self.proj_input_dim, 1024).to(self.device)
        self.lang_proj2 = nn.Linear(self.proj_input_dim, 512).to(self.device)
        self.lang_proj3 = nn.Linear(self.proj_input_dim, 256).to(self.device)

        # vision
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        ).to(self.device)

        self.up1 = Up(2048, 1024 // self.up_factor, self.bilinear).to(self.device)

        self.up2 = Up(1024, 512 // self.up_factor, self.bilinear).to(self.device)

        self.up3 = Up(512, 256 // self.up_factor, self.bilinear).to(self.device)

        self.layer1 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        ).to(self.device)

        self.layer2 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        ).to(self.device)

        self.layer3 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        ).to(self.device)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
        ).to(self.device)

    def forward(self, x, l):
        x = self.preprocess(x, dist='clip')
       
        # pdb.set_trace()

        in_type = x.dtype
        in_shape = x.shape
        x = x[:,:3]  #  RGB
        x, im = self.encode_image(x)
        x = x.to(in_type)

        # encode text
        l_enc, l_emb, l_mask = self.encode_text(l)
        l_input = l_emb if 'word' in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=x.dtype)

        # encode image
        assert x.shape[1] == self.input_dim
        # pdb.set_trace()
        x = self.conv1(x)

        x = self.lang_fuser1(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj1)
        x = self.up1(x, im[-2])

        x = self.lang_fuser2(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj2)
        x = self.up2(x, im[-3])

        x = self.lang_fuser3(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj3)
        x = self.up3(x, im[-4])

        for layer in [self.layer1, self.layer2, self.layer3, self.conv2]:
            x = layer(x)

        # pdb.set_trace()
        x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
        return x
    
class CLIPLingUNetLessLayers(CLIPLingUNetLat):
    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super().__init__(input_shape, output_dim, cfg, device, preprocess)

    def _build_decoder(self):
        # language
        self.lang_fuser1 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 2)
        self.lang_fuser2 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 4)
        self.lang_fuser3 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 8)

        self.proj_input_dim = 512 if 'word' in self.lang_fusion_type else 1024
        self.lang_proj1 = nn.Linear(self.proj_input_dim, 1024)
        self.lang_proj2 = nn.Linear(self.proj_input_dim, 512)
        self.lang_proj3 = nn.Linear(self.proj_input_dim, 256)

        # vision
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )

        self.up1 = Up(2048, 1024 // self.up_factor, self.bilinear)

        self.up2 = Up(1024, 512 // self.up_factor, self.bilinear)

        self.up3 = Up(512, 256 // self.up_factor, self.bilinear)

        self.layer1 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=4),
        )

        self.conv21 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1)
        )

        self.conv22 = nn.Sequential(
            nn.Conv2d(32, self.output_dim, kernel_size=1)
        )

    def forward(self, x, l):
        x = self.preprocess(x, dist='clip')
        # pdb.set_trace()

        in_type = x.dtype
        in_shape = x.shape
        x = x[:,:3]  #  RGB
        x, im = self.encode_image(x)
        x = x.to(in_type)

        # encode text
        l_enc, l_emb, l_mask = self.encode_text(l)
        l_input = l_emb if 'word' in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=x.dtype)

        # encode image
        assert x.shape[1] == self.input_dim
        x = self.conv1(x)

        x = self.lang_fuser1(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj1)
        x = self.up1(x, im[-2])

        x = self.lang_fuser2(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj2)
        x = self.up2(x, im[-3])

        x = self.lang_fuser3(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj3)
        x = self.up3(x, im[-4])     # [1, 128, 80, 80]

        # pdb.set_trace()
        x = self.layer1(x)          # [1, 64, 320, 320]

        x = self.conv21(x)          # [1, 32, 320, 320]
        x = self.conv22(x)          # [1, 1, 320, 320]

        # pdb.set_trace()
        x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear') #
        return x