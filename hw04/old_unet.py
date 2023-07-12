import torch
from torch import nn
from typing import Tuple
import torchvision as tv
from functools import reduce

class UnetFromPretrained_old(torch.nn.Module):
    '''This is my super cool, but super dumb module'''
 
    def __init__(self, encoder: nn.Module, num_classes: int):
        '''
        :param encoder: nn.Sequential, pretrained encoder
        :param num_classes: Python int, number of segmentation classes
        '''
        super(UnetFromPretrained_old, self).__init__()
        self.num_classes = num_classes
        self.num_enc_layers = 5

        self.down_blocks, self.pools, self.block_out_channels = self.get_down_blocks(encoder)
        self.up_blocks               = self.get_up_blocks()

        for i, layer in enumerate(self.down_blocks):
            self.add_module(module=self.down_blocks[i], name=f'encoder_layer_{i}')

        for i, layer in enumerate(self.up_blocks['seq_blocks']):
            self.add_module(module=self.up_blocks['trans_convs'][i], name=f'decoder_layer_trans_conv_{i}')
            self.add_module(module=self.up_blocks['seq_blocks'][i], name=f'decoder_layer_{i}')

        self.add_module(module=self.last_conv, name="last_conv")



    def get_down_blocks(self, encoder):
        '''
           return 4 blocks of convolutions ending with maxpool
           according to U-net erchitechture
        '''
        for param in encoder.parameters():
            param.requires_grad = False

        enc_convs = []
        enc_acts  = []
        enc_bns   = []
        convs_out_channels = []
        for mod in encoder.modules():
            if issubclass(type(mod), torch.nn.modules.conv.Conv2d):
                enc_convs.append(mod)
                convs_out_channels.append(mod.out_channels)
            elif issubclass(type(mod), torch.nn.modules.batchnorm.BatchNorm2d):
                enc_bns.append(mod)    
            elif mod.__module__ == 'torch.nn.modules.activation':
                enc_acts.append(mod)     
        
        if len(enc_acts) >= len(enc_convs) <= len(enc_bns):
            conv_layers = list(map(list, (zip(enc_convs, enc_bns, enc_acts))))
        else:
            conv_layers = list(map(list, (zip(enc_convs))))

        num_of_layers_enc = len(conv_layers)
        convs_per_unet_layer = num_of_layers_enc // self.num_enc_layers
        while convs_per_unet_layer == 0:
            self.num_enc_layers -= 1
            convs_per_unet_layer = num_of_layers_enc // self.num_enc_layers
        convs_left = num_of_layers_enc % self.num_enc_layers

        unet_down_blocks = []
        pools = []
        block_out_channels = []

        num_of_convs_array = [convs_per_unet_layer if i < (self.num_enc_layers - convs_left) else (convs_per_unet_layer + 1) for i in range(self.num_enc_layers)] 
        ind_of_conv = 0
        ind_of_block = 0
        while ind_of_conv < num_of_layers_enc:
            layer = conv_layers[ind_of_conv:ind_of_conv+num_of_convs_array[ind_of_block]]
            ind_of_conv += num_of_convs_array[ind_of_block]
            block_out_channels.append(convs_out_channels[ind_of_conv - 1])

            if layer:
                layer = reduce(lambda a, b: a+b, layer)
            if ind_of_block != self.num_enc_layers - 1:
                pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            unet_down_blocks.append(layer)
            ind_of_block += 1

        return [nn.Sequential(*block) for block in unet_down_blocks], pools, block_out_channels
    
    def get_up_blocks(self):
        seq_blocks = []
        trans_convs = []
        for i in range(self.num_enc_layers - 1):
            layer = []
            is_first_conv = True
            for mod in self.down_blocks[i][::-1]:
                if issubclass(type(mod), torch.nn.modules.conv.Conv2d):
                    if is_first_conv:
                        if i == self.num_enc_layers - 2:
                            trans_convs.append(nn.ConvTranspose2d(in_channels=self.block_out_channels[i+1], out_channels=mod.out_channels, kernel_size=2, stride=2, padding=0))
                        else:
                            trans_convs.append(nn.ConvTranspose2d(in_channels=mod.out_channels, out_channels=mod.out_channels, kernel_size=2, stride=2, padding=0))
                       
                        layer.append(nn.Conv2d(mod.out_channels * 2, mod.in_channels,
                            kernel_size=3, stride=1, padding=1))

                        is_first_conv = False
                    else:
                        layer.append(nn.Conv2d(mod.out_channels, mod.in_channels,
                            kernel_size=3, stride=1, padding=1))

            if i == 0:
                # layer.append(nn.Conv2d((layer[0]).out_channels, self.num_classes,
                #     kernel_size=3, stride=1, padding=1))
                self.last_conv = nn.Conv2d((layer[0]).out_channels, self.num_classes,
                                         kernel_size=3, stride=1, padding=1)
            
            seq_blocks.append(layer)
        
        return {
                'trans_convs' : trans_convs,
                'seq_blocks': [nn.Sequential(*block) for block in seq_blocks]
                }


 
    def forward(self, x):
        shape = x.shape
        enc_to_concat = []
        for i in range(self.num_enc_layers):
            x = self.down_blocks[i](x)
            enc_to_concat.append(x)
            if i < self.num_enc_layers - 1:
                x = self.pools[i](x)

        for i in range(self.num_enc_layers - 2, -1, -1):
            x = self.up_blocks['trans_convs'][i](x)
            x = torch.cat([enc_to_concat[i], x], dim=1)
            x = self.up_blocks['seq_blocks'][i](x)

        x = self.last_conv(x)

        return x

    