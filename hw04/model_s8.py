import torch
from torch import nn
from typing import Tuple
import torchvision as tv
from collections import namedtuple
import os
import numpy as np
  
class UnetFromPretrained(torch.nn.Module):
    '''This is my super cool, but super dumb module'''
   
    def __init__(self, encoder: nn.Module, num_classes: int):
        '''
        :param encoder: nn.Sequential, pretrained encoder
        :param num_classes: Python int, number of segmentation classes
        '''
        super(UnetFromPretrained, self).__init__()
        self.num_classes = num_classes
        self.encoder = encoder
        self.encoder_layers, self.pools, self.decoder_layers, self.trans_convs = self.build_network(encoder)
        self.num_of_pools = len(self.pools)
 
        for i, layer in enumerate(self.decoder_layers):
            self.add_module(module=self.trans_convs[i], name=f'decoder_layer_trans_conv_{i}')
            self.add_module(module=self.decoder_layers[i], name=f'decoder_layer_{i}')
 
 
    def build_network(self, encoder):
        encoder_path = []
        encoder_path_pools = []
        decoder_path = []
        encoder_path_conv_pools = []
        enc_layer = []
        last_enc_layer_out_channels = 0
  
        for i, mod in enumerate(encoder):
            if not isinstance(mod, nn.MaxPool2d):
                enc_layer.append(mod)
 
            if isinstance(mod, nn.Conv2d):
                encoder_path_conv_pools.append(mod)
                last_enc_layer_out_channels = mod.out_channels
            elif isinstance(mod, nn.MaxPool2d):
                encoder_path.append(enc_layer)
                encoder_path_pools.append(mod)
                enc_layer = []
                encoder_path_conv_pools.append(mod)
         
        if enc_layer:
            encoder_path.append(enc_layer)
 
         
        decoder_convs_channels = []
        decoder_layer, trans_convs = [], []
        ConvChannels = namedtuple("ConvChannels", "in_channels out_channels")
        reverse_encoder_path = encoder_path_conv_pools[::-1]
        for i, mod in enumerate(reverse_encoder_path):
            if isinstance(mod, nn.MaxPool2d):
                reverse_encoder_path = reverse_encoder_path[i:]
                break
 
        is_first_conv_in_layer = True
 
        for i, mod in enumerate(reverse_encoder_path):
            if isinstance(mod, nn.Conv2d):
 
                if is_first_conv_in_layer:
                    in_chan_for_conv = mod.out_channels * 2
                    is_first_conv_in_layer = False
                else:
                    in_chan_for_conv = mod.out_channels
                 
                if i == len(reverse_encoder_path) - 1:
                    out_chan_for_conv = mod.out_channels
                else:
                    out_chan_for_conv = mod.in_channels
 
                decoder_convs_channels.append(ConvChannels(in_chan_for_conv, out_chan_for_conv))
                decoder_layer.append(nn.Conv2d(in_chan_for_conv, out_chan_for_conv,
                            kernel_size=mod.kernel_size, stride=mod.stride, padding=mod.padding))
                decoder_layer.append(nn.BatchNorm2d(out_chan_for_conv, eps=1e-05, momentum=0.1,
                                    affine=True, track_running_stats=True))
                decoder_layer.append(nn.ReLU(inplace=True))
 
            elif isinstance(mod, nn.MaxPool2d):
                if i == 0:
                    last_enc_layer_out_channels = reverse_encoder_path[i + 1].out_channels
                    trans_conv = nn.Sequential(
                        (nn.Conv2d(in_channels=last_enc_layer_out_channels,
                         out_channels=last_enc_layer_out_channels * 2, kernel_size=3,
                         stride=1, padding=1)),
                        (nn.BatchNorm2d(last_enc_layer_out_channels * 2, eps=1e-05, momentum=0.1,
                         affine=True, track_running_stats=True)),
                        (nn.ReLU(inplace=True)),
                        # (nn.Conv2d(in_channels=last_enc_layer_out_channels * 2,
                        #  out_channels=last_enc_layer_out_channels * 2, kernel_size=3,
                        #  stride=1, padding=1)),
                        # (nn.BatchNorm2d(last_enc_layer_out_channels * 2, eps=1e-05, momentum=0.1,
                        #  affine=True, track_running_stats=True)),
                        # (nn.ReLU(inplace=True)),
                        (nn.Conv2d(in_channels=last_enc_layer_out_channels * 2,
                         out_channels=reverse_encoder_path[i + 1].out_channels, kernel_size=3,
                         stride=1, padding=1)),
                        (nn.BatchNorm2d(reverse_encoder_path[i + 1].out_channels, eps=1e-05, momentum=0.1,
                         affine=True, track_running_stats=True)),
                        (nn.ReLU(inplace=True)),
                        (nn.ConvTranspose2d(in_channels=reverse_encoder_path[i + 1].out_channels,
                    out_channels=reverse_encoder_path[i + 1].out_channels, kernel_size=mod.kernel_size,
                    stride=mod.stride, padding=mod.padding)))
 
                else:
                    trans_conv = (nn.ConvTranspose2d(in_channels=decoder_convs_channels[-1].out_channels,
                    out_channels=reverse_encoder_path[i + 1].out_channels, kernel_size=mod.kernel_size,
                    stride=mod.stride, padding=mod.padding))
                 
                trans_convs.append(trans_conv)
                if i != 0:
                    decoder_path.append(decoder_layer)
                decoder_layer = []
                is_first_conv_in_layer = True
 
        last_conv = (nn.Conv2d(decoder_convs_channels[-1].out_channels, self.num_classes,
                            kernel_size=3, stride=1, padding=1))
        decoder_layer.append(last_conv)
        decoder_layer.append(nn.BatchNorm2d(self.num_classes, eps=1e-05, momentum=0.1,
                                    affine=True, track_running_stats=True))
        decoder_layer.append(nn.ReLU(inplace=True))
 
        decoder_layer.append(nn.Conv2d(self.num_classes, self.num_classes,
                            kernel_size=1, stride=1, padding=0))
        decoder_layer.append(nn.BatchNorm2d(self.num_classes, eps=1e-05, momentum=0.1,
                                    affine=True, track_running_stats=True))
        decoder_layer.append(nn.ReLU(inplace=True))
 
        decoder_path.append(decoder_layer)
 
        return [nn.Sequential(*layer) for layer in encoder_path], encoder_path_pools, \
               [nn.Sequential(*layer) for layer in decoder_path], trans_convs
  
 
    def forward(self, x):
        from_enc_to_concat = []
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if i < self.num_of_pools:
                from_enc_to_concat.append(x)
                x = self.pools[i](x)
 
        from_enc_to_concat = from_enc_to_concat[::-1]
         
        for i, mod in enumerate(self.decoder_layers):
            x = self.trans_convs[i](x)
            x = torch.cat([from_enc_to_concat[i], x], dim=1)
            x = mod(x)
  
        return x
 
                 
 
          
  
def save_model(model, destination):
    decoder_dict = dict()
    for key, value in model.state_dict().items():
        if not key.startswith("encoder"):
            decoder_dict[key] = value
    torch.save(decoder_dict, destination)
 
def load_model() -> Tuple[nn.Module, str]:
    '''
    :return: model: your trained NN; encoder_name: name of NN, which was used to create your NN
    '''
    vgg16_bn = tv.models.vgg16_bn(True)
    num_classes = 6
    model = UnetFromPretrained(vgg16_bn.features, num_classes)
    decoder_dict = torch.load(f'weights_s8.pth', map_location=torch.device('cpu'))
    model_state_dict = model.state_dict()
    for key, value in decoder_dict.items():
            model_state_dict[key] = value 
    encoder_name = 'vgg16_bn'
    model.load_state_dict(model_state_dict)
 
    return model, encoder_name
 
def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    index = np.argmax(memory_available[:-1])  # Skip the 7th card --- it is reserved for evaluation!!!
    return int(index)
  
def get_device():  
    if torch.cuda.is_available():
        gpu = get_free_gpu()
        device = torch.device(gpu)
    else:
        device = 'cpu'
    return device
 
def get_device_mps():  
    if torch.has_mps:
        gpu = 'mps'
        device = torch.device(gpu)
    else:
        device = 'cpu'
    return device