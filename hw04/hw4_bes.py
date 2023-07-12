import torch
from torch import nn
from typing import Tuple
import torchvision as tv
from functools import reduce
 
class UnetFromPretrained(torch.nn.Module):
    '''This is my super cool, but super dumb module'''
  
    def __init__(self, encoder: nn.Module, num_classes: int):
        '''
        :param encoder: nn.Sequential, pretrained encoder
        :param num_classes: Python int, number of segmentation classes
        '''
        super(UnetFromPretrained, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential()
 
        self.down_blocks, self.pools, self.block_in_out_channels = self.get_down_blocks(encoder)
        self.num_enc_layers = len(self.down_blocks)
 
 
        self.up_blocks = self.get_up_blocks()
 
        for i, layer in enumerate(self.down_blocks):
            self.add_module(module=self.down_blocks[i], name=f'encoder_layer_{i}')
            if i != len(self.down_blocks) - 1:
                self.add_module(module=self.pools[i], name=f'encoder_layer_pool_{i}')
 
        for i, layer in enumerate(self.up_blocks['seq_blocks']):
            self.add_module(module=self.up_blocks['trans_convs'][i], name=f'decoder_layer_trans_conv_{i}')
            self.add_module(module=self.up_blocks['seq_blocks'][i], name=f'decoder_layer_{i}')
 
        self.add_module(module=self.last_conv, name="last_conv")
 
        # self.features = get_features()
 
     
    # def get_features(self):
    #     features = []
    #     for module in self.modules():
    #         features.
 
 
    def get_down_blocks(self, encoder):
        '''
           return 4 blocks of convolutions ending with maxpool
           according to U-net erchitechture
        '''
        for param in encoder.parameters():
            param.requires_grad = False
 
        modules = []
        curr_layer_mods = []
        pools = []
        block_in_out_channels = []
        convs = []
 
        # for i, mod in enumerate(encoder.modules()):
        for i, mod in enumerate(encoder.features):

            if not(isinstance(mod, nn.MaxPool2d) or isinstance(mod, nn.Conv2d)
                   or isinstance(mod, nn.BatchNorm2d)):
                continue
 
            if isinstance(mod, nn.Linear):
                break
            elif isinstance(mod, nn.Conv2d):
                convs.append(mod)
 
            if isinstance(mod, nn.MaxPool2d):
                modules.append(curr_layer_mods)
                pools.append(mod)
                block_in_out_channels.append(((convs[-1]).in_channels, (convs[-1]).out_channels))
                curr_layer_mods = []
            else:
                curr_layer_mods.append(mod)
             
        if curr_layer_mods:
            modules.append(curr_layer_mods)
        if isinstance(modules[-1][-1], nn.MaxPool2d):
            modules[-1] = modules[-1][:-1]
 
        # if len(pools) == len(modules):
            # pools = pools[:-1]
 
 
        return [nn.Sequential(*block) for block in modules], pools, block_in_out_channels
     
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
                            trans_convs.append(nn.ConvTranspose2d(in_channels=self.block_in_out_channels[i + 1][1],
                                             out_channels=mod.out_channels, kernel_size=2, stride=2, padding=0))
                            # trans_convs.append(nn.ConvTranspose2d(in_channels=self.block_in_out_channels[i + 1][1],
                            #                  out_channels=self.block_in_out_channels[i][1], kernel_size=2, stride=2, padding=0))
                        else:
                            trans_convs.append(nn.ConvTranspose2d(in_channels=mod.out_channels,
                                             out_channels=mod.out_channels, kernel_size=2, stride=2, padding=0))
                            # trans_convs.append(nn.ConvTranspose2d(in_channels=self.block_in_out_channels[i][1],
                            #                  out_channels=mod.out_channels, kernel_size=2, stride=2, padding=0))
                            # trans_convs.append(nn.ConvTranspose2d(in_channels=self.block_in_out_channels[i][1],
                            #                  out_channels=self.block_in_out_channels[i + 1][0], kernel_size=2, stride=2, padding=0))
                        self.features.append(trans_convs[-1])
                        layer.append(nn.Conv2d(mod.out_channels * 2, mod.in_channels,
                            kernel_size=3, stride=1, padding=1))
 
                        is_first_conv = False
                    else:
                        layer.append(nn.Conv2d(mod.out_channels, mod.in_channels,
                            kernel_size=3, stride=1, padding=1))
 
                    layer.append(nn.BatchNorm2d(mod.in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                    layer.append(nn.ReLU(inplace=True))
 
 
            if i == 0:
                # layer.append(nn.Conv2d((layer[0]).out_channels, self.num_classes,
                #     kernel_size=3, stride=1, padding=1))
                # self.last_conv =    nn.Conv2d(self.block_in_out_channels[0][0], self.num_classes, kernel_size=3, stride=1, padding=1)
 
                self.last_conv = nn.Sequential(
                    nn.Conv2d(self.block_in_out_channels[0][0], self.num_classes, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(self.num_classes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True)
                )
 
            for mod in layer:
                self.features.append(mod)
            for mod in self.last_conv:
                self.features.append(mod)
             
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
            print("here?")
 
        for i in range(self.num_enc_layers - 2, -1, -1):
            print(self.up_blocks['trans_convs'][i].in_channels, self.up_blocks['trans_convs'][i].out_channels, "trans in out channels")
            print(x.shape, "x shape")
            x = self.up_blocks['trans_convs'][i](x)
            x = torch.cat([enc_to_concat[i], x], dim=1)
            x = self.up_blocks['seq_blocks'][i](x)
 
        x = self.last_conv(x)
 
        return x
         
 
def save_model(model, destination):
    torch.save(model.state_dict(), destination)
  
def load_model() -> Tuple[nn.Module, str]:
    '''
    :return: model: your trained NN; encoder_name: name of NN, which was used to create your NN
    '''
    vgg13_bn = tv.models.vgg13_bn(True)
    num_classes = 6
    model = UnetFromPretrained(vgg13_bn.features, num_classes)
    model.load_state_dict(torch.load(f'best_model.pth', map_location=torch.device('cpu')))
    encoder_name = 'vgg13_bn'
    return model, encoder_name



class SimpleConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        # self.pool = nn.MaxPool2d(2, 1)

        self.model = nn.Sequential(
            nn.Conv2d(3,3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ,
            nn.ReLU(inplace=True)
            ,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ,
            nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ,
            nn.ReLU(inplace=True)
            ,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ,
            nn.Conv2d(3,1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ,
            nn.ReLU(inplace=True)
            ,
            nn.Conv2d(1, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,
            nn.BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ,
            nn.ReLU(inplace=True)
            ,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ,
            nn.Conv2d(40, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ,
            nn.ReLU(inplace=True)
            ,
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ,
            nn.ReLU(inplace=True)
            ,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ,
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ,
            nn.ReLU(inplace=True)
            ,
            nn.Conv2d(512, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,
            nn.BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ,
            nn.ReLU(inplace=True)
            ,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.features = [elem for elem in self.model]

    def forward(self, x):
        x = self.model(x)
        return x
