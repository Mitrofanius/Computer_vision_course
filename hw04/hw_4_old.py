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
        self.num_enc_layers = 5

        self.down_blocks, self.pools, self.block_out_channels = self.get_down_blocks(encoder)
        self.up_blocks               = self.get_up_blocks()

        for i, layer in enumerate(self.down_blocks):
            self.add_module(module=self.down_blocks[i], name=f'encoder_layer_{i}')

        for i, layer in enumerate(self.up_blocks['seq_blocks']):
            self.add_module(module=self.up_blocks['trans_convs'][i], name=f'decoder_layer_trans_conv_{i}')
            self.add_module(module=self.up_blocks['seq_blocks'][i], name=f'decoder_layer_{i}')
        
        # self.add_module(module=)

        # print(self.num_enc_layers)



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
            # print(mod)
            if issubclass(type(mod), torch.nn.modules.conv.Conv2d):
                enc_convs.append(mod)
                convs_out_channels.append(mod.out_channels)
            # elif issubclass(type(mod), torch.nn.modules.batchnorm.BatchNorm2d):
            #     enc_bns.append(mod)    
            # elif mod.__module__ == 'torch.nn.modules.activation':
            #     enc_acts.append(mod)     
        
        # if len(enc_acts) >= len(enc_convs) <= len(enc_bns):
        #     conv_layers = list(map(list, (zip(enc_convs, enc_bns, enc_acts))))
        # else:
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

        # print(*conv_layers[0])


        num_of_convs_array = [convs_per_unet_layer if i < (self.num_enc_layers - convs_left) else (convs_per_unet_layer + 1) for i in range(self.num_enc_layers)] 
        ind_of_conv = 0
        ind_of_block = 0
        # # for i in range(0, num_of_layers_enc, convs_per_unet_layer):
        while ind_of_conv < num_of_layers_enc:
            layer = conv_layers[ind_of_conv:ind_of_conv+num_of_convs_array[ind_of_block]]
            ind_of_conv += num_of_convs_array[ind_of_block]
        #     # if i + convs_per_unet_layer * convs_left >= num_of_layers_enc:
        #         # layer.append(*(conv_layers[-convs_left:]))
        #     if ind_of_block != 0 and ((num_of_layers_enc - ) % (5 - ind_of_block)) == 0:
        #         ind_of_conv += 1
        #         layer.append(*(conv_layers[ind_of_conv]))
            
            block_out_channels.append(convs_out_channels[ind_of_conv - 1])
            # print(len(block_out_channels), "len")

            if layer:
                layer = reduce(lambda a, b: a+b, layer)
            # if ind_of_conv <= num_of_layers_enc:
            if ind_of_block != self.num_enc_layers - 1:
                pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            unet_down_blocks.append(layer)
            ind_of_block += 1

        # unet_down_blocks = reduce(lambda a, b: a+b, unet_down_blocks)

        return [nn.Sequential(*block) for block in unet_down_blocks], pools, block_out_channels
    
    def get_bridge(self, in_channels):
        pass

    def get_up_blocks(self):
        seq_blocks = []
        trans_convs = []
        for i in range(self.num_enc_layers - 1):
            layer = []
            is_first_conv = True
            for mod in self.down_blocks[i][::-1]:
                if issubclass(type(mod), torch.nn.modules.conv.Conv2d):
                    if is_first_conv:
                        # print(mod.out_channels)
                        # print(self.block_out_channels[i+1], "out channels")
                        if i == self.num_enc_layers - 2:
                            trans_convs.append(nn.ConvTranspose2d(in_channels=self.block_out_channels[i+1], out_channels=mod.out_channels, kernel_size=2, stride=2, padding=0))
                        else:
                            trans_convs.append(nn.ConvTranspose2d(in_channels=mod.out_channels, out_channels=mod.out_channels, kernel_size=2, stride=2, padding=0))
                        layer.append(nn.Conv2d(mod.out_channels * 2, mod.in_channels,
                            kernel_size=3, stride=1, padding=1))
                        # layer.append(nn.Conv2d(mod.out_channels * 2, mod.in_channels,
                        #     kernel_size=mod.kernel_size, stride=mod.stride, padding=mod.padding))
                        # layer.append(nn.Conv2d(mod.out_channels * 2, mod.out_channels // 2,
                        #     kernel_size=3, stride=1, padding=1))
                        # layer.append(nn.BatchNorm2d(mod.out_channels // 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

                        is_first_conv = False
                    else:
                        layer.append(nn.Conv2d(mod.out_channels, mod.in_channels,
                            kernel_size=3, stride=1, padding=1))
                        # layer.append(nn.Conv2d(mod.out_channels, mod.in_channels,
                            # kernel_size=mod.kernel_size, stride=mod.stride, padding=mod.padding))
                        # layer.append(nn.Conv2d(mod.out_channels // 2, mod.out_channels // 2,
                            # kernel_size=3, stride=1, padding=1))
                        # layer.append(nn.BatchNorm2d(mod.out_channels // 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))


                    # layer.append(nn.BatchNorm2d(mod.in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                    # layer.append(nn.ReLU(inplace=True))
                

            if i == 0:
                # layer.append(nn.Conv2d((layer[-3]).out_channels, self.num_classes,
                #     kernel_size=3, stride=1, padding=1))
                layer.append(nn.Conv2d((layer[0]).out_channels, self.num_classes,
                    kernel_size=3, stride=1, padding=1))
            
            seq_blocks.append(layer)

        
        
        return {
                'trans_convs' : trans_convs,
                'seq_blocks': [nn.Sequential(*block) for block in seq_blocks]
                }


 
    def forward(self, x):
        # shape = x.shape
        # return torch.randn(shape[0], self.num_classes, *shape[2:], device=x.device)
        enc_to_concat = []
        for i in range(self.num_enc_layers):
            # print(x.shape, "before")
            x = self.down_blocks[i](x)
            enc_to_concat.append(x)
            # print(x.shape, "in_concat")
            if i < self.num_enc_layers - 1:
                x = self.pools[i](x)
            # print(x.shape, "after")

        # print(x.shape)

        for i in range(self.num_enc_layers - 2, -1, -1):
            # if (self.up_blocks['trans_convs'][i]).in_channels != x.shape[1]:
            #     (self.up_blocks['trans_convs'][i]).in_channels = x.shape[1]
            # print(x.shape, "x shape")
            # print((self.up_blocks['trans_convs'][i]).in_channels, (self.up_blocks['trans_convs'][i]).out_channels, "trans channels")
            x = self.up_blocks['trans_convs'][i](x)
            # print(x.shape, "after transpose")
            # print((enc_to_concat[i]).shape)
            x = torch.cat([enc_to_concat[i], x], dim=1)
            # print(x.shape)
            # print(self.up_blocks['seq_blocks'][i][0].in_channels)
            x = self.up_blocks['seq_blocks'][i](x)
            # print(x.shape)


        # test_brute_conv  = nn.Conv2d(in_channels=x.shape[1], out_channels=self.num_classes, kernel_size=3, stride=1, padding=1)
        # test_brute_trans = nn.ConvTranspose2d(in_channels=self.num_classes, out_channels=self.num_classes, kernel_size=2, stride=2, padding=0)
        # x = test_brute_conv(x)
        # while x.shape[2] != shape[2]:
        #     x = test_brute_trans(x)


        return x


# class DownBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride = 1):
#         super(DownBlock, self).__init__()
#         self.conv1 = nn.Sequential(
#                         nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
#                         nn.BatchNorm2d(out_channels),
#                         nn.ReLU())
#         self.conv2 = nn.Sequential(
#                         nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
#                         nn.BatchNorm2d(out_channels))

#         self.bn1 = torch.nn.BatchNorm2d(out_channels)
#         self.bn2 = torch.nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
#         self.out_channels = out_channels
        
#     def forward(self, x):
#         residual = x
#         out = self.bn1(self.conv1(x))
#         out = self.bn2(self.conv2(out))
#         residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out


 
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



# class SimpleConvModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         return x
