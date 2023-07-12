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
        self.encoder = encoder
        self.encoder_path, self.decoder_path, self.last_conv = self.build_network(encoder)

        # self.features = nn.Sequential(*self.encoder_path, *self.decoder_path, self.last_conv)
        # self.features = nn.Sequential(*self.decoder_path, self.last_conv)
        # for i, mod in enumerate(self.features):
            # self.add_module(module=mod, name=f"module_{i}")
 
    def build_network(self, encoder):
        # for param in encoder.parameters():
        #     param.requires_grad = False

        encoder_path = []
        decoder_path = []
        encoder_path_conv_pools = []
 
        for i, mod in enumerate(encoder):
            encoder_path.append(mod)

            if isinstance(mod, nn.Conv2d):
                encoder_path_conv_pools.append(mod)
                # decoder_path.append(nn.Conv2d(mod.out_channels, mod.in_channels,
                            # kernel_size=mod.kernel_size, stride=mod.stride, padding=mod.padding))
            elif isinstance(mod, nn.MaxPool2d):
                encoder_path_conv_pools.append(mod)
                # decoder_path.append(nn.ConvTranspose2d(in_channels=decoder_path[-1].out_channels,
                                            #  out_channels=decoder_path[-1].out_channels, kernel_size=mod.kernel_size, stride=mod.stride, padding=mod.padding))

        reverse_encoder_path = encoder_path_conv_pools[::-1]
        for i, mod in enumerate(reverse_encoder_path):
            if isinstance(mod, nn.Conv2d):
                decoder_path.append(nn.Conv2d(mod.out_channels, mod.in_channels,
                            kernel_size=mod.kernel_size, stride=mod.stride, padding=mod.padding))
            elif isinstance(mod, nn.MaxPool2d):
                if i == 0:
                    decoder_path.append(nn.ConvTranspose2d(in_channels=reverse_encoder_path[i + 1].out_channels,
                    out_channels=reverse_encoder_path[i + 1].out_channels, kernel_size=mod.kernel_size, stride=mod.stride, padding=mod.padding))
                else:
                    decoder_path.append(nn.ConvTranspose2d(in_channels=decoder_path[-1].out_channels,
                    out_channels=reverse_encoder_path[i + 1].out_channels, kernel_size=mod.kernel_size, stride=mod.stride, padding=mod.padding))
        last_conv = (nn.Conv2d(decoder_path[-1].out_channels, self.num_classes,
                            kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*encoder_path), nn.Sequential(*decoder_path), last_conv
 

    def forward(self, x):
        x = self.encoder_path(x)
        # x = self.encoder(x)
        x = self.decoder_path(x)
        x = self.last_conv(x)
 
        return x
         
 
def save_model(model, destination):
    # torch.save(model.state_dict(), destination)
    decoder_dict = dict()
    for key, value in model.state_dict().items():
        if not key.startswith("encoder"):
            decoder_dict[key] = value
    torch.save(decoder_dict, destination)

def load_model() -> Tuple[nn.Module, str]:
    '''
    :return: model: your trained NN; encoder_name: name of NN, which was used to create your NN
    '''
    # vgg11_bn = tv.models.vgg11_bn(weights=tv.models.VGG11_BN_Weights.IMAGENET1K_V1)
    vgg11_bn = tv.models.vgg11_bn()
    num_classes = 6
    model = UnetFromPretrained(vgg11_bn.features, num_classes)
    # model.decoder_path.load_state_dict(torch.load(f'best_model.pth', map_location=torch.device('cpu')))
    decoder_dict = torch.load(f'best_model.pth', map_location=torch.device('cpu'))
    # model_state_dict = model.state_dict()
    for key, value in decoder_dict.items():
            # model_state_dict[key] = value
            model.state_dict()[key] = value 
    encoder_name = 'vgg11_bn'
    return model, encoder_name














# import torch
# from torch import nn
# from typing import Tuple
# import torchvision as tv
# from functools import reduce
  
# class UnetFromPretrained(torch.nn.Module):
#     '''This is my super cool, but super dumb module'''
   
#     def __init__(self, encoder: nn.Module, num_classes: int):
#         '''
#         :param encoder: nn.Sequential, pretrained encoder
#         :param num_classes: Python int, number of segmentation classes
#         '''
#         super(UnetFromPretrained, self).__init__()
#         self.num_classes = num_classes
#         self.encoder = encoder
#         self.encoder_path, self.decoder_path, self.last_conv = self.build_network(encoder)
 
#         # self.features = nn.Sequential(*self.encoder_path, *self.decoder_path, self.last_conv)
#         # self.features = nn.Sequential(*self.decoder_path, self.last_conv)
#         # for i, mod in enumerate(self.features):
#             # self.add_module(module=mod, name=f"module_{i}")
  
#     def build_network(self, encoder):
#         # for param in encoder.parameters():
#         #     param.requires_grad = False
 
#         encoder_path = []
#         decoder_path = []
#         encoder_path_conv_pools = []
  
#         for i, mod in enumerate(encoder):
#             encoder_path.append(mod)
 
#             if isinstance(mod, nn.Conv2d):
#                 encoder_path_conv_pools.append(mod)
#                 # decoder_path.append(nn.Conv2d(mod.out_channels, mod.in_channels,
#                             # kernel_size=mod.kernel_size, stride=mod.stride, padding=mod.padding))
#             elif isinstance(mod, nn.MaxPool2d):
#                 encoder_path_conv_pools.append(mod)
#                 # decoder_path.append(nn.ConvTranspose2d(in_channels=decoder_path[-1].out_channels,
#                                             #  out_channels=decoder_path[-1].out_channels, kernel_size=mod.kernel_size, stride=mod.stride, padding=mod.padding))
 
#         reverse_encoder_path = encoder_path_conv_pools[::-1]
#         for i, mod in enumerate(reverse_encoder_path):
#             if isinstance(mod, nn.Conv2d):
#                 decoder_path.append(nn.Conv2d(mod.out_channels, mod.in_channels,
#                             kernel_size=mod.kernel_size, stride=mod.stride, padding=mod.padding))
#             elif isinstance(mod, nn.MaxPool2d):
#                 if i == 0:
#                     decoder_path.append(nn.ConvTranspose2d(in_channels=reverse_encoder_path[i + 1].out_channels,
#                     out_channels=reverse_encoder_path[i + 1].out_channels, kernel_size=mod.kernel_size, stride=mod.stride, padding=mod.padding))
#                 else:
#                     decoder_path.append(nn.ConvTranspose2d(in_channels=decoder_path[-1].out_channels,
#                     out_channels=reverse_encoder_path[i + 1].out_channels, kernel_size=mod.kernel_size, stride=mod.stride, padding=mod.padding))
#         last_conv = (nn.Conv2d(decoder_path[-1].out_channels, self.num_classes,
#                             kernel_size=3, stride=1, padding=1))
#         return nn.Sequential(*encoder_path), nn.Sequential(*decoder_path), last_conv
  
 
#     def forward(self, x):
#         x = self.encoder_path(x)
#         # x = self.encoder(x)
#         x = self.decoder_path(x)
#         x = self.last_conv(x)
  
#         return x
          
  
# def save_model(model, destination):
#     # torch.save(model.state_dict(), destination)
#     decoder_dict = dict()
#     for key, value in model.state_dict().items():
#         if not key.startswith("encoder"):
#             decoder_dict[key] = value
#     torch.save(decoder_dict, destination)
 
# def load_model() -> Tuple[nn.Module, str]:
#     '''
#     :return: model: your trained NN; encoder_name: name of NN, which was used to create your NN
#     '''
#     # vgg11_bn = tv.models.vgg11_bn(weights=tv.models.VGG11_BN_Weights.IMAGENET1K_V1)
#     vgg11_bn = tv.models.vgg11_bn()
#     num_classes = 6
#     model = UnetFromPretrained(vgg11_bn.features, num_classes)
#     # model.decoder_path.load_state_dict(torch.load(f'best_model.pth', map_location=torch.device('cpu')))
#     decoder_dict = torch.load(f'best_model.pth', map_location=torch.device('cpu'))
#     # model_state_dict = model.state_dict()
#     for key, value in decoder_dict.items():
#             # model_state_dict[key] = value
#             model.state_dict()[key] = value 
#     encoder_name = 'vgg11_bn'
#     return model, encoder_name