import torch

class Conv2D():
    def __init__(self, channels, kernel_size=(15, 15), stride=1, padding=7):
        ''' Parameters of Convolution
        channels: channels of input (binary image has 1, RGB something else)
        kernel_size: dimension of weights
        stride: jumping, use 1
        padding: adding zeros to borders of image, recompute for original image size '''
        self.conv = torch.nn.Conv2d(
            in_channels=channels, out_channels=1, kernel_size=kernel_size, \
            padding=padding, stride=stride, bias=False
            )

    def set_weights(self, weights=None, bias=None):
        ''' For reassigning learned weights and bias for testing '''
        self.conv.weight = torch.nn.Parameter(weights.unsqueeze(0))
        

    def forward(self, x):
        ''' FeedForward pass'''
        x = self.conv(x)
        x = (-(torch.abs(torch.log(x)).detach()))
        x = x.squeeze(0)
        return x

    def check_output_shape(self, *args):
        pass
