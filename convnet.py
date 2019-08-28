import numpy as np
import torch.nn as nn 

class PrintLayer(nn.Module):
    """ Use this to make print() while running the sequantial net """
    def __init__(self, name):
        super(PrintLayer, self).__init__()
        self.name = name

    def forward(self, x):
        print(f'{self.name}: {x.shape}')
        return x

class ReshapeDynamic(nn.Module):
    """ Reshape input """
    def __init__(self, *dim):
        """ Dim can be any shape """
        super(ReshapeDynamic, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.view(*self.dim)

class ConvNet(nn.Module):
    """ A convolutional neural network """
    def __init__(self, code_size, img_size, num_channels, device):
        # Conv2d params: (in_channels, out_channels, kernel size)
        # Kernel size 2 means a square kernel with dimension (2, 2)
        # The kernel weights are all learnt by the network.  Initially,
        #  all these weight are set randomly.
        # Use the following code to check initial values:
        #    print(self.conv1.weight)
        super(ConvNet, self).__init__()
        self.num_channels = num_channels
        self.img_size = img_size
        output_paddings = []
        s = img_size
        # Check if we need to add output padding to our layers when decoding
        #   the signal. Currently, we do this in three steps since we have 
        #   three convolutional nets that all reduce the size of following 
        #   layers
        for _ in range(3):
          if s/2 == s//2:
            output_paddings.append(0)
          else:
            output_paddings.append(1)
          s = s // 2
        # Define the number of features we'll have when we connect to
        #   "the code". The origin of 128 is the number of channels
        #   in the layer adjacent to the code.
        self.num_features = 128*s*s
        self.code_size = code_size
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, 2, stride=2), # Result: 32 channels of size 14x14
            nn.ReLU(True), # Perform ReLU in-place
            nn.Conv2d(32, 64, 2, stride=2), # Result: 64 channels of size 7x7
            nn.ReLU(True),
            nn.Conv2d(64, 128, 2, stride=2), # Result: 128 channels of size 3x3
            nn.ReLU(True),
            ReshapeDynamic((-1, 1, self.num_features)),
            nn.Linear(self.num_features, self.code_size).to(device))
        self.decoder = nn.Sequential(
            nn.Linear(self.code_size, self.num_features).to(device),
            ReshapeDynamic((-1, 128, s, s)),
            nn.ConvTranspose2d(128, 64, 2, stride=2, output_padding=output_paddings.pop()),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 2, stride=2, output_padding=output_paddings.pop()),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, num_channels, 2, stride=2, output_padding=output_paddings.pop()),
            nn.Tanh())
        
    def forward(self, x):
        """ Run the signal through the encoder and the decoder """
        x = self.encoder(x)
        x = self.decoder(x)
        return x