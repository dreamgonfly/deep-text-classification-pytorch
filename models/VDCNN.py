import torch
from torch import nn
from torch.autograd import Variable

class ConvolutionalBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, first_stride=1):
        super(ConvolutionalBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=first_stride, padding=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        
        return self.sequential(x)

class KMaxPool(nn.Module):
    
    def __init__(self, k='half'):
        super(KMaxPool, self).__init__()
        
        self.k = k
    def forward(self, x):
        # x : batch_size, channel, time_steps
        if self.k == 'half':
            time_steps = x.shape(2)
            self.k = time_steps//2
        kmax, kargmax = x.topk(self.k, dim=2)
        return kmax
    
class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, downsample=False, downsample_type='resnet', optional_shortcut=True):
        super(ResidualBlock, self).__init__()
        self.optional_shortcut = optional_shortcut
        self.downsample = downsample
        
        if self.downsample:
            if downsample_type == 'resnet':
                self.pool = None
                first_stride = 2
            elif downsample_type == 'kmaxpool':
                self.pool = KMaxPool(k='half')
                first_stride = 1
            elif downsample_type == 'vgg':
                self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
                first_stride = 1
            else:
                raise NotImplementedError()
        else:
            first_stride = 1

        self.convolutional_block = ConvolutionalBlock(in_channels, out_channels, first_stride=first_stride)
                
        if self.optional_shortcut and self.downsample:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        
        residual = x
        if self.downsample and self.pool:
            x = self.pool(x)
        x = self.convolutional_block(x)
        
        if self.optional_shortcut and self.downsample:
            residual = self.shortcut(residual)

        if self.optional_shortcut:
            x = x + residual
            
        return x
    
class VDCNN(nn.Module):
    
    def __init__(self, n_classes, dictionary, args):
        super(VDCNN, self).__init__()

        vocabulary_size = dictionary.vocabulary_size
        
        depth = args.depth # 29
        embed_size = args.embed_size # 16
        optional_shortcut = args.optional_shortcut # True
        k = args.k # 8
        
        if depth == 9:
            n_conv_layers = {'conv_block_512':2, 'conv_block_256':2,
                             'conv_block_128':2, 'conv_block_64':2}
        elif depth == 17:
            n_conv_layers = {'conv_block_512':2, 'conv_block_256':2,
                             'conv_block_128':2, 'conv_block_64':2}
        elif depth == 29:
            n_conv_layers = {'conv_block_512':4, 'conv_block_256':4,
                             'conv_block_128':10, 'conv_block_64':10}
        elif depth == 49:
            n_conv_layers = {'conv_block_512':6, 'conv_block_256':10,
                             'conv_block_128':16, 'conv_block_64':16}
        
        # quantization
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=16, padding_idx=0)
        
        conv_layers = []
        conv_layers.append(nn.Conv1d(16, 64, kernel_size=3, padding=1))

        for i in range(n_conv_layers['conv_block_64']):
            conv_layers.append(ResidualBlock(64, 64, optional_shortcut=optional_shortcut))
            
        for i in range(n_conv_layers['conv_block_128']):
            if i == 0:
                conv_layers.append(ResidualBlock(64, 128, downsample=True, optional_shortcut=optional_shortcut))
            conv_layers.append(ResidualBlock(128, 128, optional_shortcut=optional_shortcut))

        for i in range(n_conv_layers['conv_block_256']):
            if i == 0:
                conv_layers.append(ResidualBlock(128, 256, downsample=True, optional_shortcut=optional_shortcut))
            conv_layers.append(ResidualBlock(256, 256, optional_shortcut=optional_shortcut))

        for i in range(n_conv_layers['conv_block_512']):
            if i == 0:
                conv_layers.append(ResidualBlock(256, 512, downsample=True, optional_shortcut=optional_shortcut))
            conv_layers.append(ResidualBlock(512, 512, optional_shortcut=optional_shortcut))
        
        self.conv_layers = nn.Sequential(*conv_layers)
        self.kmax_pooling = KMaxPool(k=k)
        
        linear_layers = []
        
        linear_layers.append(nn.Linear(512 * k, 2048))
        linear_layers.append(nn.Linear(2048, 2048))
        linear_layers.append(nn.Linear(2048, n_classes))
        
        self.linear_layers = nn.Sequential(*linear_layers)
        
    def forward(self, sentences):
    
        x = self.embedding(sentences)
        x = x.transpose(1,2) # (batch_size, sequence_length, embed_size) -> (batch_size, embed_size, sequence_length)
        x = self.conv_layers(x)
        x = self.kmax_pooling(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.linear_layers(x)

        return x
    
if __name__ == '__main__':
    
    model = VDCNN(depth=29, vocabulary_size=29, embed_size=16, n_classes=2, k=2)
    rand_inputs = Variable(torch.LongTensor(8, 1104).random_(0, 29))
    print(model(rand_inputs))