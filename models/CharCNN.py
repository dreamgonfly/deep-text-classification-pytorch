import torch
from torch import nn

class CharCNN(nn.Module):
    
    def __init__(self, n_classes, dictionary, args):
        super(CharCNN, self).__init__()

        vocabulary_size = dictionary.vocabulary_size
        embed_size=dictionary.vocabulary_size - 1 # except for padding
        embedding_weight=dictionary.embedding
        mode = args.mode
        max_length = args.max_length
        
        if mode == 'large':
            conv_features = 1024
            linear_features = 2048
        elif mode == 'small':
            conv_features = 256
            linear_features = 1024
        else:
            raise NotImplementedError()
        
        # quantization
        self.embedding = nn.Embedding(vocabulary_size, embed_size)
        if embedding_weight is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(embedding_weight), requires_grad=False)
        
        
        conv1 = nn.Sequential(
            nn.Conv1d(in_channels=embed_size, out_channels=conv_features, kernel_size=7),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU()
        )
        conv2 = nn.Sequential(
            nn.Conv1d(in_channels=conv_features, out_channels=conv_features, kernel_size=7),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU()
        )
        conv3 = nn.Sequential(
            nn.Conv1d(in_channels=conv_features, out_channels=conv_features, kernel_size=3),
            nn.ReLU()
        )
        conv4 = nn.Sequential(
            nn.Conv1d(in_channels=conv_features, out_channels=conv_features, kernel_size=3),
            nn.ReLU()
        )
        conv5 = nn.Sequential(
            nn.Conv1d(in_channels=conv_features, out_channels=conv_features, kernel_size=3),
            nn.ReLU()
        )
        conv6 = nn.Sequential(
            nn.Conv1d(in_channels=conv_features, out_channels=conv_features, kernel_size=3),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU()
        )
         
        # according to the paper
        initial_linear_size = (max_length - 96) // 27 * conv_features
        
        linear1 = nn.Sequential(
            nn.Linear(initial_linear_size, linear_features),
            nn.Dropout(),
            nn.ReLU()
        )
        linear2 = nn.Sequential(
            nn.Linear(linear_features, linear_features),
            nn.Dropout(),
            nn.ReLU()
        )
        linear3 = nn.Linear(linear_features, n_classes)
        
        self.convolution_layers = nn.Sequential(conv1, conv2, conv3, conv4, conv5, conv6)
        self.linear_layers = nn.Sequential(linear1, linear2, linear3)
        
    def forward(self, sentences):
#         print(sentences.shape)
        x = self.embedding(sentences)
#         print(x.shape)
        x = x.transpose(1,2)
#         print(x.shape)
        x = self.convolution_layers(x)
#         print(x.shape)
        x = x.view(x.size(0), -1)
#         print(x.shape)
        x = self.linear_layers(x)
#         print(x.shape)

        return x
    
if __name__ == '__main__':
    
    class D:
        vocabulary_size = 69
        embedding = None

    model = CharCNN(mode='small', dictionary=D, n_classes=2)
    from torch.autograd import Variable
    print(model(Variable(torch.LongTensor([[9]*1014]))))
#     model = CharCNN(mode='small', 
#                 vocabulary_size=dictionary.vocabulary_size, embed_size=dictionary.vector_size, num_classes=2,
#                 embedding_weight=dictionary.embedding, max_length=1014, )