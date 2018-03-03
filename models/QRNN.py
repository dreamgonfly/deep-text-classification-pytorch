import torch
from torch import nn
from torch.autograd import Variable

class QRNNLayer(nn.Module):
    
    def __init__(self, input_size, hidden_size, kernel_size=2, pooling='fo', zoneout=0.5):
        super(QRNNLayer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.zoneout = zoneout
        
        self.conv_z = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size)
        self.conv_f = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size)
        self.conv_o = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size)
        self.conv_i = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=zoneout)
        
    def forward(self, x):
        
        zero_padding = Variable(torch.zeros(x.size(0), self.input_size, self.kernel_size-1), requires_grad=False)
        if x.is_cuda:
            zero_padding = zero_padding.cuda()
        x_padded = torch.cat([zero_padding, x], dim=2)
        
        z = self.tanh(self.conv_z(x_padded))
        if self.zoneout > 0:
            f = 1 - self.dropout(1 - self.sigmoid(self.conv_f(x_padded)))
        else:
            f = self.sigmoid(self.conv_f(x_padded))
        o = self.sigmoid(self.conv_o(x_padded))
        i = self.sigmoid(self.conv_i(x_padded))
        
        h_list, c_list = [], []
        h_prev = Variable(torch.zeros(x.size(0), self.hidden_size), requires_grad=False)
        c_prev = Variable(torch.zeros(x.size(0), self.hidden_size), requires_grad=False)
        if x.is_cuda:
            h_prev = h_prev.cuda()
            c_prev = c_prev.cuda()
            
        for t in range(x.size(2)):
            z_t = z[:, :, t]
            f_t = f[:, :, t]
            o_t = o[:, :, t]
            i_t = i[:, :, t]
            h_prev, c_prev = self.pool(h_prev, c_prev, z_t, f_t, o_t, i_t)
            h_list.append(h_prev)
            if c_prev is not None:
                c_list.append(c_prev)
        
        h = torch.stack(h_list, dim=2)
        if c_prev is not None:
            c = torch.stack(c_list, dim=2)    
            return h, c
        else:
            return h, None
        
    def pool(self, h_prev, c_prev, z_t, f_t, o_t, i_t):
        
        if self.pooling == 'f':
            c_t = None
            h_t = f_t * h_prev + (1-f_t) * z_t
        elif self.pooling == 'fo':
            c_t = f_t * c_prev + (1-f_t) * z_t
            h_t = o_t * c_t
        elif self.pooling == 'ifo':
            c_t = f_t * c_prev + i_t * z_t
            h_t = o_t * c_t
            
        return h_t, c_t
    
class QRNN(nn.Module):
    
    def __init__(self, n_classes, dictionary, args):
        super(QRNN, self).__init__()
        
        vocab_size = dictionary.vocabulary_size
        embed_size = dictionary.vector_size
        hidden_size = args.hidden_size
        num_layers = args.num_layers # default : 1
        kernel_size = args.kernel_size # default : 2
        pooling =  args.pooling # default : 'fo'
        zoneout = args.zoneout # default : 0.5
        dropout = args.dropout # default : 0.3
        dense =  args.dense # default : True
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        if dictionary.embedding is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(dictionary.embedding), requires_grad=False)
        self.dropout = nn.Dropout(p=dropout)
        self.dense = dense
        
        qrnn_layers = []
        input_size = embed_size
        for _ in range(num_layers-1):
            qrnn_layers.append(QRNNLayer(input_size, hidden_size, kernel_size, pooling, zoneout))
            if self.dense:
                input_size += hidden_size
            else:
                input_size = hidden_size

        self.qrnn_layers = nn.ModuleList(qrnn_layers)
        self.linear = nn.Linear(in_features=input_size, out_features=n_classes)
        
    def forward(self, x):
        
        # x : (batch_size, timestpes)
        
        x = self.embedding(x).transpose(1,2) # batch_size, channels, timesteps
        for qrnn_layer in self.qrnn_layers:
            residual = x
            h, c = qrnn_layer(x)
            x = self.dropout(h)
            if self.dense:
                x = torch.cat([x, residual], dim=1) 
            else:
                x = x
        
        last_timestep = x[:, :, -1]
        return self.linear(last_timestep)        
    
if __name__ == '__main__':
    
    import argparse

    QRNN_parser = argparse.ArgumentParser('QRNN')
    QRNN_parser.add_argument('--batch_size', type=int, default=64)
#     QRNN_parser.add_argument('--preprocess_level', type=str, default='word', choices=['word', 'char'])
#     QRNN_parser.add_argument('--dictionary', type=str, default='WordDictionary', choices=['WordDictionary', 'AllCharDictionary'])
#     QRNN_parser.add_argument('--max_vocab_size', type=int, default=50000) 
#     QRNN_parser.add_argument('--min_count', type=int, default=None)
#     QRNN_parser.add_argument('--start_end_tokens', type=bool, default=False)
#     QRNN_parser.add_argument('--min_length', type=int, default=5)
#     QRNN_parser.add_argument('--max_length', type=int, default=300) 
#     QRNN_parser.add_argument('--sort_dataset', action='store_true')
    QRNN_parser.add_argument('--embed_size', type=int, default=128)
    QRNN_parser.add_argument('--hidden_size', type=int, default=300)
    QRNN_parser.add_argument('--num_layers', type=int, default=4)
    QRNN_parser.add_argument('--kernel_size', type=int, default=2)
    QRNN_parser.add_argument('--pooling', type=str, default='fo')
    QRNN_parser.add_argument('--zoneout', type=float, default=0.5)
    QRNN_parser.add_argument('--dropout', type=float, default=0.3)
    QRNN_parser.add_argument('--dense', type=bool, default=True)
    QRNN_parser.add_argument('--epochs', type=int, default=10)
    QRNN_parser.set_defaults(model=QRNN)
    
    class dictionary:
        vocabulary_size = 10000
        vector_size = 128
        embedding = None

    seq_len = 231
    n_classes = 2
    args = QRNN_parser.parse_args()
    
    rand_inputs = torch.autograd.Variable(torch.LongTensor(args.batch_size, seq_len).random_(0, dictionary.vocabulary_size))
    
    model = QRNN(n_classes, dictionary, args)
    rand_outputs = model(rand_inputs)
    
    assert rand_outputs.shape == (args.batch_size, n_classes)