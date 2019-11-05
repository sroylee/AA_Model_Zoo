import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class CNN_n(nn.Module):
    def __init__(self, args, data_info):
        super(CNN_n, self).__init__()
        self.args = args
        
        V = data_info['emb_num']
        D = args.emb_dim
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        C = args.output_size
        
        self.ngram_emb = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, 500)
        self.fc2 = nn.Linear(500, C)
        
    def forward(self, x):
        x = self.ngram_emb(x)
        #x = self.dropout(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        #x = self.dropout(x)
        predict = self.fc1(x)
        predict = self.fc2(predict)
        return predict
