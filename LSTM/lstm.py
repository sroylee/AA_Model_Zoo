import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import time

class BaseRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, input_dropout_p, dropout_p, \
                          n_layers, rnn_cell_name):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.rnn_cell_name = rnn_cell_name
        if rnn_cell_name.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell_name.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell_name))
        self.dropout_p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

class EncoderRNN(BaseRNN):
    def __init__(self, args, vocab_size, embed_model=None, emb_size=100, hidden_size=128, \
                 input_dropout_p=0, dropout_p=0, n_layers=1, bidirectional=False, \
                 rnn_cell=None, rnn_cell_name='gru', variable_lengths=True):
        super(EncoderRNN, self).__init__(vocab_size, emb_size, hidden_size,
              input_dropout_p, dropout_p, n_layers, rnn_cell_name)
        self.variable_lengths = variable_lengths
        self.bidirectional = bidirectional
        self.fc = nn.Linear(hidden_size, args.output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        if embed_model == None:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        else:
            self.embedding = embed_model
        if rnn_cell == None:
            self.rnn = self.rnn_cell(emb_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        else:
            self.rnn = rnn_cell

    def forward(self, input_var, input_lengths=None):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        #pdb.set_trace()
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        predict = self.fc(hidden).squeeze(0)
        return predict
