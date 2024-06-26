# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class StackLSTM_With_Attention_Network(nn.Module):
    def __init__(self, num_layers, dropout_rate, concat_embeddings, hidden_size, input_unit_num, unit_num,
                 embedding_dim, attention_window, num_class, trainable_embeddings, bidirectional):
        super(StackLSTM_With_Attention_Network, self).__init__()
        self.num_layers = num_layers
        self.attention_window = attention_window
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.concat_embedding = nn.Embedding(len(concat_embeddings),embedding_dim)  # vocab_size词汇表大小,embedding_dim词嵌入维度
        self.concat_embedding.weight.data.copy_(concat_embeddings)  # 第一句就是导入词向量
        self.concat_embedding.weight.requires_grad = trainable_embeddings

        if self.bidirectional == True:
            self.BiLstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                                  bidirectional=True, dropout=dropout_rate)
            self.LinearLayer1 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
            self.LinearLayer2 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
            self.LinearLayer3 = nn.Linear(2 * hidden_size, 1, bias=False)
            self.LinearLayer4 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
            self.LinearLayer5 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
            self.fc1 = nn.Linear(input_unit_num + 2*hidden_size, unit_num)
            self.fc2 = nn.Linear(unit_num, unit_num)
            self.fc3 = nn.Linear(unit_num, unit_num)
            self.fc4 = nn.Linear(unit_num, num_class)


        else:
            self.BiLstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                                  bidirectional=False, dropout=dropout_rate)
            self.LinearLayer1 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.LinearLayer2 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.LinearLayer3 = nn.Linear(hidden_size, 1, bias=False)
            self.LinearLayer4 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.LinearLayer5 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.fc1 = nn.Linear(input_unit_num + hidden_size, unit_num)
            self.fc2 = nn.Linear(unit_num, unit_num)
            self.fc3 = nn.Linear(unit_num, unit_num)
            self.fc4 = nn.Linear(unit_num, num_class)


    def forward(self, text, features):
        concat_embedding = self.concat_embedding(text)
        # torch.Size([sentence_lens,batch_size,embedding_size])
        # torch.Size([75, 32, 50])
        output, (hidden, cell) = self.BiLstm(concat_embedding)
        # output shape: torch.Size([sentence_lens,batch_size,hidden_size*2])
        # torch.Size([75,32,200])
        # hidden shape: torch.Size([layer_num*2,batch_size,hidden_size])
        # torch.Size([4,32,100])
        output_state = output[:self.attention_window]
        # Forward output
        # Bidirectional:torch.Size([15,32,100])
        if self.bidirectional == True:
            final_state = torch.cat((hidden[-2], hidden[-1]), 1)
        else:
            final_state = hidden[-1]

        final_state_ = final_state.expand(output_state.shape[0], output_state.shape[1], output_state.shape[2])
        M = F.tanh(self.LinearLayer1(output_state) + self.LinearLayer2(final_state_))
        alpha = F.softmax(self.LinearLayer3(M))
        a = output_state.permute(1, 0, 2)
        b = alpha.permute(1, 2, 0)
        r = torch.bmm(b, a).permute(1, 0, 2)
        h = F.tanh(self.LinearLayer4(final_state) + self.LinearLayer5(r))
        concat_features = torch.cat([h.squeeze(), features], 1)
        o1 = F.tanh(self.fc1(concat_features))
        o2 = F.tanh(self.fc2(o1))
        o3 = F.tanh(self.fc3(o2))
        output = self.fc4(o3)
        return output

class LSTM_With_Attention_Network(nn.Module):
    def __init__(self, num_layers, dropout_rate, concat_embeddings, hidden_size, embedding_dim, attention_window,
                 num_class, trainable_embeddings):
        super(LSTM_With_Attention_Network, self).__init__()
        self.num_layers = num_layers
        self.attention_window = attention_window
        self.concat_embedding = nn.Embedding(len(concat_embeddings),embedding_dim)
        self.concat_embedding.weight.data.copy_(concat_embeddings)
        self.concat_embedding.weight.requires_grad = trainable_embeddings

        self.BiLstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                              bidirectional=False, dropout=dropout_rate)
        self.LinearLayer1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.LinearLayer2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.LinearLayer3 = nn.Linear(hidden_size, 1, bias=False)
        self.LinearLayer4 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.LinearLayer5 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.LinearLayer6 = nn.Linear(hidden_size, num_class)

    def forward(self, text):
        concat_embedding = self.concat_embedding(text)  # torch.Size([sentence_lens,batch_size,embedding_size]),[75, 32, 50]
        output, (hidden, cell) = self.BiLstm(concat_embedding)
        # output shape: torch.Size([sentence_lens,batch_size,hidden_size]) torch.Size([75,32,100])
        # hidden shape: torch.Size([layer_num,batch_size,hidden_size])   torch.Size([2,32,100])
        output_state = output[:self.attention_window]  # torch.Size([15,32,100])
        final_state = hidden[self.num_layers - 1]  # torch.Size([1,32,100])
        final_state_ = final_state.expand(output_state.shape[0], hidden.shape[1],
                                          hidden.shape[2])  # torch.Size([15,32,100])
        M = F.tanh(self.LinearLayer1(output_state) + self.LinearLayer2(final_state_))  # torch.Size([15,32,100])
        alpha = F.softmax(self.LinearLayer3(M))  # torch.Size([15,32,1])
        a = output_state.permute(1, 0, 2)  # torch.Size([32,1,15])
        b = alpha.permute(1, 2, 0)  # torch.Size([32,15,100])
        r = torch.bmm(b, a).permute(1, 0, 2)  # torch.Size([1, 32, 100])
        h = F.tanh(self.LinearLayer4(final_state) + self.LinearLayer5(r))  # torch.Size([1, 32, 100])
        output = self.LinearLayer6(h.squeeze())
        return output

class Attention_StackLSTM_With_Multi_FC(nn.Module):
    def __init__(self, num_layers, dropout_rate, concat_embeddings, hidden_size, input_unit_num, unit_num,
                 embedding_dim, attention_window, num_class, trainable_embeddings, bidirectional, num_fc=4):
        super(Attention_StackLSTM_With_Multi_FC, self).__init__()

        self.num_layers = num_layers
        self.attention_window = attention_window
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_fc = num_fc

        # Concatenate embeddings
        self.concat_embedding = nn.Embedding(len(concat_embeddings), embedding_dim)
        self.concat_embedding.weight.data.copy_(concat_embeddings)
        self.concat_embedding.weight.requires_grad = trainable_embeddings

        # BiLSTM layer
        self.BiLstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                              bidirectional=bidirectional, dropout=dropout_rate)

        # Attention layers
        if self.bidirectional:
            self.LinearLayer1 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
            self.LinearLayer4 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        else:
            self.LinearLayer1 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.LinearLayer4 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.LinearLayer2 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.LinearLayer3 = nn.Linear(2 * hidden_size, 1, bias=False)
        self.LinearLayer5 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(input_unit_num + (2 * hidden_size if bidirectional else hidden_size), unit_num))
        for _ in range(self.num_fc - 1):  # Create num_fc - 1 additional fc layers
            self.fc_layers.append(nn.Linear(unit_num, unit_num))
        self.fc_layers.append(nn.Linear(unit_num, num_class))

    def forward(self, text, features):
        concat_embedding = self.concat_embedding(text)
        output, (hidden, cell) = self.BiLstm(concat_embedding)

        output_state = output[:self.attention_window]
        if self.bidirectional:
            final_state = torch.cat((hidden[-2], hidden[-1]), 1)
        else:
            final_state = hidden[-1]

        final_state_ = final_state.expand(output_state.shape[0], output_state.shape[1], output_state.shape[2])
        M = F.tanh(self.LinearLayer1(output_state) + self.LinearLayer2(final_state_))
        alpha = F.softmax(self.LinearLayer3(M))

        a = output_state.permute(1, 0, 2)
        b = alpha.permute(1, 2, 0)
        r = torch.bmm(b, a).permute(1, 0, 2)

        h = F.tanh(self.LinearLayer4(final_state) + self.LinearLayer5(r))
        concat_features = torch.cat([h.squeeze(), features], 1)

        for layer in self.fc_layers[:-1]:
            concat_features = F.tanh(layer(concat_features))

        output = self.fc_layers[-1](concat_features)
        return output


# define the initial function to init the layer's parameters for the network
def weigth_init(m):
    if isinstance(m, nn.LSTM):
        init.xavier_uniform_(m.weight_ih_l0.data, gain=1)
        init.xavier_uniform_(m.weight_hh_l0.data, gain=1)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight, gain=1)