import torch
import torch.nn as nn
import torch.nn.functional as F



class MortalityPredGRU(nn.Module):
    def __init__(self, config, baseInfo_len, topk, max_seq, device):
        super(MortalityPredGRU, self).__init__()
        self.baseInfo_dim = config.baseInfo_dim
        self.hidden_dim = config.model_hidden_dim
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.baseInfo_len = baseInfo_len
        self.topk = topk
        self.max_seq = max_seq
        self.device = device
        self.embedding = nn.Embedding(num_embeddings=self.baseInfo_len, embedding_dim=self.baseInfo_dim)
        self.gru = nn.GRU(input_size=self.topk+self.baseInfo_dim*3, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=self.dropout)
        self.relu = nn.ReLU()
        self.attention = Attention(self.hidden_dim*2)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim*2, 1),
            nn.Sigmoid()
        )


    def init_hidden(self):
        h0 = torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim).to(self.device)
        return h0


    def forward(self, baseInfo_data, feature_data):
        self.batch_size = baseInfo_data.size(0)
        h0 = self.init_hidden()

        baseInfo_data = self.embedding(baseInfo_data).view(self.batch_size, -1).unsqueeze(1)
        baseInfo_data = baseInfo_data.expand(self.batch_size, self.max_seq, baseInfo_data.size(-1))
        feature_data = torch.cat((baseInfo_data, feature_data), dim=-1)

        feature_data, _ = self.gru(feature_data, h0)
        attn_output = self.attention(self.relu(feature_data))
        feature_data = feature_data * attn_output.unsqueeze(-1)
        feature_data = torch.sum(feature_data, dim=1)
        feature_data = self.fc(feature_data)
        return feature_data.squeeze(-1)



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size, int(self.hidden_size/2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_size/2), 1)
        )


    def forward(self, x):
        x = self.attention(x)
        x = x.squeeze(2)
        x = F.softmax(x, dim=1)
        return x
