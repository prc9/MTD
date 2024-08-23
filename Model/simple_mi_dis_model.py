import copy
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from torch.autograd import Variable

class SimpleDiscriminator(nn.Module):
    def __init__(self, itemCount, feat_shape, out_feat_shape):
        super(SimpleDiscriminator, self).__init__()
        input_dim = itemCount + out_feat_shape
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, Adj, embedding):
        x = torch.cat((Adj, embedding), 1)
        return self.model(x)

class SimpleGenerator(nn.Module):
    def __init__(self, itemCount, feat_shape, out_feat_shape):
        super(SimpleGenerator, self).__init__()
        self.itemCount = itemCount
        self.feat_shape = feat_shape
        self.out_feat_shape = out_feat_shape

        self.hetero_conv = dglnn.HeteroGraphConv({
            'MvsD': dglnn.GraphConv(self.feat_shape, self.out_feat_shape),
            'DvsM': dglnn.GraphConv(self.feat_shape, self.out_feat_shape),
            'MvsT': dglnn.GraphConv(self.feat_shape, self.out_feat_shape),
            'TvsM': dglnn.GraphConv(self.feat_shape, self.out_feat_shape),
            'TvsD': dglnn.GraphConv(self.feat_shape, self.out_feat_shape),
            'DvsT': dglnn.GraphConv(self.feat_shape, self.out_feat_shape)
        }, aggregate='sum')

        self.model = nn.Sequential(
            nn.Linear(self.itemCount + self.out_feat_shape, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, self.itemCount),
            nn.Sigmoid()
        )

    def forward(self, g, h, Adj, size, leftIndex):
        h_conv = self.hetero_conv(g, h)
        mirna_feat = h_conv.get('MiRNA')
        mirna_feat = torch.nn.functional.normalize(mirna_feat, p=1, dim=1)
        mirna_feat = torch.tensor(mirna_feat)

        fake_embedding = Variable(copy.deepcopy(mirna_feat[leftIndex:leftIndex + size]))
        M = torch.cat([Adj, fake_embedding], 1)

        output = self.model(M)
        return fake_embedding, output