import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = dglnn.HeteroGraphConv({
            'MvsT': dglnn.GATConv(in_dim, hidden_dim, num_heads),
            'TvsM': dglnn.GATConv(in_dim, hidden_dim, num_heads),
            'MvsD': dglnn.GATConv(in_dim, hidden_dim, num_heads),
            'DvsM': dglnn.GATConv(in_dim, hidden_dim, num_heads),
            'TvsD': dglnn.GATConv(in_dim, hidden_dim, num_heads),
            'DvsT': dglnn.GATConv(in_dim, hidden_dim, num_heads)
        }, aggregate='sum')

        self.layer2 = dglnn.HeteroGraphConv({
            'MvsT': dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads),
            'TvsM': dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads),
            'MvsD': dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads),
            'DvsM': dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads),
            'TvsD': dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads),
            'DvsT': dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
        }, aggregate='sum')

        self.layer3 = dglnn.HeteroGraphConv({
            'MvsT': dglnn.GATConv(hidden_dim * num_heads, out_dim, 1),
            'TvsM': dglnn.GATConv(hidden_dim * num_heads, out_dim, 1),
            'MvsD': dglnn.GATConv(hidden_dim * num_heads, out_dim, 1),
            'DvsM': dglnn.GATConv(hidden_dim * num_heads, out_dim, 1),
            'TvsD': dglnn.GATConv(hidden_dim * num_heads, out_dim, 1),
            'DvsT': dglnn.GATConv(hidden_dim * num_heads, out_dim, 1)
        }, aggregate='sum')

    def forward(self, g, inputs):
        h = self.layer1(g, inputs)
        h = {k: torch.nn.functional.elu(v.flatten(1)) for k, v in h.items()}

        h = self.layer2(g, h)
        h = {k: torch.nn.functional.elu(v.flatten(1)) for k, v in h.items()}

        h = self.layer3(g, h)
        h = {k: v.mean(1) for k, v in h.items()}  # 平均注意力头输出

        return h


