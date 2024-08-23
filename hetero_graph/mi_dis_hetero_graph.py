import dgl
import torch

def get_hetero_graph(input_net, true_input_net, mirna_target, target_disease, mirna_feat, target_feat, disease_feat):
    Noise_MTDN_data = {
        ('MiRNA', 'MvsD', 'Disease'): (torch.tensor(input_net[0]), torch.tensor(input_net[1])),
        ('Disease', 'DvsM', 'MiRNA'): (torch.tensor(input_net[1]), torch.tensor(input_net[0])),
        ('MiRNA', 'MvsT', 'Target'): (torch.tensor(mirna_target[0]), torch.tensor(mirna_target[1])),
        ('Target', 'TvsM', 'MiRNA'): (torch.tensor(mirna_target[1]), torch.tensor(mirna_target[0])),
        ('Target', 'TvsD', 'Disease'): (torch.tensor(target_disease[0]), torch.tensor(target_disease[1])),
        ('Disease', 'DvsT', 'Target'): (torch.tensor(target_disease[1]), torch.tensor(target_disease[0]))
    }

    Noise_MTDN = dgl.heterograph(Noise_MTDN_data)
    Noise_MTDN.nodes['MiRNA'].data['feat'] = mirna_feat
    Noise_MTDN.nodes['Target'].data['feat'] = target_feat
    Noise_MTDN.nodes['Disease'].data['feat'] = disease_feat
    Noise_MTDN_h = {'MiRNA': Noise_MTDN.nodes['MiRNA'].data['feat'],
                    'Target': Noise_MTDN.nodes['Target'].data['feat'],
                    'Disease': Noise_MTDN.nodes['Disease'].data['feat']}

    True_MTDN_data = {
        ('MiRNA', 'MvsD', 'Disease'): (torch.tensor(true_input_net[0]), torch.tensor(true_input_net[1])),
        ('Disease', 'DvsM', 'MiRNA'): (torch.tensor(true_input_net[1]), torch.tensor(true_input_net[0])),
        ('MiRNA', 'MvsT', 'Target'): (torch.tensor(mirna_target[0]), torch.tensor(mirna_target[1])),
        ('Target', 'TvsM', 'MiRNA'): (torch.tensor(mirna_target[1]), torch.tensor(mirna_target[0])),
        ('Target', 'TvsD', 'Disease'): (torch.tensor(target_disease[0]), torch.tensor(target_disease[1])),
        ('Disease', 'DvsT', 'Target'): (torch.tensor(target_disease[1]), torch.tensor(target_disease[0]))
    }

    True_MTDN = dgl.heterograph(True_MTDN_data)
    True_MTDN.nodes['MiRNA'].data['feat'] = mirna_feat
    True_MTDN.nodes['Target'].data['feat'] = target_feat
    True_MTDN.nodes['Disease'].data['feat'] = disease_feat
    True_MTDN_h = {'MiRNA': True_MTDN.nodes['MiRNA'].data['feat'],
                   'Target': True_MTDN.nodes['Target'].data['feat'],
                   'Disease': True_MTDN.nodes['Disease'].data['feat']}

    return Noise_MTDN, Noise_MTDN_h, True_MTDN, True_MTDN_h
