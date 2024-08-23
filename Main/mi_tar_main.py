import torch
import random
import warnings
import numpy as np
import get_miRNA_feature
from Model import mi_tar_model
from Model import simple_mi_dis_model
from Train import mi_tar_train
from sklearn.model_selection import KFold
from Data.load_dataset1 import dataset1
from Data.load_dataset2 import dataset2
from Data.load_dataset3 import dataset3
from hetero_graph.mi_tar_hetero_graph import get_hetero_graph
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    epochs = 100
    pro_ZR = 50
    pro_PM = 50
    alpha = 0.1

    feat_shape = 64
    out_feat = 64

    G_step = 5
    D_step = 2
    batchSize = 32

    G_PATH = '../weights/G.pth'
    D_PATH = '../weights/D.pth'

    AUC = []

    # mirna_num, target_num, disease_num, mirna_disease, target_disease, mirna_target, mirna_name, target_name, disease_name = dataset1()
    # mirna_num, target_num, disease_num, mirna_disease, target_disease, mirna_target, mirna_name, target_name, disease_name = dataset2()
    mirna_num, target_num, disease_num, mirna_disease, target_disease, mirna_target, mirna_name, target_name, disease_name = dataset3()

    mirna_feat = torch.rand(mirna_num, feat_shape)
    # mirna_feat = get_miRNA_feature.get_feature(mirna_name, '../Data/miRNA_sequence.xlsx')
    target_feat = torch.rand(target_num, feat_shape)
    disease_feat = torch.rand(disease_num, feat_shape)

    print("\nData Details:")
    print("miRNA:{}  target:{}  disease:{}".format(mirna_num, target_num, disease_num))
    print("miRNA-disease:{}  target-disease:{}  miRNA-target:{}".format(len(mirna_disease[0]), len(target_disease[0]),
                                                                        len(mirna_target[0])))
    print("Sparsity of miRNA-disease associated data: {:.6f}\n".format(
        len(mirna_target[0]) / (mirna_num * target_num)))

    edge_data = []
    for i in range(len(mirna_target[0])):
        x = []
        x.append(mirna_target[0][i])
        x.append(mirna_target[1][i])
        edge_data.append(x)
    edge_data = np.array(edge_data)

    negativeSample_edge = []
    for i in range(len(edge_data)):
        row = random.randint(0, mirna_num - 1)
        col = random.randint(0, target_num - 1)
        while ([row, col] in edge_data.tolist() or [row, col] in negativeSample_edge):
            row = random.randint(0, mirna_num - 1)
            col = random.randint(0, target_num - 1)
        negativeSample_edge.append([row, col])

    kf = KFold(n_splits=2, shuffle=True)
    for train_index, test_index in kf.split(edge_data):
        train_negative = random.sample(negativeSample_edge, int(len(train_index)))
        test_negative = [data_negative for data_negative in negativeSample_edge if data_negative not in train_negative]
        train_index = train_index.tolist()
        train_mirna_20 = random.sample(train_index, int(len(train_index) * 0.25))
        train_mirna_60 = [i for i in train_index if i not in train_mirna_20]
        test_mirna = test_index.tolist()

        input_net = [[], []]
        for i in train_mirna_60:
            input_net[0].append(edge_data[i][0])
            input_net[1].append(edge_data[i][1])
        for i in range(len(train_negative)):
            input_net[0].append(train_negative[i][0])
            input_net[1].append(train_negative[i][1])

        true_input_net = [[], []]
        for i in train_index:
            true_input_net[0].append(edge_data[i][0])
            true_input_net[1].append(edge_data[i][1])

        test_input_net = [[], []]
        for i in test_mirna:
            test_input_net[0].append(edge_data[i][0])
            test_input_net[1].append(edge_data[i][1])
        for i in range(len(test_negative)):
            test_input_net[0].append(test_negative[i][0])
            test_input_net[1].append(test_negative[i][1])

        G = mi_tar_model.generator(target_num,feat_shape,out_feat)
        D = mi_tar_model.discriminator(target_num,feat_shape,out_feat)
        # G = simple_mi_dis_model.SimpleGenerator(disease_num,feat_shape,out_feat)
        # D = simple_mi_dis_model.SimpleDiscriminator(disease_num,feat_shape,out_feat)
        Noise_MTDN, Noise_MTDN_h, True_MTDN, True_MTDN_h = get_hetero_graph(input_net, true_input_net, mirna_disease,
                                                                            target_disease, mirna_feat, target_feat,
                                                                            disease_feat)

        auc = mi_tar_train.main(mirna_num, target_num, epochs, pro_ZR, pro_PM, alpha, batchSize,
                                input_net, true_input_net, test_input_net, test_negative,
                                Noise_MTDN, Noise_MTDN_h, True_MTDN, True_MTDN_h,
                                G, D, G_step, D_step, G_PATH, D_PATH)

        AUC.append(auc)
        print('\n2_fold_auc:{}'.format(AUC))
        print('Mean auc:{}\n'.format(sum(AUC) / len(AUC)))