import torch
import pandas as pd

def get_basic_g(n, k, letters, m='', k_letter=''):
    if m == '':
        m = n - k
        k_letter = list(letters[:n])
    if n == m + 1: return k_letter
    num_of_perm = []
    for perm in get_basic_g(n - 1, k, letters, m, k_letter):
        temp_letter = [i for i in k_letter]
        num_of_perm += [perm + i for i in temp_letter]
    return num_of_perm

def get_k_mer(xulie, k=3):
    basic = ['A', 'C', 'G', 'T']
    basic_group = get_basic_g(len(basic), k, basic)
    base_group_count = list(dict(zip(basic_group, [0 for _ in range(len(basic_group))])) for _ in range(len(xulie)))

    for i in range(len(xulie)):
        for j in range(len(xulie[i]) - k + 1):
            kmer = xulie[i][j:j + k]
            if kmer in base_group_count[i]:
                base_group_count[i][kmer] += 1
            else:
                base_group_count[i][kmer] = 1

    feat_f = [list(base_group_count[i].values()) for i in range(len(base_group_count))]
    feat = []
    for i in range(len(feat_f)):
        sum_f = sum(feat_f[i])
        if sum_f == 0:
            feat.append([0] * len(feat_f[0]))
        else:
            feat.append([round(feat_f[i][n] / sum_f, 4) for n in range(len(feat_f[0]))])
    return feat

def get_feature(mirna_name, file_path):
    df = pd.read_excel(file_path, header=None)
    df.columns = ['miRNA', 'Sequence']

    mirna_name_seq = dict(zip(df['miRNA'], df['Sequence']))

    mirna_xulie = []
    for name in mirna_name:
        if name in mirna_name_seq:
            mirna_xulie.append(mirna_name_seq[name])
        else:
            mirna_xulie.append("")

    k_mer = get_k_mer(mirna_xulie)
    k_mer = torch.tensor(k_mer)

    print("The k_mer feature construction of the node is completed!")
    return k_mer
