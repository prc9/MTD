import pandas as pd

def dataset5():
    mirna_name = []
    target_name = []
    disease_name = []

    input_mirna_disease = [[], []]
    input_target_disease = [[], []]
    input_mirna_target = [[], []]

    td = pd.read_excel("../Data/dataset5/ge_dis.xlsx", header=None, names=['target', 'disease'])
    td['target'] = td['target'].str.lower()
    td['disease'] = td['disease'].str.lower()
    for _, row in td.iterrows():
        target = row['target']
        disease = row['disease']
        if target not in target_name:
            target_name.append(target)
        if disease not in disease_name:
            disease_name.append(disease)
        input_target_disease[0].append(target_name.index(target))
        input_target_disease[1].append(disease_name.index(disease))

    mt = pd.read_excel("../Data/dataset5/miRTarBase_MTI.xlsx", header=None, names=['mirna', 'target'])
    mt['mirna'] = mt['mirna'].str.lower()
    mt['target'] = mt['target'].str.lower()
    for _, row in mt.iterrows():
        mirna = row['mirna']
        target = row['target']
        if mirna not in mirna_name:
            mirna_name.append(mirna)
        if target not in target_name:
            target_name.append(target)
        input_mirna_target[0].append(mirna_name.index(mirna))
        input_mirna_target[1].append(target_name.index(target))
    print(mirna_name)

    md = pd.read_excel("../Data/dataset5/mirna_disease.xlsx", header=None, names=['mirna', 'disease'])
    md['mirna'] = md['mirna'].str.lower()
    md['disease'] = md['disease'].str.lower()
    for _, row in md.iterrows():
        mirna = row['mirna']
        disease = row['disease']
        if mirna not in mirna_name:
            mirna_name.append(mirna)
        if disease not in disease_name:
            disease_name.append(disease)
        input_mirna_disease[0].append(mirna_name.index(mirna))
        input_mirna_disease[1].append(disease_name.index(disease))
    print(mirna_name)

    mirna_num = len(mirna_name)
    target_num = len(target_name)
    disease_num = len(disease_name)

    print("\nDataset5 loading completed!")



    return mirna_num, target_num, disease_num, input_mirna_disease, input_target_disease, input_mirna_target, mirna_name, target_name, disease_name

if __name__ == '__main__':
    dataset5()
