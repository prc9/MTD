def dataset1():
    mirna_name = []
    target_name = []
    disease_name = []

    mirna_disease = []
    mirna_target = []
    target_disease = []

    # f = open("../Data/miRNAlist.txt", 'r', encoding='utf-8')
    # contents = f.readlines()
    # for content in contents:mirna_name.append(content.lower().strip('\n'))
    # print(mirna_name)
    # f.close()

    f = open("../Data/dataset1/diseaseList.txt", 'r', encoding='utf-8')
    contents = f.readlines()
    for content in contents:
        value = content.split('\t')
        value[0] = value[0].lower().strip()
        if value[0] not in disease_name : disease_name.append(value[0])
    print(disease_name)
    f.close()

    f = open("../Data/dataset1/miRNA-target.txt", 'r', encoding='utf-8')
    contents = f.readlines()
    for content in contents:
        value = content.split('\t')
        value[0] = value[0].lower().strip()
        if value[0] not in mirna_name: mirna_name.append(value[0])
        value[1] = value[1].lower().strip('\n')
        if value[1] not in target_name: target_name.append(value[1])
        mirna_target.append(value)
    f.close()

    f = open("../Data/dataset1/miRNA-disease.txt", 'r', encoding='utf-8')
    contents = f.readlines()
    for content in contents:
        value = content.split('\t')
        value[0] = value[0].lower().strip()
        if value[0] not in mirna_name: mirna_name.append(value[0])
        value[1] = value[1].lower().strip('\n')
        if value[1] not in disease_name : disease_name.append(value[1])
        mirna_disease.append(value)
    print(disease_name)
    f.close()

    f = open("../Data/dataset1/target-disease.txt", 'r', encoding='utf-8')
    contents = f.readlines()
    for content in contents:
        value = content.split('\t')
        value[0] = value[0].lower().strip()
        if value[0] not in target_name: target_name.append(value[0])
        value[1] = value[1].lower().strip('\n')
        if value[1] not in disease_name: disease_name.append(value[1])
        target_disease.append(value)
    f.close()

    mirna_num = len(mirna_name)
    target_num = len(target_name)
    disease_num = len(disease_name)

    mirna_index = dict(zip(mirna_name, range(0, mirna_num)))
    target_index = dict(zip(target_name, range(0, target_num)))
    disease_index = dict(zip(disease_name, range(0, disease_num)))

    input_mirna_disease = [[],[]]
    for i in range(len(mirna_disease)):
        input_mirna_disease[0].append(mirna_index.get(mirna_disease[i][0]))
        input_mirna_disease[1].append(disease_index.get(mirna_disease[i][1]))

    input_mirna_target = [[],[]]
    for i in range(len(mirna_target)):
        input_mirna_target[0].append(mirna_index.get(mirna_target[i][0]))
        input_mirna_target[1].append(target_index.get(mirna_target[i][1]))

    input_target_disease = [[],[]]
    for i in range(len(target_disease)):
        input_target_disease[0].append((target_index.get(target_disease[i][0])))
        input_target_disease[1].append(disease_index.get(target_disease[i][1]))

    print("\nDataset loading completed!")
    print("mirna num:", mirna_num)
    print("target num:", target_num)
    print("disease num:", disease_num)

    print(mirna_name)
    print(target_name)
    return mirna_num, target_num, disease_num, input_mirna_disease, input_target_disease,input_mirna_target, mirna_name, target_name, disease_name



if __name__ == '__main__':
    dataset1()

