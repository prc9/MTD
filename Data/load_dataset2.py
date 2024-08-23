def dataset2():
    mirna_name = []
    target_name = []
    disease_name = []

    input_mirna_disease = [[], []]
    input_target_disease = [[], []]
    input_mirna_target = [[], []]

    f = open("../Data/dataset2/mirname.txt", "r", encoding="utf-8")
    contents = f.readlines()
    for content in contents: mirna_name.append(content.lower().strip('\n'))
    f.close()

    f = open("../Data/dataset2/genename.txt", "r", encoding="utf-8")
    contents = f.readlines()
    for content in contents: target_name.append(content.lower().strip('\n'))

    f = open("../Data/dataset2/disname.txt", "r", encoding="utf-8")
    contents = f.readlines()
    for content in contents: disease_name.append(content.lower().strip('\n'))

    f = open("../Data/dataset2/md_delete.txt", "r+", encoding="utf-8")
    contents = f.readlines()
    for content_i in range(len(contents)):
        value = contents[content_i].split('\t')
        for i in range(len(value)):
            if int(value[i].strip('\n')) == 1:
                input_mirna_disease[0].append(int(content_i))
                input_mirna_disease[1].append(int(i))
    f.close()

    f = open("../Data/dataset2/mg_delete.txt", "r", encoding="utf-8")
    contents = f.readlines()
    for content_i in range(len(contents)):
        value = contents[content_i].lower().split('\t')
        for i in range(len(value)):
            if int(value[i].strip('\n')) == 1:
                input_mirna_target[0].append(int(content_i))
                input_mirna_target[1].append(int(i))
    f.close()

    f = open("../Data/dataset2/dg_delete.txt", "r", encoding="utf-8")
    contents = f.readlines()
    for content_i in range(len(contents)):
        value = contents[content_i].lower().split('\t')
        for i in range(len(value)):
            if int(value[i].strip('\n')) == 1:
                input_target_disease[1].append(int(content_i))
                input_target_disease[0].append(int(i))
    f.close()

    mirna_num = len(mirna_name)
    target_num = len(target_name)
    disease_num = len(disease_name)

    print("\nDataset2 loading completed!")
    print("mirna num:", mirna_num)
    print("target num:", target_num)
    print("disease num:", disease_num)
    print(mirna_name)
    print(target_name)
    print(disease_name)


    return mirna_num, target_num, disease_num, input_mirna_disease, input_target_disease, input_mirna_target, mirna_name, target_name, disease_name

if __name__ == '__main__':
    dataset2()