import numpy as np

file_path = "./Data/dataset2/dd_delete.txt"
def read_similarity_matrix(file_path):
    # 读取疾病相似度矩阵
    similarity_matrix = np.loadtxt(file_path)
    return similarity_matrix


def get_feature(disease_num, feat_shape):
    # 读取疾病相似度矩阵
    similarity_matrix = read_similarity_matrix(file_path)

    # 确保相似度矩阵的大小与disease_num一致
    assert similarity_matrix.shape[0] == disease_num, "相似度矩阵的大小与疾病数量不一致"

    # 随机生成初始疾病特征
    disease_feat = np.random.rand(disease_num, feat_shape)

    # 将相似度矩阵添加为附加特征
    combined_feat = np.hstack((disease_feat, similarity_matrix))

    return combined_feat