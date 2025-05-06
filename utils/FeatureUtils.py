from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_pca_model(datas, target: int = 2):
    """
    获取PCA模型
    """

    # 展平数据
    flattened_data = datas.reshape(datas.shape[0], -1)

    reducer = PCA(n_components=target)

    reducer = reducer.fit(flattened_data)
    return reducer


def use_pca_model(model, datas):
    """
    使用PCA模型进行降维
    """

    # 展平数据
    flattened_data = datas.reshape(datas.shape[0], -1)

    reduced_data = model.transform(flattened_data)

    return reduced_data


def pca_reduce(datas, target: int = 2):
    """
    PCA降维
    """

    # 展平数据
    flattened_data = datas.reshape(datas.shape[0], -1)

    reducer = PCA(n_components=target)

    # 降维
    reduced_data = reducer.fit_transform(flattened_data)
    return reduced_data


def tsne_reduce(datas, target: int = 2, random_state=42):
    """
    TSNE降维
    """

    # 展平数据
    flattened_data = datas.reshape(datas.shape[0], -1)

    reducer = TSNE(n_components=target, random_state=random_state)

    # 降维
    reduced_data = reducer.fit_transform(flattened_data)
    return reduced_data
