from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_dimension(datas, target: int = 2, method="pca"):
    """
    降维
    """

    # 展平数据
    flattened_data = datas.reshape(datas.shape[0], -1)

    # 选择降维方法
    method = method.lower()
    if method == "pca":
        reducer = PCA(n_components=target)
    elif method == "tsne":
        reducer = TSNE(n_components=target, random_state=42)
    else:
        raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")

    # 降维
    reduced_data = reducer.fit_transform(flattened_data)
    return reduced_data
