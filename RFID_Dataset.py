from torch.utils.data import Dataset


class RFID_Dataset(Dataset):
    """
    数据集
    """

    def __init__(
        self,
        data_path: str,
        transform=None,
    ):
        self.data_path = data_path
        self.transform = transform
        self.data_group = []
        # TODO

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.data_group)
