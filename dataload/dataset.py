from torch.utils.data import Dataset
import cv2
import torch


class CustomDataset(Dataset):
    def __init__(self, dataframe_set, src_dir, transform=None, mode='train'):
        self.images_path = dataframe_set['file_name'].values

        if mode != 'inf':
            self.label_pair = dataframe_set['label'].values
            self.label_n_unique = len(dataframe_set['label'].unique())

        self.src_dir = src_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        input_path = self.images_path[idx]
        input_value = self._data_generation(input_path)

        if self.mode != 'inf':
            labels = torch.zeros(self.label_n_unique)
            tmp = self.label_pair[idx]
            labels[tmp] = 1
            return input_value, labels
        else:
            return input_value

    def _data_generation(self, input_path):

        input_path = self.src_dir + input_path
        input_value = cv2.imread(input_path)

        input_value = self.transform(image=input_value)
        return input_value['image']
