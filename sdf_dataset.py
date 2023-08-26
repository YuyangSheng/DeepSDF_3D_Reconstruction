import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_data(rootpath):
    point_path = os.path.join(rootpath, "noisy_points")
    sdf_path = os.path.join(rootpath, "sdf")

    xyz_filename = sorted(os.listdir(point_path))
    sdf_filename = sorted(os.listdir(sdf_path))
    
    xyz_files = [os.path.join(point_path, file) for file in xyz_filename]
    sdf_files = [os.path.join(sdf_path, file) for file in sdf_filename]

    assert len(xyz_files) == len(sdf_files)
    print(f"loading {len(xyz_files)} models")

    data = []
    for i in range(len(xyz_files)):
        # load data
        xyz = torch.from_numpy(np.load(xyz_files[i]))
        sdf = torch.from_numpy(np.load(sdf_files[i]))
        _data = torch.cat((xyz, sdf.unsqueeze(1)), dim=1)

        # remove NaN
        sdf_nan = torch.isnan(sdf)
        _data = _data[~sdf_nan, :]

        data.append(_data)

    return data

def _train_test_split(data, train_ratio):

    indices = list(range(len(data)))
    random.seed(4)
    random.shuffle(data)
    random.shuffle(indices)
    train_size = int(len(data) * train_ratio)
    print("the order of loaded data:", indices)

    return data[:train_size], data[train_size:], indices[:train_size], indices[train_size:]

def get_dataset(root, num_subsample, train_ratio, mod, split_pos_neg):
    data = _load_data(root)
    train_data, test_data, train_indices, test_indices = _train_test_split(data, train_ratio)
    if mod == "train":
      train_dataset = sdfData(train_data, num_subsample, split_pos_neg)

      return train_dataset, train_indices

    elif mod == "test":
      test_dataset = sdfData(test_data, num_subsample, split_pos_neg)

      return test_dataset, test_indices

    else:
      train_dataset = sdfData(train_data, num_subsample, split_pos_neg)
      test_dataset = sdfData(test_data, num_subsample, split_pos_neg)

    return train_dataset, test_dataset, train_indices, test_indices


class sdfData(Dataset):
    """
    Arguments:
    rootpath: "./train" or "./val"
    """
    
    def __init__(self, data, subsample, split_pos_neg):

        super(sdfData, self).__init__()
        self.subsample = subsample
        self.data = data
        self.split_pos_neg = split_pos_neg
        self.split_data = []

        if split_pos_neg:
            for i in range(len(data)):
                # find positive and negative sdf
                piece_data = data[i]
                pos_data = piece_data[piece_data[:,3]>=0, :]
                neg_data = piece_data[piece_data[:,3]<0, :]
                self.split_data.append([pos_data[torch.randperm(pos_data.shape[0])],
                            neg_data[torch.randperm(neg_data.shape[0])]])
            
    def __getitem__(self, index):
        if self.split_pos_neg:
            pos_data = self.split_data[index][0]
            neg_data = self.split_data[index][1]

            pos_size = pos_data.shape[0]
            neg_size = neg_data.shape[0]

            max_size = max(pos_size, neg_size)
            min_size = min(pos_size, neg_size)
            data_size = self.subsample//2

            if min_size > data_size:
                pos_sample = pos_data[: data_size]
                neg_sample = neg_data[: data_size]

                samples_remain = torch.cat([pos_data[data_size :], neg_data[data_size :]], 0)
            else:
                if pos_size < neg_size:
                    pos_sample = pos_data
                    neg_sample = neg_data[: self.subsample-pos_size]

                    samples_remain = neg_data[self.subsample-pos_size :]
                else:
                    pos_sample = pos_data[: self.subsample-neg_size]
                    neg_sample = neg_data

                    samples_remain = pos_data[self.subsample-pos_size :]

            samples = torch.cat([pos_sample, neg_sample], 0)
        else:
            all_samples = self.data[index]
            all_samples = all_samples[torch.randperm(samples.shape[0])]
            samples = all_samples[:self.subsample]
            samples_remain = all_samples[self.subsample :]

        return samples, samples_remain, index

    def __len__(self):

        return len(self.data)


