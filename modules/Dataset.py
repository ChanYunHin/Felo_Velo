import numpy as np
import torch
# from torch._C import dtype
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import pdb
from parameters import args 
# import matplotlib.pyplot as plt
# plt.switch_backend('nbAgg')


class BatchGenerator:
    def __init__(self, x, yy):
        self.x = x
        self.y = yy
        self.size = len(x)
        self.random_order = list(range(len(x)))
        np.random.shuffle(self.random_order)
        self.start = 0

    def get_dataset_size(self):
        return self.size

    def next_test_batch(self, batch_size):
        if self.start + batch_size >= len(self.random_order):
            overflow = (self.start + batch_size) - len(self.random_order)
            perm0 = self.random_order[self.start:] +\
                 self.random_order[:overflow]
            # self.start = overflow
            # retraining this dataset
            np.random.shuffle(self.random_order)
            self.start = 0
        else:
            perm0 = self.random_order[self.start:self.start + batch_size]
            self.start += batch_size

        assert len(perm0) == batch_size

        return torch.tensor(self.x[perm0]), torch.tensor(self.y[perm0], dtype=torch.long)


    def next_batch(self, batch_size):
        # np.random.shuffle(self.random_order)
        # perm0 = self.random_order[self.start:self.start + batch_size]
        # return torch.tensor(self.x[perm0]), torch.tensor(self.y[perm0], dtype=torch.long), perm0


        if self.start + batch_size >= len(self.random_order):
            overflow = (self.start + batch_size) - len(self.random_order)
            perm0 = self.random_order[self.start:] +\
                 self.random_order[:overflow]
            # self.start = overflow
            # retraining this dataset
            np.random.shuffle(self.random_order)
            self.start = 0
        else:
            perm0 = self.random_order[self.start:self.start + batch_size]
            self.start += batch_size

        assert len(perm0) == batch_size
        
        return torch.tensor(self.x[perm0]), torch.tensor(self.y[perm0], dtype=torch.long), perm0

    # support slice
    def __getitem__(self, val):
        return self.x[val], self.y[val]


class Dataset(object):
    def __init__(self, dataset_name, one_hot=True, split=0, distorted_data=False):
        
        if dataset_name == "cifar10":

            transform_train = transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])


            self.train_data = datasets.CIFAR10(
                root = "data",
                train = True,
                download = True,
                transform = transform_train
            )
            self.test_data = datasets.CIFAR10(
                root = "data",
                train = False,
                download = True,
                transform = transform_test
            )
        
        x_train, y_train, x_test, y_test = self.seperate_data(self.train_data, self.test_data)
        print("Dataset: train-%d, test-%d" % (len(x_train), len(x_test)))


        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')


        if split == 0:
            self.train = BatchGenerator(x_train, y_train)
        else:
            if args.noniid_flag:
                self.train = self.noniid_dirichlet(x_train, y_train, split, 1)
            else:
                self.train = self.splited_batch(x_train, y_train, split)

        self.test = BatchGenerator(x_test, y_test)



    def seperate_data(self, train_data, test_data):
        x_train, y_train, x_test, y_test = [], [], [], []

        for data_point in train_data:
            x_train.append(data_point[0].numpy())
            y_train.append(data_point[1])

        for data_point in test_data:
            x_test.append(data_point[0].numpy())
            y_test.append(data_point[1])

        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    

    def noniid_dirichlet(self, x_data, y_data, client_num, alpha=10):

        # classify data by labels
        res = []
        classes_data_index = {}
        tmp_classes_data_index = {}
        classes_num = y_data.shape[1]
        for label in range(classes_num):
            tmp_classes_data_index[label] = (np.argmax(y_data, axis=1) == label)
        for label, data_index in tmp_classes_data_index.items():
            classes_data_index[label] = []
            for idx, val in enumerate(data_index):
                if val == True:
                    classes_data_index[label].append(idx)

        # shuffle the data_index
        tmp_classes_data_index = {}
        for label, data_idx in classes_data_index.items():
            np.random.shuffle(classes_data_index[label])

        # produce a non-iid dataset followed dirichlet distribution
        np.random.seed(1)
        sample_ratio = np.random.dirichlet(np.repeat(alpha, client_num), classes_num)
        for i in range(client_num):
            selected_x_index_list = []
            for label in range(classes_num):
                data_num = int(len(classes_data_index[label]) * sample_ratio[i, label])
                begin_idx = int(len(classes_data_index[label]) * sum(sample_ratio[0:i, label]))
                selected_x_index_list.extend(classes_data_index[label][begin_idx: begin_idx+data_num])
            res.append(
                BatchGenerator(x_data[selected_x_index_list], y_data[selected_x_index_list])
            )


        return res

    def splited_batch(self, x_data, y_data, count):
        res = []
        l = len(x_data)
        if args.test:
            if args.data_size_fold == 1:
                for i in range(0, l, l//count):
                    res.append(
                        BatchGenerator(x_data[i:i + l // count],
                                    y_data[i:i + l // count]))
            else:
                for i in range(count):
                    random_order = list(range(len(x_data)))
                    np.random.shuffle(random_order)
                    res.append(
                        BatchGenerator(x_data[random_order[:(l // count) * args.data_size_fold]],
                                    y_data[random_order[:(l // count) * args.data_size_fold]])
                    )
        else:

                for i in range(0, l, l//count):
                    res.append(
                        BatchGenerator(x_data[i:i + l // count],
                                    y_data[i:i + l // count]))
        return res

        

