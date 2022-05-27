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
        elif dataset_name == "cifar100":
            stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
            transform_train = transforms.Compose([
                # transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*stats),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*stats),
            ])


            self.train_data = datasets.CIFAR100(
                root = "data",
                train = True,
                download = True,
                transform = transform_train
            )
            self.test_data = datasets.CIFAR100(
                root = "data",
                train = False,
                download = True,
                transform = transform_test
            )
        
        elif dataset_name == "emnist_bl": 
            
            transform_format = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(32),
            ])
            
            self.train_data = datasets.EMNIST (
                root = "data",
                train = True,
                download = True,
                transform = transform_format,
                split = "balanced"
            )
            self.test_data = datasets.EMNIST(
                root = "data",
                train = False,
                download = True,
                transform = transform_format,
                split = "balanced"
            )

        elif dataset_name == "Fmnist": 
            
            transform_format = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(32),
            ])
            
            self.train_data = datasets.FashionMNIST (
                root = "data",
                train = True,
                download = True,
                transform = transform_format
            )
            self.test_data = datasets.FashionMNIST(
                root = "data",
                train = False,
                download = True,
                transform = transform_format
            )

        else:
            transform_format = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(32),
            ])
            
            self.train_data = datasets.MNIST (
                root = "data",
                train = True,
                download = True,
                transform = transform_format
            )
            self.test_data = datasets.MNIST(
                root = "data",
                train = False,
                download = True,
                transform = transform_format
            )

        x_train, y_train, x_test, y_test = self.seperate_data(self.train_data, self.test_data)
        print("Dataset: train-%d, test-%d" % (len(x_train), len(x_test)))

        # if one_hot:
        #     y_train = F.one_hot(torch.as_tensor(y_train), 10)
        #     y_test = F.one_hot(torch.as_tensor(y_test), 10)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # pdb.set_trace()
        # if dataset_name == "mnist" or dataset_name == "emnist":
        #     x_train = np.expand_dims(x_train, -1)
        #     x_test = np.expand_dims(x_test, -1)

        
        # if not distorted_data:
        #     x_train /= 255.0
        #     x_test /= 255.0
        #     mean_image = np.mean(x_train, axis=0)
        #     x_train -= mean_image
        #     x_test -= mean_image

        # if distorted_data:
        #     # plt.imsave('before_crop.png', x_train[0])
        #     x_train = tf.image.central_crop(x_train, 0.75)
        #     x_train = np.array(x_train)
        #     # plt.imsave('crop.png', x_train[0])
        #     x_test = tf.image.central_crop(x_test, 0.75)
        #     x_test = np.array(x_test)

        # divided_y_list = []
        # if args.noniid_flag:
        #     class_num_each_clients = 2
        #     split_y_list = list(range(split))
        #     np.random.shuffle(split_y_list)
        #     # divided_y_parts_num = split // class_num_each_clients
        #     for i in range(split):
        #         if (i+1) * class_num_each_clients % split == 0:
        #             divided_y_list.append(
        #                 split_y_list[(i*class_num_each_clients) % split:]
        #             )
        #         else:
        #             divided_y_list.append(
        #                 split_y_list[(i*class_num_each_clients) % split: (i+1)*class_num_each_clients % split]
        #             )

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

    # def split_noniid(self, x_data, y_data, client_num, divided_y_list):
    #     # get the non-iid dataset, divided by labels
    #     # return the non-iid dataset. In this case, each client will just have data from two labels.
    #     res = []
    #     classes_data_index = {}
    #     tmp_classes_data_index = {}
    #     classes_num = y_data.shape[1]
    #     for label in range(classes_num):
    #         tmp_classes_data_index[label] = (np.argmax(y_data, axis=1) == label)
    #     for label, data_index in tmp_classes_data_index.items():
    #         classes_data_index[label] = []
    #         for idx, val in enumerate(data_index):
    #             if val == True:
    #                 classes_data_index[label].append(idx)

    #     # classes_x_data = {}
    #     # for label, index in classes_data_index.items():
    #     #     classes_x_data[label] = x_data[index]

    #     label_num_each_clients = len(divided_y_list[0])
    #     selected_fraction = 0.5
    #     client_data_index = {}
    #     client_data = {}


    #     for i in range(client_num):
    #         labels = divided_y_list[i]
    #         selected_x_index_list = []
    #         for label in labels:
    #             data_index = classes_data_index[label]
    #             random_data_index = np.random.permutation(data_index)
    #             selected_len = int(len(random_data_index) * selected_fraction)
    #             selected_x_index_list.extend(random_data_index[:selected_len])
    #         res.append(
    #             BatchGenerator(x_data[selected_x_index_list], y_data[selected_x_index_list])
    #         )
            
    #     # for label, x_data in classes_x_data.items():
    #     #     len_x = x_data.shape[0]
    #     #     random_list = np.random.permutation(list(range(len_x)))
    #     #     client_data[label] = 
    #     return res


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
            if args.dataset == "cifar100":
                for i in range(count):
                    random_order = list(range(len(x_data)))
                    np.random.shuffle(random_order)
                    res.append(
                        BatchGenerator(x_data[random_order[:(l // count) * 5]],
                                    y_data[random_order[:(l // count) * 5]])
                    )
            else:
                # for i in range(count):
                #     random_order = list(range(len(x_data)))
                #     np.random.shuffle(random_order)
                #     res.append(
                #         BatchGenerator(x_data[random_order[:(l // count) * 1]],
                #                     y_data[random_order[:(l // count) * 1]])
                #     )
                for i in range(0, l, l//count):
                    res.append(
                        BatchGenerator(x_data[i:i + l // count],
                                    y_data[i:i + l // count]))
        return res


class Dataset_MD(Dataset):
    def __init__(self, dataset_name, one_hot=True, split=10, distorted_data=False):
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
        elif dataset_name == "cifar100":
            stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
            transform_train = transforms.Compose([
                # transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*stats),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*stats),
            ])


            self.train_data = datasets.CIFAR100(
                root = "data",
                train = True,
                download = True,
                transform = transform_train
            )
            self.test_data = datasets.CIFAR100(
                root = "data",
                train = False,
                download = True,
                transform = transform_test
            )


        elif dataset_name == "emnist_bl": 
            
            transform_format = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(32),
            ])
            
            self.train_data = datasets.EMNIST (
                root = "data",
                train = True,
                download = True,
                transform = transform_format,
                split = "balanced"
            )
            self.test_data = datasets.EMNIST(
                root = "data",
                train = False,
                download = True,
                transform = transform_format,
                split = "balanced"
            )

        elif dataset_name == "Fmnist": 
            
            transform_format = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(32),
            ])
            
            self.train_data = datasets.FashionMNIST (
                root = "data",
                train = True,
                download = True,
                transform = transform_format
            )
            self.test_data = datasets.FashionMNIST(
                root = "data",
                train = False,
                download = True,
                transform = transform_format
            )

        else:
            self.train_data = datasets.MNIST (
                root = "data",
                train = True,
                download = True,
                transform = transforms.ToTensor()
            )
            self.test_data = datasets.MNIST(
                root = "data",
                train = False,
                download = True,
                transform = transforms.ToTensor()
            )
        
        public_dataset, private_dataset = \
            self.get_public_private_dataset(self.train_data)
        
        public_x, public_y = self.seperate_data(public_dataset)
        private_x_train, private_y_train = self.seperate_data(private_dataset)
        x_test, y_test = self.seperate_data(self.test_data)
        
        public_x, public_y = \
            public_x.astype('float32'), public_y.astype('float32')
        private_x_train, private_y_train = \
            private_x_train.astype('float32'), private_y_train.astype('float32')
        
        
        self.public_dataset = BatchGenerator(public_x, public_y)
        
        self.train = self.splited_batch(private_x_train, private_y_train, split)

        self.test = BatchGenerator(x_test, y_test)
    
    
    def seperate_data(self, target_dataset):
        x_data, y_data= [], []

        for data_point in target_dataset:
            x_data.append(data_point[0].numpy())
            y_data.append(data_point[1])

        return np.array(x_data), np.array(y_data)
    
    def get_public_private_dataset(self, train_data):
        public_dataset = []
        private_dataset = []
        train_dataset_size = len(train_data)
        public_dataset_size = int(5000)
        private_dataset_size = int(train_dataset_size - 5000)
        for public_idx in range(public_dataset_size):
            public_dataset.append(train_data[public_idx])
        for private_idx in range(private_dataset_size):
            private_dataset.append(train_data[public_dataset_size + private_idx])
        
        return public_dataset, private_dataset
        

