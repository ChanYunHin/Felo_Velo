
import math
import numpy as np
import copy
from collections import namedtuple
from numpy.lib.function_base import gradient

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib
from parameters import args 
if not args.old_CNN:
    from modules.Model import ClientModel
    from modules.Model import ClientModel_2CNN
else:
    from modules.old_Model import ClientModel
    from modules.old_Model import ClientModel_2CNN
from modules.ResnetModel import Resnet50, SmallResnet50, SmallResnet1, SmallResnet2, SmallResnet3
from modules.HeteroFL.models.resnet import resnet50
from modules.HeteroFL.data import fetch_dataset, make_data_loader, split_dataset, SplitDataset
from modules.HeteroFL.metrics import Metric
from modules.HeteroFL.utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from modules.HeteroFL.config import cfg
from modules.HeteroFL import models


# # The definition of fed model
# FedModel = namedtuple('FedModel', 'X Y DROP_RATE train_op loss_op acc_op')



class Local:
    def __init__(self, model_rate, data_loader, label_split, device):
        self.model_rate = model_rate
        self.data_loader = data_loader
        self.label_split = label_split
        self.device = device

    def train(self, local_parameters, lr):
        metric = Metric()
        model = eval('models.{}(model_rate=self.model_rate).to(self.device)'.format(cfg['model_name']))
        model.load_state_dict(local_parameters)
        model.train(True)
        optimizer = make_optimizer(model, lr)
        for local_epoch in range(1, cfg['num_epochs']['local'] + 1):
            for i, input in enumerate(self.data_loader):
                # input = collate(input)
                # input_size = input['img'].size(0)
                # input['label_split'] = torch.tensor(self.label_split)
                ''' input[0] = x, input[1] = y '''
                input = to_device(input, self.device)
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(cfg['metric_name']['train']['Local'], input, output)
                # logger.append(evaluation, 'train', n=input_size)
        local_parameters = model.state_dict()
        return local_parameters, evaluation



class Client:
    def __init__(self,
                 input_shape,
                 num_classes,
                 learning_rate,
                 client_id,
                 batch_size,
                 clients_training_epoch,
                 device,
                 dataset,
                 federation,
                 data_split,
                 label_split,
                 is_worker=True):
        # self.graph = tf.Graph()
        # self.sess = tf.Session(graph=self.graph)

        self.input_shape = input_shape
        self.device = device

        # Call the create function to build the computational graph

        # # If you don't have this function, you would not have true weights in the network,
        # # because you only get your weights when data pass through the network.
        # self.create_weights_for_model(self.model)

        # self.model.to(self.device)
        self.lr = learning_rate
        
        # total_epoches = args.epoch * args.clients_sample_ratio * args.clients_number * args.clients_training_epoch
        
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
        #                                                       milestones=[int(total_epoches * 0.15),
        #                                                                   int(total_epoches * 0.25),
        #                                                                   int(total_epoches * 0.40),
        #                                                                   int(total_epoches * 0.55),
        #                                                                   int(total_epoches * 0.70),
        #                                                                   int(total_epoches * 0.85),
        #                                                                   int(total_epoches * 0.95)],
        #                                                       gamma=0.5)
        
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
        #                                                             factor=0.5, patience=1,
        #                                                             threshold=0.01)
        
        self.loss_func = F.cross_entropy
        # can not use cross-entropy loss in the distillation.
        # because cross-entropy didn't consider the information from other classes.
        if args.KLD_flag:
            self.distill_loss_func_logits = F.kl_div
            self.distill_loss_func_features = F.mse_loss
            # self.distill_loss = tf.keras.metrics.KLDivergence()
        else:
            self.distill_loss_func_features = F.mse_loss
            self.distill_loss_func_logits = self.distill_loss_func_features
            # self.distill_loss = tf.keras.metrics.MeanSquaredError()
        # self.accuracy = tf.keras.metrics.CategoricalAccuracy(name='accuracy_{}'.format(client_id))
        self.mean_loss = []
        self.accuracy = []
        self.distill_loss, self.feature_loss = [], []
        self.bn_mean_loss = []
        self.bn_var_loss = []
        # tf.keras.metrics.CategoricalCrossentropy(from_logits=True,
        #                                                           name="mean_loss_{}".format(client_id))

        # initialize
        self.num_classes = num_classes
        self.cid = client_id
        self.is_worker = is_worker
        self.batch_size = batch_size
        self.clients_training_epoch = clients_training_epoch
        
        tmp_local_parameters, self.param_idx = federation.distribute([client_id])
        self.local_parameters = tmp_local_parameters[0]
        
        model_rate_m = federation.model_rate[client_id]
        
        data_loader_m = make_data_loader({'train': SplitDataset(dataset, data_split[client_id])})['train']
        self.local = Local(model_rate_m, data_loader_m, label_split[client_id],
                           self.device)
        
        

    def training(self, 
                 lr,
                 distillation=False,
                 **kwargs):
        self.lr = lr
        self.local_parameters, evaluation_metric = self.local.train(self.local_parameters, self.lr)
        return self.local_parameters, self.param_idx, evaluation_metric

       

    def server_apply_gradient(self, gradients):
        self.optimizer.apply_gradients(zip(gradients,
                                           self.model.trainable_variables))

    def evaluating_step(self, x, y):
        
        with torch.no_grad():
            predictions = self.model(x, training=True)
            # y = torch.tensor(y, dtype=torch.long)
            pred_loss = self.loss_func(predictions, y)
        self.accuracy.append((y.eq(predictions.max(dim=1).indices).sum() / y.shape[0]).item())
            
        self.mean_loss.append(pred_loss.item())

    def evaluation(self, dataset, batch_size):
        epoch = math.ceil(dataset.size // batch_size)
        self.accuracy = []
        self.mean_loss = []

        for i in range(epoch):
            x, y = dataset.next_test_batch(batch_size)
            x, y = x.to(self.device), y.to(self.device)
            self.evaluating_step(x, y)
            # x, y = x.to("cpu"), y.to("cpu")

        mean_loss = np.mean(np.array(self.mean_loss))
        acc = np.mean(np.array(self.accuracy))

        return mean_loss, acc

    def get_trainable_weights(self):
        return self.model.trainable_variables

    def get_client_id(self):
        """ Return the client id """
        return self.cid

    # def get_trainable_variables(self):
    #     return self.model.trainable_variables

    def get_weights(self):
        """ Return the model's parameters """
        return self.model.get_weights()

    # def set_model_parameters(self, server_model):
    #     """ Assign server model's parameters to this client """
    #     self.model.set_weights(server_model.get_model_parameters())

    def set_weights(self, weights):
        """ Assign server model's parameters to this client """
        self.model.set_weights(weights)



