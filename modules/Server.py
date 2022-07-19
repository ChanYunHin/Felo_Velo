import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import datetime
from numpy.lib.function_base import select
from pytest import param

from tqdm import tqdm
import psutil
import pdb
import gc
import pandas as pd
import tensorflow as tf
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


from modules.Dataset import Dataset
from modules.Model import ClientModel
from parameters import args
from modules.utils import get_model_mask, get_tb_logs_name

from modules.Client import Client

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Server:
    def __init__(self, parameters):

        #### SOME TRAINING PARAMS ####
        self.clients_number = parameters.clients_number
        self.clients_sample_ratio = parameters.clients_sample_ratio
        self.epoch = parameters.epoch
        self.learning_rate = parameters.learning_rate
        self.decay_rate = parameters.decay_rate
        self.num_input = parameters.num_input  # image shape: 32*32
        self.num_input_channel = parameters.num_input_channel  # image channel: 3
        self.num_classes = parameters.num_classes  # Cifar-10 total classes (0-9 digits)
        self.batch_size = parameters.batch_size
        self.clients_training_epoch = parameters.clients_training_epoch
        self.dataset = parameters.dataset
        self.distorted_data = parameters.distorted_data
        self.device = parameters.device

        #### CREATE CLIENT AND LOAD DATASET ####
        # ray.init(num_cpus=1, local_mode=True, ignore_reinit_error=True, include_dashboard=True)


        self.dataset_server = Dataset(self.dataset,
                                      split=self.clients_number,
                                      distorted_data=self.distorted_data)

        self.clients_dict, self.server_model = self.build_clients_and_server(self.clients_number,
                                                                             self.distorted_data,
                                                                             self.dataset_server)

    def build_clients_and_server(self, num, distorted_data, dataset_server):
        learning_rate = self.learning_rate
        num_input = self.num_input
        num_input_channel = self.num_input_channel
        num_classes = self.num_classes
        batch_size = self.batch_size
        clients_training_epoch = self.clients_training_epoch
        dataset = self.dataset

        clients_dict = {}
        input_data_shape = [batch_size, num_input_channel, num_input, num_input]

        if args.asyn_FL_flag:
            # create Clients and models
            for cid in range(num):
                clients_dict[cid] = Client(input_shape=input_data_shape,
                                        num_classes=num_classes,
                                        learning_rate=learning_rate,
                                        client_id=cid,
                                        distorted_data=distorted_data,
                                        batch_size=batch_size,
                                        clients_training_epoch=clients_training_epoch,
                                        dataset=dataset_server.train[cid],
                                        dataset_name=dataset,
                                        is_worker=True,
                                        device=self.device)

                # clients_dict[cid] = Client(input_shape=[batch_size, num_input, num_input, num_input_channel],
                #                            num_classes=num_classes,
                #                            learning_rate=learning_rate,
                #                            client_id=cid).remote()

            server_model = Client(input_shape=input_data_shape,
                                    num_classes=num_classes,
                                    learning_rate=learning_rate,
                                    client_id=-1,
                                    distorted_data=distorted_data,
                                    batch_size=batch_size,
                                    clients_training_epoch=clients_training_epoch,
                                    is_worker=False,
                                    device=self.device)
        else:
            client_models = {}
            if args.homo_flag or args.middle_flag:
                if args.resnet_flag:
                    client_models[0] = {"model_type": "small_resnet"}
                    client_models[1] = {"model_type": "small_resnet"}
                    client_models[2] = {"model_type": "small_resnet"}
                    client_models[3] = {"model_type": "small_resnet"}
                    client_models[4] = {"model_type": "small_resnet"}
                    client_models[5] = {"model_type": "small_resnet"}
                    client_models[6] = {"model_type": "small_resnet"}
                    client_models[7] = {"model_type": "small_resnet"}
                    client_models[8] = {"model_type": "small_resnet"}
                    client_models[9] = {"model_type": "small_resnet"}
                else:
                    client_models[0] = {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
                    client_models[1] = {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
                    client_models[2] = {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
                    client_models[3] = {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
                    client_models[4] = {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
                    client_models[5] = {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
                    client_models[6] = {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
                    client_models[7] = {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
                    client_models[8] = {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
                    client_models[9] = {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
            elif args.resnet_flag:
                client_models[0] = {"model_type": "small_resnet"}
                client_models[1] = {"model_type": "small_resnet"}
                client_models[2] = {"model_type": "small_resnet_1"}
                client_models[3] = {"model_type": "small_resnet_1"}
                client_models[4] = {"model_type": "small_resnet_2"}
                client_models[5] = {"model_type": "small_resnet_2"}
                client_models[6] = {"model_type": "small_resnet_3"}
                client_models[7] = {"model_type": "small_resnet_3"}
                client_models[8] = {"model_type": "resnet"}
                client_models[9] = {"model_type": "resnet"}
            else:
                client_models[0] = {"model_type": "2_layer_CNN", "params": {"n1": 64, "n2": 256, "dropout_rate": 0.2}}
                client_models[1] = {"model_type": "2_layer_CNN", "params": {"n1": 64, "n2": 384, "dropout_rate": 0.2}}
                client_models[2] = {"model_type": "2_layer_CNN", "params": {"n1": 64, 'n2': 512, "dropout_rate": 0.2}}
                client_models[3] = {"model_type": "2_layer_CNN", "params": {"n1": 64, "n2": 256, "dropout_rate": 0.3}}
                client_models[4] = {"model_type": "2_layer_CNN", "params": {"n1": 64, "n2": 512, "dropout_rate": 0.4}}
                client_models[5] = {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 256, "dropout_rate": 0.2}}
                client_models[6] = {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 192, "dropout_rate": 0.2}}
                client_models[7] = {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 192, "n3": 256, "dropout_rate": 0.2}}
                client_models[8] = {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 128, "dropout_rate": 0.3}}
                client_models[9] = {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
            

            # create Clients and models
            for cid in range(num):
                clients_dict[cid] = Client(input_shape=input_data_shape,
                                        num_classes=num_classes,
                                        learning_rate=learning_rate,
                                        client_id=cid,
                                        distorted_data=distorted_data,
                                        batch_size=batch_size,
                                        clients_training_epoch=clients_training_epoch,
                                        dataset=dataset_server.train[cid],
                                        dataset_name=dataset,
                                        device=self.device,
                                        is_worker=True,
                                        neural_network_shape=client_models[cid])

                # clients_dict[cid] = Client(input_shape=[batch_size, num_input, num_input, num_input_channel],
                #                            num_classes=num_classes,
                #                            learning_rate=learning_rate,
                #                            client_id=cid).remote()

           
            server_model = Client(input_shape=input_data_shape,
                                    num_classes=num_classes,
                                    learning_rate=learning_rate,
                                    client_id=-1,
                                    distorted_data=distorted_data,
                                    batch_size=batch_size,
                                    device=self.device,
                                    clients_training_epoch=clients_training_epoch,
                                    is_worker=False)


        return clients_dict, server_model

    def choose_clients(self):
        """ randomly choose some clients """
        client_num = self.clients_number
        ratio = self.clients_sample_ratio

        self.choose_num = math.floor(client_num * ratio)
        return np.random.permutation(client_num)[:self.choose_num]

    def list_divided_int(self, a, b):
        if len(a) <= 1:
            return b
        else:
            for k in range(len(a)):
                a[k] /= b
            return a
    
    def list_add(self, a, b):
        if a == 0:
            return b
        else:
            assert len(a) == len(b)
            for k in range(len(a)):
                a[k] += b[k]
            return a

    def evaluate(self, dataset, batch_size, eval_model):
        # batch_size = batch_size * 10
        epoch = math.ceil(dataset.size // batch_size)
        accuracy = []
        mean_loss = []
        loss_func = F.cross_entropy
        eval_model.model.eval()
        with torch.no_grad():
            for i in range(epoch):
                x, y = dataset.next_test_batch(batch_size)
                x, y = x.to(self.device), y.to(self.device)
                
                predictions = eval_model.model(x)
                # pdb.set_trace()
                pred_loss = loss_func(predictions, y)
                accuracy.append((y.eq(predictions.max(dim=1).indices).sum() / y.shape[0]).item())
                mean_loss.append(pred_loss.item())
                # x, y = x.to("cpu"), y.to("cpu")
        # eval_model.model.train()

        mean_loss = np.mean(np.array(mean_loss))
        acc = np.mean(np.array(accuracy))

        # print("\nevaluating: loss={} | accuracy={}".format(mean_loss, acc))

        return mean_loss, acc

    def completed_client_this_iter(self, outdated_flag, completed_number):
        nonzero_index = np.nonzero(outdated_flag)
        if len(nonzero_index[0]) > 0:
            completed_index = np.random.choice(nonzero_index[0], completed_number, replace=False)
            return completed_index
        else:
            return []

    # def delay_compensation(self, client_gradients, client_parameters_dict, outdated_time_steps, cid):
    #     ready_gradients = client_gradients
    #     lam = 12
    #     server_parameters = self.server_model.get_trainable_weights()
    #     training_variables = client_parameters_dict[cid]
    #     for k in range(len(training_variables)):
    #         gaps = server_parameters[k] - training_variables[k]
    #         # a = lam * client_gradients[k] * client_gradients[k] * gaps

    #         ready_gradients[k] = client_gradients[k] + \
    #                              lam * client_gradients[k] * client_gradients[k] * gaps
    #     return ready_gradients

    def collect_logits_bn_features(self, 
                                   res,
                                   cid,
                                   client_y_logits,
                                   client_bn_features):
        
        if "server_y_logits" not in res:
            res["server_y_logits"] = {}
        # server_y_logits = res["server_y_logits"]
        if "server_labels_counts_y_logits" not in res:
            res["server_labels_counts_y_logits"] = {}
        # server_labels_counts_y_logits = res["server_labels_counts_y_logits"]
        if "bn_features" not in res:
            res["bn_features"] = {"two_layers": {},
                                  "three_layers": {}}
        # if "count_for_bn_features" not in res:
        #     res["count_for_bn_features"] = {}

        for key, val in client_y_logits.items():
            if key in res["server_y_logits"]:
                res["server_labels_counts_y_logits"][key] += 1
                res["server_y_logits"][key] += val
            else:
                res["server_labels_counts_y_logits"][key] = 1
                res["server_y_logits"][key] = val

        for key, val in client_bn_features.items():
            if val.shape[0] == 2:
                if key in res["bn_features"]["two_layers"]:
                    res["bn_features"]["two_layers"][key].append(val)
                else:
                    res["bn_features"]["two_layers"][key] = [val]
            else:
                if key in res["bn_features"]["three_layers"]:
                    res["bn_features"]["three_layers"][key].append(val)
                else:
                    res["bn_features"]["three_layers"][key] = [val]
                
        return res

    def average_logits(self, res, clients_y_logits=None):   
        # update the logits and bn_features every n epoches  
        server_y_logits = res["server_y_logits"]
        server_labels_counts = res["server_labels_counts_y_logits"]
        tmp_y_logits = server_y_logits.copy()
        for key, val in server_y_logits.items():
            tmp_y_logits[key] = val / server_labels_counts[key]
        return tmp_y_logits

    # def get_server_bn_features(self, res, cid):
    #     if cid <= 4:
    #         server_bn_mean, server_bn_var \
    #             = self.get_server_different_layers_bn_features(res, "two_layers")
    #     else:
    #         server_bn_mean, server_bn_var \
    #             = self.get_server_different_layers_bn_features(res, "three_layers")
    #     return server_bn_mean, server_bn_var

    # def get_server_different_layers_bn_features(self, res, layer_type):
    #     len_bn_features = len(res["bn_features"][layer_type]["mean"])
    #     tmp_idx = list(range(len_bn_features))
    #     np.random.shuffle(tmp_idx)
    #     random_idx = tmp_idx[0] # this one can be changed, based on the results.
    #     server_bn_mean = res["bn_features"][layer_type]["mean"][random_idx]
    #     server_bn_var = res["bn_features"][layer_type]["var"][random_idx]
    #     return server_bn_mean, server_bn_var

    def initialize_counting(self, res):
        server_labels_counts = res["server_labels_counts_y_logits"]
        
        tmp_labels_counts = server_labels_counts.copy()
        for key, val in server_labels_counts.items():
            tmp_labels_counts[key] = 1
        res["server_labels_counts_y_logits"] = tmp_labels_counts

        # if len(res["bn_features"]["two_layers"]["mean"]) >= 300:
        #     res["bn_features"]["two_layers"]["mean"] \
        #         = res["bn_features"]["two_layers"]["mean"][-150:]
        #     res["bn_features"]["two_layers"]["var"] \
        #         = res["bn_features"]["two_layers"]["var"][-150:]

        # if len(res["bn_features"]["three_layers"]["mean"]) >= 300:
        #     res["bn_features"]["three_layers"]["mean"] \
        #         = res["bn_features"]["three_layers"]["mean"][-150:]
        #     res["bn_features"]["three_layers"]["var"] \
        #         = res["bn_features"]["three_layers"]["var"][-150:]

        return res


    def set_avg_weights(self, completed_clients):
        with torch.no_grad():
            weights_list = []
            for idx, cid in enumerate(completed_clients):
                train_model = self.clients_dict[cid]
                if idx == 0:
                    weights_list = [param.detach() for param in train_model.get_weights()]
                else:
                    tmp_weights_list = [param.detach() for param in train_model.get_weights()]
                    weights_list = [weights_list[ii] + tmp_weights_list[ii] for ii, val in enumerate(tmp_weights_list)]
                    # weights_list += tmp_weights_list
                # named_weights_list = [i for i in train_model.get_named_weights()]
                # weight += weights_list[-2]
                # bias += weights_list[-1]
            weights_list = [torch.nn.Parameter(weights_list[ii] / len(completed_clients))
                            for ii, val in enumerate(weights_list)]
            self.server_model.set_weights(weights_list)


    def asynchronous_training(self):
        epoch = self.epoch
        decay_rate = self.decay_rate
        ratio = self.clients_sample_ratio
        batch_size = self.batch_size
        clients_number = self.clients_number

        # record the delayed time steps
        outdated_flag = np.zeros(self.clients_number)

        clients_parameters_dict = {}
        evaluate_client_acc = {}
        mean_loss_list = []
        acc_list = []
        train_loss_list = []
        train_acc_list = []
        train_distill_loss_list, train_bn_mean_loss_list = [], []
        train_bn_var_loss_list = []
        train_loss = 0
        train_acc = 0
        train_distill_loss, train_bn_mean_loss, train_bn_var_loss = 0, 0, 0
        cnt = 0
        cnt_distill = 0
        tmp_distill_loss, ready_gradient = 0, 0
        server_bn_mean, server_bn_var = 0, 0
        alpha = args.alpha
        avg_server_y_logits, avg_server_distilled_inputs = 0, 0
        clients_y_logits = {}
        res = {}
        distillation_flag = False
        max_training_epoches = int(epoch * clients_number * ratio)

        # tensorboard parameters
        model_mask = get_model_mask(args)
        
        cmp_log_dir = get_tb_logs_name(args)
        
        compared_summary_writer = SummaryWriter(cmp_log_dir)


        if args.asyn_FL_flag:
            assert not args.y_distillation_flag
            # assert args.aggregate_flag
        
        if args.delayed_gradients_flag:
            assert not args.delayed_gradients_divided_flag

        for ep in tqdm(range(max_training_epoches)):
            # randomly choose some clients each epoch
            selected_clients = self.choose_clients()

            if not args.asyn_FL_flag:
                
                if ep >= int(max_training_epoches * 0.05):
                    distillation_flag = args.y_distillation_flag
                # objgraph.show_growth()
                # pdb.set_trace()

                # if ep % (clients_number * 10) == 0 and ep > 0:
                #     server_y_logits = self.average_logits(res)
                #     res["server_y_logits"] = server_y_logits
                #     res = self.initialize_counting(res)

            # update clients' states
            for idx, cid in enumerate(selected_clients):
                # for cid in range(clients_number):
                train_model = self.clients_dict[cid]

                if outdated_flag[cid] > 0:
                    outdated_flag[cid] += 1
                    continue
                else:
                    if cid in selected_clients:
                        outdated_flag[cid] = 1

                        # can't set weights in this algorithm
                        
                        if args.asyn_FL_flag:
                            clients_parameters_dict[cid] = self.server_model.get_weights()
                            train_model.set_weights(self.server_model.get_weights())
                            # train_model.set_weights(self.server_model.get_trainable_weights())

            # Clients which finish their updating processes.
            completed_clients = self.completed_client_this_iter(outdated_flag, int(clients_number * ratio))

            # training process
            for idx, cid in enumerate(completed_clients):
                assert outdated_flag[cid] > 0
                train_model = self.clients_dict[cid]
                if distillation_flag:
                    # server_bn_mean, server_bn_var = self.get_server_bn_features(res,
                    #                                                             cid)
                    if cid not in clients_y_logits:
                        avg_server_y_logits = self.average_logits(res)
                    else:
                        avg_server_y_logits = self.average_logits(res,
                                                                  clients_y_logits[cid])
                # new_weights = [param.detach() for param in train_model.get_weights()]
                
                training_result = train_model.training(server_bn_mean, 
                                                       server_bn_var, 
                                                       avg_server_y_logits,
                                                       distillation_flag)

                # unpack the training result
                if distillation_flag:
                    tmp_label_loss = training_result[0]
                    tmp_distill_loss = training_result[1]
                    tmp_train_acc = training_result[2]
                    client_gradients = training_result[3]
                    tmp_y_logits = training_result[4]
                    tmp_bn_features = training_result[5]
                    tmp_bn_mean_loss = training_result[6]
                    tmp_bn_var_loss = training_result[7]
                else:
                    tmp_label_loss = training_result[0]
                    tmp_train_acc = training_result[1]
                    client_gradients = training_result[2]
                    tmp_y_logits = training_result[3]
                    tmp_bn_features = training_result[4]
                clients_y_logits[cid] = tmp_y_logits

                # Can we make any criteria to check the efficiency of gradients?
                # if args.asyn_FL_flag:
                #     if args.delayed_gradients_divided_flag:
                #         ready_gradient += self.list_divided_int(client_gradients,
                #                                                outdated_flag[cid])
                #     # elif args.delayed_gradients_flag:
                #     #     ready_gradient = self.delay_compensation(client_gradients,
                #     #                                              clients_parameters_dict,
                #     #                                              outdated_flag[cid],
                #     #                                              cid)
                #     else:
                #         ready_gradient = self.list_add(ready_gradient, client_gradients)
                    
                


                outdated_flag[cid] = 0


                if not args.asyn_FL_flag:
                    res = self.collect_logits_bn_features(res, 
                                                          cid,
                                                          tmp_y_logits, 
                                                          tmp_bn_features)

                # if not args.asyn_FL_flag and ep < int(max_training_epoches * 0.2):
                #     res = self.collect_logits_bn_features(res, 
                #                                           cid,
                #                                           tmp_y_logits, 
                #                                           tmp_bn_features)

                train_loss += tmp_label_loss
                train_acc += tmp_train_acc
                cnt += 1
                if distillation_flag:
                    cnt_distill += 1
                    train_distill_loss += tmp_distill_loss
                    train_bn_mean_loss += tmp_bn_mean_loss
                    train_bn_var_loss += tmp_bn_var_loss
                
                # with train_summary_writer.as_default():
                #     tf.summary.scalar("loss", tmp_label_loss, step=ep)
                #     tf.summary.scalar("accuracy", tmp_train_acc, step=ep)
                #     if distillation_flag:
                #         tf.summary.scalar("BN_mean_loss", tmp_bn_mean_loss, step=ep)
                #         tf.summary.scalar("BN_var_loss", tmp_bn_var_loss, step=ep)

            if args.asyn_FL_flag:
                self.set_avg_weights(completed_clients)
                # self.server_model.server_apply_gradient(self.list_divided_int(ready_gradient,
                #                                                               len(completed_clients)))

            # for idx in range(self.clients_number):
            #     tmp_model = self.clients_dict[idx]
            #     tmp_model.scheduler.step()

            eval_interval = 0
            if max_training_epoches <= 500:
                eval_interval = int(max_training_epoches / 100)
            else:
                eval_interval = int(max_training_epoches / 100)

            if ep % eval_interval == 0:
                train_acc /= cnt
                train_loss /= cnt

                evaluate_mean_loss = []
                evaluate_mean_acc = []
                evaluate_num = 0
                
                # if args.asyn_FL_flag:
                #     mean_loss, acc = self.evaluate(self.dataset_server.test,
                #                                     batch_size * 10)
                        
                #     evaluate_mean_loss.append(mean_loss)
                #     evaluate_mean_acc.append(acc)
                # else:
                if self.clients_number <= 20:
                    evaluate_num = self.clients_number
                else:
                    evaluate_num = int(self.clients_number * 0.1)
                for idx in range(evaluate_num):
                    if self.clients_number >= 20:
                        selected_idx = np.random.choice(self.clients_number)
                    else:
                        selected_idx = idx
                    # self.server_model = self.clients_dict[selected_idx]
                    # self.server_model.set_weights(
                    #     self.clients_dict[selected_idx].get_trainable_weights()
                    # )

                    mean_loss, acc = self.evaluate(self.dataset_server.test,
                                                batch_size * 10,
                                                self.clients_dict[selected_idx])
                    # if selected_idx not in evaluate_client_acc:
                    #     evaluate_client_acc[selected_idx] = []

                    # evaluate_client_acc[selected_idx].append(acc)
                    evaluate_mean_loss.append(mean_loss)
                    evaluate_mean_acc.append(acc)

                mean_loss = np.mean(np.array(evaluate_mean_loss))
                acc = np.mean(np.array(evaluate_mean_acc))
                
                for idx in range(self.clients_number):
                    tmp_model = self.clients_dict[idx]
                    tmp_model.scheduler.step()

                mean_loss_list.append(mean_loss)
                acc_list.append(acc)

                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)

                # test_summary_writer.add_scalar("loss", mean_loss, int(ep/eval_interval))
                # test_summary_writer.add_scalar("acc", acc, int(ep/eval_interval))
                compared_summary_writer.add_scalars("loss", {'FedHe_loss': mean_loss}, int(ep/eval_interval))
                compared_summary_writer.add_scalars("accuracy", {'FedHe_acc': acc}, int(ep/eval_interval))

                # with test_summary_writer.as_default():
                #     tf.summary.scalar("loss", mean_loss, step=ep//100)
                #     tf.summary.scalar("accuracy", acc, step=ep//100)
                    

                if distillation_flag:
                    train_distill_loss /= cnt_distill
                    train_bn_mean_loss /= cnt_distill
                    train_bn_var_loss /= cnt_distill
                    train_distill_loss_list.append(train_distill_loss)
                    train_bn_mean_loss_list.append(train_bn_mean_loss)
                    train_bn_var_loss_list.append(train_bn_var_loss)


                    # with test_summary_writer.as_default():
                    #     tf.summary.scalar("avg_bn_mean_loss", train_bn_mean_loss, step=ep//100)
                    #     tf.summary.scalar("avg_bn_var_loss", train_bn_var_loss, step=ep//100)

                    
                    print("\nDistillation: loss={}".format(train_distill_loss))
                    # print("BN_mean: loss={}".format(train_bn_mean_loss))
                    # print("BN_var: loss={}".format(train_bn_var_loss))
                    train_distill_loss = 0
                    train_bn_mean_loss, train_bn_var_loss = 0, 0
                    cnt_distill = 0
                
                print("training: loss={} | accuracy={}".format(train_loss,
                                                            train_acc))
                print("evaluating: loss={} | accuracy={} | GPU: {}".format(mean_loss, acc, args.GPU_num))
                print("model_mask: {}".format(model_mask))
                train_loss = 0
                train_acc = 0
                cnt = 0

            # if ep % (clients_number * 10) == 0:
            #     gc.collect()

        return train_loss_list, train_acc_list, \
            train_distill_loss_list, train_bn_mean_loss_list, \
                train_bn_var_loss_list, mean_loss_list, \
                acc_list
        #, evaluate_client_acc

    def run(self):

        # if asynchronous:
        # ray.init(num_gpus=1)
        # train_loss_list, train_acc_list, mean_loss_list, acc_list = self.asynchronous_training()
        # train_loss_list, train_acc_list, train_distill_loss_list, eval_loss_list, eval_acc_list, evaluate_client_acc = self.asynchronous_training()
        train_loss_list, train_acc_list, \
            train_distill_loss_list, train_bn_mean_loss_list,\
                train_bn_var_loss_list, eval_loss_list, eval_acc_list \
                = self.asynchronous_training()

        # print(self.server_model.model.summary())
        # df_client_acc = pd.DataFrame.from_dict(evaluate_client_acc)
        

        model_mask = get_model_mask(args)
        # model_mask = get_model_mask(args)

        base_path = "{}/{}/alpha{}_lr{}/clientEpoches{}/".format(self.dataset, model_mask, args.alpha, self.learning_rate, args.clients_training_epoch)

        # model_save_path = "trained_model/{}".format(base_path)
        # makedirs(model_save_path)
        # for idx in range(self.clients_number):
        #     client_model = self.clients_dict[idx]
        #     torch.save(client_model.model.state_dict(), "{}client{}_paras.pt".format(model_save_path, idx))

        # if self.dataset == "mnist":
        txt_save_path = "txt_result/{}".format(base_path)
        makedirs(txt_save_path)
        np.savetxt('{}train_loss.txt'.format(txt_save_path), np.array(train_loss_list))
        np.savetxt('{}train_acc.txt'.format(txt_save_path), np.array(train_acc_list))
        np.savetxt('{}distill_loss.txt'.format(txt_save_path), np.array(train_distill_loss_list))
        np.savetxt('{}evaluate_loss.txt'.format(txt_save_path), np.array(eval_loss_list))
        np.savetxt('{}evaluate_acc.txt'.format(txt_save_path), np.array(eval_acc_list))
        # df_client_acc.to_pickle('{}evaluate_client_acc.pkl'.format(txt_save_path))

        plt.plot(train_loss_list, label="train_loss")
        plt.plot(eval_loss_list, label="evaluate_loss")
        plt.title("Loss")
        plt.ylabel("loss")
        plt.xlabel("Number of iterations")
        plt.legend()
        pic_save_path = "pic/{}".format(base_path)
        makedirs(pic_save_path)
        plt.savefig("{}asydis_loss_alpha{}_lr{}.png".format(pic_save_path, args.alpha, self.learning_rate))
        plt.close()

        plt.plot(train_acc_list, label="train_acc")
        plt.plot(eval_acc_list, label="evaluate_acc")
        plt.title("Accuracy")
        plt.ylabel("Acc")
        plt.xlabel("Number of iterations")
        plt.legend()
        plt.savefig("{}asydis_Acc_alpha{}_lr{}.png".format(pic_save_path, args.alpha, self.learning_rate))
        plt.close()

        plt.plot(train_distill_loss_list, label="distill_loss")
        plt.title("Distillation_loss")
        plt.ylabel("loss")
        plt.xlabel("Number of iterations")
        plt.legend()
        plt.savefig("{}asydis_distill_loss_alpha{}_lr{}.png".format(pic_save_path, args.alpha, self.learning_rate))
        plt.close()

        # plt.plot(train_bn_mean_loss_list, label="bn_mean_loss")
        # plt.title("BN_mean_loss")
        # plt.ylabel("loss")
        # plt.xlabel("Number of iterations")
        # plt.legend()
        # plt.savefig("{}asydis_bn_mean_loss_alpha{}_lr{}.png".format(pic_save_path, args.alpha, self.learning_rate))
        # plt.close()

        # plt.plot(train_bn_var_loss_list, label="bn_var_loss")
        # plt.title("BN_var_loss")
        # plt.ylabel("loss")
        # plt.xlabel("Number of iterations")
        # plt.legend()
        # plt.savefig("{}asydis_bn_var_loss_alpha{}_lr{}.png".format(pic_save_path, args.alpha, self.learning_rate))
        # plt.close()
