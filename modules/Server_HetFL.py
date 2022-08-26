import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import datetime
from numpy.lib.function_base import select
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
import psutil
import pdb
import gc
import objgraph
import pandas as pd
# import tensorflow as tf
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


from modules.Dataset import Dataset
# from modules.Model import ServerModel
from parameters import args
from modules.utils import get_model_mask, get_tb_logs_name

from modules.Client_HetFL import Client
from modules.HeteroFL.fed import Federation
from modules.HeteroFL.data import fetch_dataset, make_data_loader, split_dataset, SplitDataset
from modules.HeteroFL.metrics import Metric
from modules.HeteroFL.utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from modules.HeteroFL.config import cfg
from modules.HeteroFL import models
# import modules.VAE_model as CVAE



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

        # cfg['control_name'] = '1_10_0.1_iid_fix_a1-b1-c1_gn_0_0'
        if args.control_name:
            cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args.control_name.split('_'))} \
                if args.control_name != 'None' else {}
        cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']])
        if self.dataset == 'cifar10':
            cfg['data_name'] = 'CIFAR10'
        else:
            cfg['data_name'] = self.dataset
        process_control()
        
        self.dataset_server = Dataset(self.dataset,
                                      split=self.clients_number,
                                      distorted_data=self.distorted_data)
        
        self.dataset, self.data_split, self.label_split \
            = self.get_dataset(self.dataset_server, self.clients_number)
        
        # cfg['model_split_rate'] = [1, 0.5, 0.25, 0.125, 0.0625]
        # cfg["global_model_rate"] = 1
        self.clients_dict, self.global_model, self.federation \
            = self.build_clients_and_server(self.clients_number,
                                            self.data_split,
                                            self.label_split,
                                            cfg['model_rate'])
        # self.server_model = 0

        # global_parameters = self.global_model.state_dict()
        self.tmp_model = eval('models.{}(model_rate=cfg["global_model_rate"]).to(self.device)'.format(cfg['model_name']))
        self.optimizer = make_optimizer(self.tmp_model, cfg['lr'])
        self.scheduler = make_scheduler(self.optimizer)
        self.learning_rate = cfg['lr']
        
        
        
        
        
        # self.CVAE = CVAE.VAE()
        
    def get_dataset(self, dataset_server, client_nums):
        datasetss = {'train': dataset_server.train_data,
                     'test': dataset_server.test_data}
        if cfg['data_name'] in ['Fmnist', 'CIFAR10']:
            cfg['classes_size'] = len(datasetss['train'].classes)
        
        data_split, label_split = split_dataset(datasetss, client_nums, 'iid')
        return datasetss, data_split, label_split
        

    def build_clients_and_server(self, num, data_split,
                                 label_split, model_split_rate):
        num_input = self.num_input
        num_input_channel = self.num_input_channel
        num_classes = self.num_classes
        batch_size = self.batch_size
        clients_training_epoch = self.clients_training_epoch
        dataset = self.dataset
        client_nums = self.clients_number

        clients_dict = {}
        input_data_shape = [batch_size, num_input_channel, num_input, num_input]
        
        global_model = eval('models.{}(model_rate=cfg["global_model_rate"]).to(self.device)'.format(cfg['model_name']))
        
        global_parameters = global_model.state_dict()
        federation = Federation(global_parameters, model_split_rate, self.label_split)


        # create Clients and models
        for cid in range(num):
            clients_dict[cid] = Client(input_shape=input_data_shape,
                                        num_classes=num_classes,
                                        learning_rate=cfg['lr'],
                                        client_id=cid,
                                        batch_size=batch_size,
                                        clients_training_epoch=clients_training_epoch,
                                        dataset=dataset['train'],
                                        device=self.device,
                                        federation=federation,
                                        data_split=data_split['train'],
                                        label_split=label_split,
                                        is_worker=True)



           
            # server_extrator = CVAE.ServerModel(num_classes).to(self.device)


        # return clients_dict, server_extrator
        return clients_dict, global_model, federation

    def choose_clients(self):
        """ randomly choose some clients """
        client_num = self.clients_number
        ratio = self.clients_sample_ratio

        self.choose_num = math.floor(client_num * ratio)
        return np.random.permutation(client_num)[:self.choose_num]

    def list_divided_int(self, a, b):
        assert len(a) > 1
        for k in range(len(a)):
            a[k] /= b
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

    def delay_compensation(self, client_gradients, client_parameters_dict, outdated_time_steps, cid):
        ready_gradients = client_gradients
        lam = 12
        server_parameters = self.server_model.get_trainable_weights()
        training_variables = client_parameters_dict[cid]
        for k in range(len(training_variables)):
            gaps = server_parameters[k] - training_variables[k]
            # a = lam * client_gradients[k] * client_gradients[k] * gaps

            ready_gradients[k] = client_gradients[k] + \
                                 lam * client_gradients[k] * client_gradients[k] * gaps
        return ready_gradients

    
    
    def stats(self, dataset, model):
        with torch.no_grad():
            test_model = eval('models.{}(model_rate=cfg["global_model_rate"], track=True).to(self.device)'
                            .format(cfg['model_name']))
            test_model.load_state_dict(model.state_dict(), strict=False)
            data_loader = make_data_loader({'train': dataset})['train']
            test_model.train(True)
            for i, input in enumerate(data_loader):
                # input = collate(input)
                input = to_device(input, self.device)
                test_model(input)
        return test_model


    def test(self, dataset, data_split, label_split, model):
        test_records = []
        with torch.no_grad():
            metric = Metric()
            model.train(False)
            for m in range(cfg['num_users']):
                data_loader = make_data_loader({'test': SplitDataset(dataset, data_split[m])})['test']
                for i, input in enumerate(data_loader):
                    # input = collate(input)
                    # input_size = input['img'].size(0)
                    # input['label_split'] = torch.tensor(label_split[m])
                    input = to_device(input, self.device)
                    output = model(input)
                    output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                    evaluation = metric.evaluate(cfg['metric_name']['test']['Local'], input, output)
                    test_records.append(evaluation)
                    # logger.append(evaluation, 'test', input_size)
            data_loader = make_data_loader({'test': dataset})['test']
            for i, input in enumerate(data_loader):
                # input = collate(input)
                # input_size = input['img'].size(0)
                input = to_device(input, self.device)
                output = model(input)
                output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                evaluation = metric.evaluate(cfg['metric_name']['test']['Global'], input, output)
                test_records.append(evaluation)
                # logger.append(evaluation, 'test', input_size)
            # info = {'info': ['Model: {}'.format(cfg['model_tag']),
            #                 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
            # logger.append(info, 'test', mean=False)
            # logger.write('test', cfg['metric_name']['test']['Local'] + cfg['metric_name']['test']['Global'])
            
            test_loss, test_acc = 0, 0
            for idx, val in enumerate(test_records):
                if 'Local-Loss' in val:
                    test_loss += val['Local-Loss']
                elif 'Global-Loss' in val:
                    test_loss += val['Global-Loss']
                if 'Global-Accuracy' in val:
                    test_acc += val['Global-Accuracy']
                elif 'Local-Accuracy' in val:
                    test_acc += val['Local-Accuracy']
            test_loss /= len(test_records)
            test_acc /= len(test_records)
            
        return test_loss, test_acc

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
        G_losses, D_losses = [], []
        train_loss = 0
        train_acc = 0
        train_distill_loss, train_bn_mean_loss, train_bn_var_loss = 0, 0, 0
        train_feature_loss = 0
        cnt = 0
        cnt_distill = 0
        tmp_distill_loss = 0
        server_features, server_generated_labels = 0, 0
        alpha = args.alpha
        avg_server_y_logits, avg_server_client_features = 0, 0
        server_feature_logits = 0
        clients_y_logits = {}
        res = {}
        distillation_flag = False
        max_training_epoches = int(epoch * clients_number * ratio)
        cnt_flip_flag = 1

        # tensorboard parameters
        model_mask = get_model_mask(args)
        
        cmp_log_dir = get_tb_logs_name(args)
        
        compared_summary_writer = SummaryWriter(cmp_log_dir)


        if args.asyn_FL_flag:
            assert not args.y_distillation_flag
            assert not args.dataset_distillation_flag
            assert not args.fixed_parameters_flag
            assert args.aggregate_flag
        
        if args.delayed_gradients_flag:
            assert not args.delayed_gradients_divided_flag

        
        for ep in tqdm(range(max_training_epoches)):
            # randomly choose some clients each epoch
            selected_clients = self.choose_clients()


            # update clients' states
            for idx, cid in enumerate(selected_clients):
                train_model = self.clients_dict[cid]

                if outdated_flag[cid] > 0:
                    outdated_flag[cid] += 1
                    continue
                else:
                    if cid in selected_clients:
                        outdated_flag[cid] = 1

            # Clients which finish their updating processes.
            # change it for all version
            completed_clients = self.completed_client_this_iter(outdated_flag, int(clients_number * ratio))
            local_parameters = [None for _ in range(completed_clients.shape[0])]
            client_id_list = [None for _ in range(completed_clients.shape[0])]
            param_idx = []
            train_records = []
            for idx, cid in enumerate(completed_clients):
                assert outdated_flag[cid] > 0
                train_model = self.clients_dict[cid]
                client_id_list[idx] = cid
                
                local_parameters[idx], tmp_param_idx, train_metric = train_model.training(self.learning_rate)
                param_idx.extend(tmp_param_idx)
                train_records.append(train_metric)
            self.federation.combine(local_parameters, param_idx, client_id_list)
            self.global_model.load_state_dict(self.federation.global_parameters)

            self.scheduler.step()
            self.learning_rate = self.optimizer.param_groups[0]['lr']
            
            eval_interval = 0
            if max_training_epoches <= 500:
                eval_interval = int(max_training_epoches / 100)
            else:
                eval_interval = int(max_training_epoches / 100)

            if ep % eval_interval == 0:
                
                test_model = self.stats(self.dataset['train'], self.global_model)
                test_loss, test_acc = self.test(self.dataset['test'], self.data_split['test'], self.label_split,
                              test_model)
                

                    
                print("test_loss={} | test_acc={}".format(test_loss, test_acc))
                print("lr: {}".format(self.learning_rate))
                
                loss_name = 'loss'
                acc_name = 'acc'
                compared_summary_writer.add_scalars("loss", {loss_name: test_loss}, int(ep/eval_interval))
                compared_summary_writer.add_scalars("accuracy", {acc_name: test_acc / 100}, int(ep/eval_interval))

            

            # if ep % (clients_number * 10) == 0:
            #     gc.collect()

        return train_loss_list, train_acc_list, \
            train_distill_loss_list, train_bn_mean_loss_list, \
                train_bn_var_loss_list, mean_loss_list, \
                acc_list, G_losses, D_losses
        #, evaluate_client_acc

    def run(self):

        # if asynchronous:
        # ray.init(num_gpus=1)
        # train_loss_list, train_acc_list, mean_loss_list, acc_list = self.asynchronous_training()
        # train_loss_list, train_acc_list, train_distill_loss_list, eval_loss_list, eval_acc_list, evaluate_client_acc = self.asynchronous_training()
        self.asynchronous_training()

        # print(self.server_model.model.summary())
        # df_client_acc = pd.DataFrame.from_dict(evaluate_client_acc)
        

        # model_mask = get_model_mask(args)

        # # if self.dataset == "mnist":
        # txt_save_path = "txt_result/{}/{}/alpha{}beta{}_lr{}/".format(self.dataset, model_mask, args.alpha, args.beta, self.learning_rate)
        # makedirs(txt_save_path)
        # np.savetxt('{}train_loss.txt'.format(txt_save_path), np.array(train_loss_list))
        # np.savetxt('{}train_acc.txt'.format(txt_save_path), np.array(train_acc_list))
        # np.savetxt('{}distill_loss.txt'.format(txt_save_path), np.array(train_distill_loss_list))
        # np.savetxt('{}evaluate_loss.txt'.format(txt_save_path), np.array(eval_loss_list))
        # np.savetxt('{}evaluate_acc.txt'.format(txt_save_path), np.array(eval_acc_list))
        # # df_client_acc.to_pickle('{}evaluate_client_acc.pkl'.format(txt_save_path))

        # # plt.plot(G_losses, label="Generator losses")
        # # plt.plot(D_losses, label="Discriminator losses")
        # # plt.title("GAN losses")
        # # plt.ylabel("loss")
        # # plt.xlabel("Number of iterations")
        # # plt.legend()
        # pic_save_path = "pic/{}/{}/alpha{}beta{}_lr{}/".format(self.dataset, model_mask, args.alpha, args.beta, self.learning_rate)
        # makedirs(pic_save_path)
        # # plt.savefig("{}GAN_losses_alpha{}_lr{}.png".format(pic_save_path, args.alpha, self.learning_rate))
        # # plt.close()


        # plt.plot(train_loss_list, label="train_loss")
        # plt.plot(eval_loss_list, label="evaluate_loss")
        # plt.title("Loss")
        # plt.ylabel("loss")
        # plt.xlabel("Number of iterations")
        # plt.legend()
        # # pic_save_path = "pic/{}/{}/alpha{}_lr{}/".format(self.dataset, model_mask, args.alpha, self.learning_rate)
        # # makedirs(pic_save_path)
        # plt.savefig("{}asydis_loss_alpha{}_lr{}.png".format(pic_save_path, args.alpha, self.learning_rate))
        # plt.close()

        # plt.plot(train_acc_list, label="train_acc")
        # plt.plot(eval_acc_list, label="evaluate_acc")
        # plt.title("Accuracy")
        # plt.ylabel("Acc")
        # plt.xlabel("Number of iterations")
        # plt.legend()
        # plt.savefig("{}asydis_Acc_alpha{}_lr{}.png".format(pic_save_path, args.alpha, self.learning_rate))
        # plt.close()

        # plt.plot(train_distill_loss_list, label="distill_loss")
        # plt.title("Distillation_loss")
        # plt.ylabel("loss")
        # plt.xlabel("Number of iterations")
        # plt.legend()
        # plt.savefig("{}asydis_distill_loss_alpha{}_lr{}.png".format(pic_save_path, args.alpha, self.learning_rate))
        # plt.close()
