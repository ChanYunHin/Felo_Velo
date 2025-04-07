import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import datetime
from numpy.lib.function_base import select
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


from modules.Dataset import Dataset
from parameters import args
from modules.utils import get_model_mask, get_tb_logs_name

from modules.Client_Velo import Client

import modules.VAE_model as CVAE



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



        self.dataset_server = Dataset(self.dataset,
                                      split=self.clients_number,
                                      distorted_data=self.distorted_data)

        self.clients_dict, self.server_extractor = self.build_clients_and_server(self.clients_number,
                                                                                 self.distorted_data,
                                                                                 self.dataset_server)
        self.server_model = 0

        

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
                                        is_worker=True, )

            server_model = Client(input_shape=input_data_shape,
                                    num_classes=num_classes,
                                    learning_rate=learning_rate,
                                    client_id=-1,
                                    distorted_data=distorted_data,
                                    batch_size=batch_size,
                                    clients_training_epoch=clients_training_epoch,
                                    is_worker=False)
        else:
            client_models = {}
            if args.homo_flag or args.middle_flag:
                if args.resnet_flag:
                    group_size = num // 5
                    group_model_name = {0: "small_resnet", 1: "small_resnet", 2: "small_resnet", 3: "small_resnet", 4: "small_resnet"}
                    for group_idx in range(5):
                        for idx in range(group_size):
                            client_models[idx + group_idx * group_size] = {"model_type": group_model_name[group_idx]}
                            print("Client {} model type: {}".format(idx + group_idx * group_size, group_model_name[group_idx]))


            elif args.resnet_flag:
                
                group_size = num // 5
                group_model_name = {0: "small_resnet", 1: "small_resnet_1", 2: "small_resnet_2", 3: "small_resnet_3", 4: "resnet"}
                for group_idx in range(5):
                    for idx in range(group_size):
                        client_models[idx + group_idx * group_size] = {"model_type": group_model_name[group_idx]}
                        print("Client {} model type: {}".format(idx + group_idx * group_size, group_model_name[group_idx]))
                
            

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


           
            server_extrator = CVAE.ServerModel(num_classes).to(self.device)


        return clients_dict, server_extrator

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


        return mean_loss, acc

    def completed_client_this_iter(self, outdated_flag, completed_number):
        nonzero_index = np.nonzero(outdated_flag)
        if len(nonzero_index[0]) > 0:
            completed_index = np.random.choice(nonzero_index[0], completed_number, replace=False)
            return completed_index
        else:
            return []

    def collect_logits_features(self, 
                                res,
                                cid,
                                client_y_logits,
                                client_features_dict,
                                client_feature_logits=None):
        
        if "server_y_logits" not in res:
            res["server_y_logits"] = {}
        if "server_labels_counts_y_logits" not in res:
            res["server_labels_counts_y_logits"] = {}
        if "server_client_features" not in res:
            res["server_client_features"] = {}


        for key, val in client_y_logits.items():
            if key in res["server_y_logits"]:
                res["server_labels_counts_y_logits"][key] += 1
                res["server_y_logits"][key].append(val)
                while len(res["server_y_logits"][key]) >= 50000:
                    res["server_y_logits"][key].pop(0)
                    res["server_labels_counts_y_logits"][key] -= 1
                assert len(res["server_y_logits"][key]) <= 50005
            else:
                res["server_labels_counts_y_logits"][key] = 1
                res["server_y_logits"][key] = [val]

        for key, val in client_features_dict.items():
            if key in res["server_client_features"]:
                res["server_client_features"][key].extend(val)
                while len(res["server_client_features"][key]) >= 5000:
                    res["server_client_features"][key].pop(0)
                assert len(res["server_client_features"][key]) <= 5001
            else:
                res["server_client_features"][key] = val

                
        return res


    def initialize_counting(self, res):
        server_labels_counts = res["server_labels_counts_y_logits"]
        
        tmp_labels_counts = server_labels_counts.copy()
        for key, val in server_labels_counts.items():
            tmp_labels_counts[key] = 1
        res["server_labels_counts_y_logits"] = tmp_labels_counts

        if len(res["bn_features"]["two_layers"]["mean"]) >= 300:
            res["bn_features"]["two_layers"]["mean"] \
                = res["bn_features"]["two_layers"]["mean"][-150:]
            res["bn_features"]["two_layers"]["var"] \
                = res["bn_features"]["two_layers"]["var"][-150:]

        if len(res["bn_features"]["three_layers"]["mean"]) >= 300:
            res["bn_features"]["three_layers"]["mean"] \
                = res["bn_features"]["three_layers"]["mean"][-150:]
            res["bn_features"]["three_layers"]["var"] \
                = res["bn_features"]["three_layers"]["var"][-150:]

        return res


    def get_server_client_features(self, res):
        x_features, y_label = [], []
        # draw_logits = {}
        '''Use mean and var to generate features
           features as dataset to train the server VAE model
           Not use mean and var vector to be the training dataset'''
        for key, client_features in res["server_client_features"].items():
            x_features.extend(client_features)
            y_size = len(client_features)
            y_label.extend([key for i in range(y_size)])
            # draw the training data for GAN
            # if key not in draw_logits:
                # draw_logits[key] = list(logit)

        x_features = torch.cat(x_features, dim=0).view(-1, x_features[0].shape[0])
        y_label = torch.tensor(y_label, device=self.device)
        # one_hot_y = F.one_hot(y_label, num_classes=10)
        # x_y_logits = torch.cat([x_features, one_hot_y], dim=1)
        features_dataset = TensorDataset(x_features, y_label) 
        features_dataloader = DataLoader(dataset=features_dataset,
                                         batch_size = self.batch_size,
                                         shuffle=True)


        return features_dataloader

    def get_features_from_generator(self, generator, 
                                    batch_size,
                                    latent_size, random_seed):
        # set a random seed manually, to reproduce the same features for all clients
        torch.manual_seed(random_seed)
        generated_label = [i for i in range(self.num_classes)]
        # np.random.shuffle(generated_label)
        generated_label = torch.tensor(generated_label, device=self.device)
        noise = torch.randn(self.num_classes, 
                            latent_size, 
                            device=self.device)
        with torch.no_grad():
            generated_features = generator.generate_data(noise, generated_label)
        
        # reset the random seed
        torch.seed()
        generated_features_dict = {i:generated_features[i, :] for i in range(self.num_classes)}
        return generated_features_dict

    def get_server_y_logits(self, res):
        server_y_logits_dict = {}
        for key, val in res["server_y_logits"].items():
            server_y_logits_dict[key] = torch.mean(torch.stack(val, dim=0), dim=0)
        return server_y_logits_dict


    def get_random_seed(self):
        return torch.random.seed()

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
        client_feature_logits = None
        alpha = args.alpha
        avg_server_y_logits, avg_server_distilled_inputs = 0, 0
        server_feature_logits = 0
        server_features_dict = {}
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


            if not args.asyn_FL_flag:
                if args.pre_train:
                    if ep == int(max_training_epoches * 0.4):
                        distillation_flag = args.y_distillation_flag
                        #training
                        features_dataloader = self.get_server_client_features(res)
                        self.server_extractor = CVAE.server_extractor_training(self.server_extractor,
                                                                               num_epochs=10,
                                                                               dataloader=features_dataloader,
                                                                               device=self.device)
                        
                else:
                    if ep >= int(max_training_epoches * 0.05) and ep % int(args.VAE_training_interval) == 0:
                        distillation_flag = args.y_distillation_flag
                        #training
                        features_dataloader = self.get_server_client_features(res)
                        self.server_extractor = CVAE.server_extractor_training(self.server_extractor,
                                                                               num_epochs=10,
                                                                               dataloader=features_dataloader,
                                                                               device=self.device)
                        


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

                       
                        if args.asyn_FL_flag:
                            clients_parameters_dict[cid] = self.server_model.get_trainable_weights()
                            train_model.set_weights(self.server_model.get_weights())

            completed_clients = self.completed_client_this_iter(outdated_flag, int(clients_number * ratio))

            rand_seed = self.get_random_seed()
            if distillation_flag:
                server_features_dict = self.get_features_from_generator(self.server_extractor,
                                                                        batch_size=batch_size,
                                                                        latent_size=args.latent_size,
                                                                        random_seed=rand_seed)
                avg_server_y_logits = self.get_server_y_logits(res)
            
            for idx, cid in enumerate(completed_clients):
                assert outdated_flag[cid] > 0
                train_model = self.clients_dict[cid]

                
                training_result = train_model.training(server_generator=self.server_extractor,
                                                       distillation=distillation_flag,
                                                       server_features=server_features_dict,
                                                       server_y_logits=avg_server_y_logits)

                # unpack the training result
                if distillation_flag:
                    tmp_label_loss = training_result["CE_loss"]
                    tmp_distill_loss = training_result["distill_loss"]
                    tmp_train_acc = training_result["Acc"]
                    client_gradients = training_result["gradients"]
                    tmp_y_logits = training_result["y_logits"]
                    client_features = training_result["features_dict"]
                    tmp_feature_loss = training_result["feature_loss"]
                else:
                    tmp_label_loss = training_result["CE_loss"]
                    tmp_train_acc = training_result["Acc"]
                    client_gradients = training_result["gradients"]
                    tmp_y_logits = training_result["y_logits"]
                    client_features = training_result["features_dict"]
                clients_y_logits[cid] = tmp_y_logits

                
                if args.asyn_FL_flag:
                    if args.delayed_gradients_divided_flag:
                        ready_gradient = self.list_divided_int(client_gradients,
                                                            outdated_flag[cid])
                    elif args.delayed_gradients_flag:
                        ready_gradient = self.delay_compensation(client_gradients,
                                                                clients_parameters_dict,
                                                                outdated_flag[cid],
                                                                cid)
                    else:
                        ready_gradient = client_gradients
                    self.server_model.server_apply_gradient(ready_gradient)
                


                outdated_flag[cid] = 0

                if not args.asyn_FL_flag:
                    res = self.collect_logits_features(res, 
                                                        cid,
                                                        tmp_y_logits,
                                                        client_features)

                train_loss += tmp_label_loss
                train_acc += tmp_train_acc
                cnt += 1
                if distillation_flag:
                    cnt_distill += 1
                    train_distill_loss += tmp_distill_loss
                    train_feature_loss += tmp_feature_loss
                    

                
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
                group_model_acc = [[] for _ in range(5)]
                evaluate_num = 0
                
                if args.asyn_FL_flag:
                    mean_loss, acc = self.evaluate(self.dataset_server.test,
                                                    batch_size * 10)
                        
                    evaluate_mean_loss.append(mean_loss)
                    evaluate_mean_acc.append(acc)
                else:
                    if self.clients_number <= 20:
                        evaluate_num = self.clients_number
                    else:
                        evaluate_num = int(self.clients_number * 0.1)
                    group_size = self.clients_number // 5
                    for idx in range(evaluate_num):
                        if self.clients_number >= 20:
                            selected_idx = np.random.choice(self.clients_number)
                        else:
                            selected_idx = idx

                        mean_loss, acc = self.evaluate(self.dataset_server.test,
                                                       batch_size * 10,
                                                       self.clients_dict[selected_idx])

                        evaluate_mean_loss.append(mean_loss)
                        evaluate_mean_acc.append(acc)
                        

                        group_model_acc[selected_idx // group_size].append(acc)
                        

                mean_loss = np.mean(np.array(evaluate_mean_loss))
                acc = np.mean(np.array(evaluate_mean_acc))
                for idx in range(len(group_model_acc)):
                    group_model_acc[idx] = np.mean(np.array(group_model_acc[idx]))
                
                

                for idx in range(self.clients_number):
                    tmp_model = self.clients_dict[idx]
                    torch.save(tmp_model.model.state_dict(), 'state_dict/Velo_{}_{}'.format(idx, self.dataset))
                

                loss_name = 'beta{}_loss'.format(args.beta)
                acc_name = 'beta{}_acc'.format(args.beta)
                compared_summary_writer.add_scalars("loss", {loss_name: mean_loss}, int(ep/eval_interval))
                compared_summary_writer.add_scalars("accuracy", {acc_name: acc}, int(ep/eval_interval))
                for idx in range(len(group_model_acc)):
                    compared_summary_writer.add_scalars("group_accuracy", {'Velo_group{}'.format(idx): group_model_acc[idx]}, int(ep/eval_interval))

                mean_loss_list.append(mean_loss)
                acc_list.append(acc)

                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)

                

                    

                if distillation_flag:
                    train_distill_loss /= cnt_distill
                    train_feature_loss /= cnt_distill
                    train_bn_mean_loss /= cnt_distill
                    train_bn_var_loss /= cnt_distill
                    train_distill_loss_list.append(train_distill_loss)
                    train_bn_mean_loss_list.append(train_bn_mean_loss)
                    train_bn_var_loss_list.append(train_bn_var_loss)


                    
                    print("\nDistillation: loss={} | feature_loss={}".format(train_distill_loss, train_feature_loss))
                    train_distill_loss, train_feature_loss = 0, 0
                    train_bn_mean_loss, train_bn_var_loss = 0, 0
                    cnt_distill = 0
                
                print("training: loss={} | accuracy={}".format(train_loss,
                                                            train_acc))
                print("evaluating: loss={} | accuracy={}".format(mean_loss, acc))
                print("model_mask: {}_beta{}".format(model_mask, args.beta))
                print("lr: {}".format(self.clients_dict[idx].optimizer.param_groups[0]['lr']))
                train_loss = 0
                train_acc = 0
                cnt = 0



        return train_loss_list, train_acc_list, \
            train_distill_loss_list, train_bn_mean_loss_list, \
                train_bn_var_loss_list, mean_loss_list, \
                acc_list, G_losses, D_losses
        #, evaluate_client_acc

    def run(self):

        train_loss_list, train_acc_list, \
            train_distill_loss_list, train_bn_mean_loss_list,\
                train_bn_var_loss_list, eval_loss_list, eval_acc_list, \
                    G_losses, D_losses \
                = self.asynchronous_training()

        

        model_mask = get_model_mask(args)

        # if self.dataset == "mnist":
        txt_save_path = "txt_result/{}/{}/alpha{}beta{}_lr{}/".format(self.dataset, model_mask, args.alpha, args.beta, self.learning_rate)
        makedirs(txt_save_path)
        np.savetxt('{}train_loss.txt'.format(txt_save_path), np.array(train_loss_list))
        np.savetxt('{}train_acc.txt'.format(txt_save_path), np.array(train_acc_list))
        np.savetxt('{}distill_loss.txt'.format(txt_save_path), np.array(train_distill_loss_list))
        np.savetxt('{}evaluate_loss.txt'.format(txt_save_path), np.array(eval_loss_list))
        np.savetxt('{}evaluate_acc.txt'.format(txt_save_path), np.array(eval_acc_list))

        pic_save_path = "pic/{}/{}/alpha{}beta{}_lr{}/".format(self.dataset, model_mask, args.alpha, args.beta, self.learning_rate)
        makedirs(pic_save_path)



        plt.plot(train_loss_list, label="train_loss")
        plt.plot(eval_loss_list, label="evaluate_loss")
        plt.title("Loss")
        plt.ylabel("loss")
        plt.xlabel("Number of iterations")
        plt.legend()

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
