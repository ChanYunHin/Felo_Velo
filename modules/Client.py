import os
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
# from modules.ResnetModel import Resnet18, Resnet22, Resnet26, Resnet30, Resnet34
import pdb

# The definition of fed model
FedModel = namedtuple('FedModel', 'X Y DROP_RATE train_op loss_op acc_op')


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


class BNFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        # nch = input[0].shape[1]

        # mean = input[0].mean([0, 2, 3])
        # var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        # r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
        #     module.running_mean.data.type(var.type()) - mean, 2)

        # self.r_feature = r_feature
        self.bn_feature_mean = module.running_var.data
        self.bn_feature_var = module.running_mean.data
        # must have no output

    def close(self):
        self.hook.remove()


class Client:
    def __init__(self,
                 input_shape,
                 num_classes,
                 learning_rate,
                 client_id,
                 distorted_data,
                 batch_size,
                 clients_training_epoch,
                 device,
                 dataset=None,
                 dataset_name=None,
                 is_worker=True,
                 neural_network_shape=None):    
        # self.graph = tf.Graph()
        # self.sess = tf.Session(graph=self.graph)

        self.input_shape = input_shape
        self.device = device

        # Call the create function to build the computational graph
        if neural_network_shape is None:
            if not args.resnet_flag:
                self.model = ClientModel(input_shape, num_classes)
            else:
                self.model = SmallResnet50(in_channels=input_shape[1], 
                                           num_classes=num_classes)
                # self.model = Resnet34(num_classes=num_classes)
        else:
            if neural_network_shape["model_type"] == "2_layer_CNN":
                self.model = ClientModel_2CNN(input_shape, 
                                              num_classes, 
                                              layer1 = neural_network_shape["params"]["n1"],
                                              layer2 = neural_network_shape["params"]["n2"],
                                              dropout_rate = neural_network_shape["params"]["dropout_rate"])
            elif neural_network_shape["model_type"] == "3_layer_CNN":
                self.model = ClientModel(input_shape, 
                                         num_classes, 
                                         layer1 = neural_network_shape["params"]["n1"],
                                         layer2 = neural_network_shape["params"]["n2"],
                                         layer3 = neural_network_shape["params"]["n3"],
                                         dropout_rate = neural_network_shape["params"]["dropout_rate"])
            elif neural_network_shape["model_type"] == "resnet":
                # self.model = Resnet34(num_classes=num_classes)
                self.model = Resnet50(in_channels=input_shape[1], 
                                      num_classes=num_classes)
            elif neural_network_shape["model_type"] == "small_resnet":
                # self.model = Resnet18(num_classes=num_classes)
                self.model = SmallResnet50(in_channels=input_shape[1], 
                                           num_classes=num_classes)
            elif neural_network_shape["model_type"] == "small_resnet_1":
                # self.model = Resnet22(num_classes=num_classes)
                self.model = SmallResnet1(in_channels=input_shape[1], 
                                           num_classes=num_classes)
            elif neural_network_shape["model_type"] == "small_resnet_2":
                # self.model = Resnet26(num_classes=num_classes)
                self.model = SmallResnet2(in_channels=input_shape[1], 
                                           num_classes=num_classes)
            elif neural_network_shape["model_type"] == "small_resnet_3":
                # self.model = Resnet30(num_classes=num_classes)
                self.model = SmallResnet3(in_channels=input_shape[1], 
                                           num_classes=num_classes)
        # # If you don't have this function, you would not have true weights in the network,
        # # because you only get your weights when data pass through the network.
        # self.create_weights_for_model(self.model)

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)
        
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
        #                                                      max_lr=0.01,
        #                                                      epochs=args.epoch*int(args.clients_sample_ratio*args.clients_number),
        #                                                      steps_per_epoch=args.clients_training_epoch)
        
        total_epoches = args.epoch * args.clients_sample_ratio * args.clients_number * args.clients_training_epoch
        
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
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.99)
        
        self.loss_func = F.cross_entropy
        # can not use cross-entropy loss in the distillation.
        # because cross-entropy didn't consider the information from other classes.
        if args.KLD_flag:
            self.distill_loss_func = F.kl_div
            # self.distill_loss = tf.keras.metrics.KLDivergence()
        else:
            self.distill_loss_func = F.mse_loss
            # self.distill_loss = tf.keras.metrics.MeanSquaredError()
        # self.accuracy = tf.keras.metrics.CategoricalAccuracy(name='accuracy_{}'.format(client_id))
        self.mean_loss = []
        self.accuracy = []
        self.distill_loss = []
        self.bn_mean_loss = []
        self.bn_var_loss = []
        # tf.keras.metrics.CategoricalCrossentropy(from_logits=True,
        #                                                           name="mean_loss_{}".format(client_id))

        # initialize
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.cid = client_id
        self.dataset = dataset
        self.is_worker = is_worker
        self.distorted_data = distorted_data
        self.batch_size = batch_size
        self.clients_training_epoch = clients_training_epoch


        self.bn_feature_layers = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                self.bn_feature_layers.append(BNFeatureHook(module))
    
    def train_step(self, x, y, 
                   server_bn_mean, 
                   server_bn_var, 
                   logit_vectors=None, 
                   distillation=False, 
                   no_record=False):
        assert self.is_worker
        # alpha = args.alpha * (np.exp(current_epoches / max_epoches) - 1)
        alpha = args.alpha
        predictions = 0
        if distillation:
            # assert logit_vectors
            if args.softmax_flag or args.KLD_flag:
                softmax_logit_vectors = F.softmax(logit_vectors)
                log_softmax_predictions = 0
        """ one training iteration """
        
        self.model.train()
        predictions = self.model(x)
        bn_mean = torch.tensor([torch.mean(mod.bn_feature_mean) for mod in self.bn_feature_layers],
                                device=self.device)
        bn_var = torch.tensor([torch.mean(mod.bn_feature_var) for mod in self.bn_feature_layers],
                               device=self.device)
        pred_loss = self.loss_func(predictions, y)
        if distillation:
            if args.softmax_flag or args.KLD_flag:
                # if args.MSE_flag:
                #     distill_loss = self.distill_loss_func(softmax_logit_vectors, log_softmax_predictions)
                # else:
                log_softmax_predictions = F.log_softmax(predictions)
                distill_loss = self.distill_loss_func(log_softmax_predictions, softmax_logit_vectors)
            else:
                distill_loss = self.distill_loss_func(predictions, 
                                                      logit_vectors)

            pred_loss = pred_loss + alpha * distill_loss

        # a = [torch.mean(mod.bn_feature_mean) for mod in self.bn_feature_layers]
        # debug_item = torch.tensor(a, device=self.device)
        
        pred_loss.backward()
        
        # torch.nn.utils.clip_grad.clip_grad_value_(self.model.parameters(), 0.1)
        
        self.optimizer.step()
        gradients = [copy.deepcopy(param.grad.detach()) for param in self.model.parameters()]
        self.optimizer.zero_grad()
        # self.scheduler.step()
        # self.scheduler.step(pred_loss.item())
        
        # set up the accuracy in pytorch
        if not no_record:
            self.accuracy.append((y.eq(predictions.max(dim=1).indices).sum() / y.shape[0]).item())
            # self.accuracy.update_state(y, F.softmax(predictions))
            self.mean_loss.append(pred_loss.item())
        if distillation:
            self.distill_loss.append(distill_loss.item())
            self.bn_mean_loss.append(0)
            self.bn_var_loss.append(0)
        # if args.softmax_flag or args.KLD_flag:
        #     return gradients, \
        #            F.softmax(predictions).detach(), \
        #            [bn_mean.detach(), bn_var.detach()]
        # else:
        return gradients, \
               predictions.detach(), \
               [bn_mean.detach(), bn_var.detach()]

    # def get_class_labels(self, one_hot_y):
    #     labels = [np.argmax(yi) for yi in one_hot_y]
    #     return labels

    def collect_logits(self, client_logit_vectors, labels, logit_vectors, count_for_labels):
        # one_hots to class labels
        # labels = self.get_class_labels(one_hot_y)
        for idx, logit_y in enumerate(client_logit_vectors):
            if labels[idx] not in logit_vectors:
                logit_vectors[labels[idx].item()] = logit_y
                count_for_labels[labels[idx].item()] = 1
            else:
                logit_vectors[labels[idx].item()] += logit_y
                count_for_labels[labels[idx].item()] += 1
        return logit_vectors, count_for_labels

    def collect_bn(self, 
                   client_bn_features, 
                   bn_features, 
                   count_for_bn_features):
        bn_mean = bn_features[0]
        bn_var = bn_features[1]
        if "mean" not in client_bn_features:
            client_bn_features["mean"] = bn_mean
            count_for_bn_features["mean"] = 1
            client_bn_features["var"] = bn_var
            count_for_bn_features["var"] = 1
        else:
            client_bn_features["mean"] += bn_mean
            count_for_bn_features["mean"] += 1
            client_bn_features["var"] += bn_var
            count_for_bn_features["var"] += 1
        return client_bn_features, count_for_bn_features
        

    def match_labels_to_logits(self, server_y_logits, labels):
        # labels = self.get_class_labels(one_hot_y)
        server_logit_vectors = []
        # match logit vectors for each label
        for idx, label in enumerate(labels):
            server_logit_vectors.append(server_y_logits[label.item()])
        # a = np.array(server_logit_vectors)
        return torch.stack(server_logit_vectors, dim=0)


    def sort_logits(self, 
                    client_y_logits, 
                    count_for_labels):

        for label, logit in client_y_logits.items():
            client_y_logits[label] = logit / count_for_labels[label]
        return client_y_logits

    def get_avg_bn_features(self,
                            client_bn_features,
                            count_for_bn_features):
        for features, val in client_bn_features.items():
            client_bn_features[features] = val / count_for_bn_features[features]
        return client_bn_features

    # def tmp_save_pic(self, save_path):
    #     for i in range(self.distilled_x.shape[0]):
    #         tf.keras.preprocessing.image.save_img("{}/{}.png".format(save_path, i), self.distilled_x[i])

    def training(self, 
                 server_bn_mean, 
                 server_bn_var,
                 server_y_logits=None, 
                 distillation=False):
        """ Training n times (n=the epoch parameter)"""

        def list_add(a, b):
            assert len(a) == len(b)
            for k in range(len(a)):
                a[k] += b[k]
            return a

        def list_multiply_int(a, b):
            assert len(a) > 1
            for k in range(len(a)):
                a[k] *= b
            return a

        def list_divided_int(a, b):
            assert len(a) > 1
            for k in range(len(a)):
                a[k] /= b
            return a

        # split x and y at the beginning of each training iteration.
        accumulate_gradients = 0
        server_logit_vectors = None
        client_y_logits = {}
        count_for_labels = {}
        self.mean_loss = []
        self.accuracy = []
        self.distill_loss = []
        self.model.train()
        # training
        # following the algorithm 1 of the paper dataset distillation.
        # dataset distillation, fixed parameters
        select_record_client_class_prob = []
        select_record_client_logits = []
        client_bn_features = {}
        count_for_bn_features = {}

        for i in range(self.clients_training_epoch):
            # training_epoches = math.ceil(self.dataset.size // self.batch_size)
            # for _ in range(training_epoches):
            x, y, _ = self.dataset.next_batch(self.batch_size)
            x, y = x.to(self.device), y.to(self.device)
            if distillation:
                server_logit_vectors = self.match_labels_to_logits(server_y_logits, y)


            if i == 0:
                accumulate_gradients, client_logit_vectors, bn_features = self.train_step(x,
                                                                                        y,
                                                                                        server_bn_mean,
                                                                                        server_bn_var,
                                                                                        logit_vectors=server_logit_vectors,
                                                                                        distillation=distillation)
            else:
                tmp_gradients, client_logit_vectors, bn_features = self.train_step(x,
                                                                                y,
                                                                                server_bn_mean,
                                                                                server_bn_var,
                                                                                logit_vectors=server_logit_vectors,
                                                                                distillation=distillation)
                accumulate_gradients = list_add(accumulate_gradients,
                                                tmp_gradients)
            # x, y = x.to("cpu"), y.to("cpu")
            # # draw logits and class probabilities
            # if abs(current_epoches / max_epoches % 0.1 - 0) <= 0.00001:
            #     select_label = 1
            #     # draw class probabilities
            #     for idx, class_label_y in enumerate(y):
            #         if class_label_y == select_label:
            #             tmp_logits = F.softmax(client_logit_vectors[idx])
            #             select_record_client_class_prob.append(tmp_logits)
            #             select_record_client_logits.append(client_logit_vectors[idx])
            #             plt.plot(tmp_logits)
            #     np.savetxt("txt_result/tmpres/class_pb_{}.txt".format(current_epoches / max_epoches), np.array(select_record_client_class_prob))
            #     plt.title("class_pb")
            #     plt.ylabel("probability")
            #     plt.xlabel("class")
            #     # plt.legend()
            #     # plt.savefig("pic/asyn/loss_no_decay.eps")
            #     plt.savefig("pic/tmpres/class_pb_{}.png".format(current_epoches / max_epoches))
            #     plt.close()
                
            #     # draw logits
            #     for idx, class_label_y in enumerate(y):
            #         if class_label_y == select_label:
            #             plt.plot(client_logit_vectors[idx])

            #     np.savetxt("txt_result/tmpres/logits_{}.txt".format(current_epoches / max_epoches), np.array(select_record_client_logits))
            #     plt.title("logits")
            #     plt.ylabel("logits")
            #     plt.xlabel("class")
            #     # plt.legend()
            #     # plt.savefig("pic/asyn/loss_no_decay.eps")
            #     plt.savefig("pic/tmpres/logits_{}.png".format(current_epoches / max_epoches))
            #     plt.close()

                    # select_label_list.append(idx)
            # select_record_client_class_prob.append(client_logit_vectors[select_label_list])




            client_y_logits, count_for_labels = self.collect_logits(client_logit_vectors,
                                                                    y,
                                                                    client_y_logits,
                                                                    count_for_labels)
            client_bn_features, count_for_bn_features = self.collect_bn(client_bn_features,
                                                                        bn_features,
                                                                        count_for_bn_features)
            # self.scheduler.step()



        mean_loss = np.mean(np.array(self.mean_loss))
        
        acc = np.mean(np.array(self.accuracy))

        # updated_gradients = list_divided_int(accumulate_gradients, self.clients_training_epoch)
        # self.optimizer.apply_gradients(zip(updated_gradients,
        #                                    self.model.trainable_variables))

        client_y_logits = self.sort_logits(client_y_logits, count_for_labels)
        client_bn_features = self.get_avg_bn_features(client_bn_features, count_for_bn_features)


        if distillation:
            distill_loss = np.mean(np.array(self.distill_loss))
            if args.BN_features:
                bn_mean_loss = np.mean(np.array(self.bn_mean_loss))
                bn_var_loss = np.mean(np.array(self.bn_var_loss))
            else:
                bn_mean_loss, bn_var_loss = 0, 0
            return [mean_loss,
                    distill_loss,
                    acc,
                    list_divided_int(accumulate_gradients, self.clients_training_epoch),
                    client_y_logits,
                    client_bn_features,
                    bn_mean_loss,
                    bn_var_loss]
        else:
            return [mean_loss,
                    acc,
                    list_divided_int(accumulate_gradients, self.clients_training_epoch),
                    client_y_logits,
                    client_bn_features]

    def server_apply_gradient(self, gradients):
        for idx, param in enumerate(self.model.parameters()):
            param.grad = gradients[idx]
        self.optimizer.step()
        self.optimizer.zero_grad()

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

    def get_weights(self):
        with torch.no_grad():
            return self.model.parameters()

    def get_client_id(self):
        """ Return the client id """
        return self.cid


    # def set_model_parameters(self, server_model):
    #     """ Assign server model's parameters to this client """
    #     self.model.set_weights(server_model.get_model_parameters())

    def set_weights(self, weights):
        """ Assign server model's parameters to this client """
        with torch.no_grad():
            new_weights = [param.detach() for param in weights]
            for idx, param in enumerate(self.model.parameters()):
                param.copy_(new_weights[idx])
        
        # self.model.set_weights(weights)



