
import math
import random
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
from modules.ResnetModel import Resnet34, Resnet18, Resnet50_ft, SmallResnet50_ft

# import modules.GAN_model as CGAN

# # The definition of fed model
# FedModel = namedtuple('FedModel', 'X Y DROP_RATE train_op loss_op acc_op')




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
            self.model = ClientModel(input_shape, num_classes)
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
                self.model = Resnet50_ft(in_channels=input_shape[1], 
                                      num_classes=num_classes)
            elif neural_network_shape["model_type"] == "small_resnet":
                # self.model = Resnet18(num_classes=num_classes)
                self.model = SmallResnet50_ft(in_channels=input_shape[1], 
                                           num_classes=num_classes)

        # # If you don't have this function, you would not have true weights in the network,
        # # because you only get your weights when data pass through the network.
        # self.create_weights_for_model(self.model)

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)
        
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
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                                    factor=args.scheduler_factor, patience=args.scheduler_patience,
                                                                    threshold=args.scheduler_threshold)
        
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.scheduler_step, gamma=args.scheduler_factor)
        
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
    

    def CE_KL_loss_function(self, predictions, y, mean, logvar):
        """
        predictions: generating images
        y: origin images
        mean: latent mean
        var: latent variance
        """
        CE = self.loss_func(predictions, y) 
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mean.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        # KL divergence
        return CE + KLD

    def train_step(self, x, y,
                   logit_vectors=None, 
                   distillation=False, 
                   no_record=False,
                   **kwargs):

        server_features = kwargs["server_features"]
        client_features_dataset = kwargs["client_features_dataset"]
        training_encoded_features_inputs = client_features_dataset[0]
        training_encoded_features = client_features_dataset[1]
        
        # server_feature_logits = kwargs["server_feature_logits"]
        assert self.is_worker
        # alpha = args.alpha * (np.exp(current_epoches / max_epoches) - 1)
        alpha = args.alpha
        predictions = 0
        # features_predictions = 0
        if distillation:
            # assert logit_vectors
            if args.softmax_flag or args.KLD_flag:
                softmax_logit_vectors = F.softmax(logit_vectors)
                log_softmax_predictions = 0
        """ one training iteration """
        
        self.model.train()
        
        encoded_feature_inputs = self.model.encode(x)
        encoded_features = self.model.decode(encoded_feature_inputs)
        predictions = self.model.fc_layers(encoded_features)
        
        # solve the process of dealing with encoded features
        
        pred_loss = self.loss_func(predictions, y)
        
        # pred_loss.backward()
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        
        if distillation:
            if args.softmax_flag or args.KLD_flag:
                # if args.MSE_flag:
                #     distill_loss = self.distill_loss_func(softmax_logit_vectors, softmax_predictions)
                # else:
                log_softmax_predictions = F.log_softmax(predictions)
                distill_loss = self.distill_loss_func_logits(log_softmax_predictions, softmax_logit_vectors)
            else:
                distill_loss = self.distill_loss_func_logits(predictions, 
                                                             logit_vectors)
            
            training_features_predictions = self.model.decode(training_encoded_features_inputs)
            features_loss = self.distill_loss_func_features(training_encoded_features, training_features_predictions)
            # features_loss = features_loss
            if args.no_logits_flag:
                pred_loss = pred_loss + alpha * (features_loss)
                
                
                # pred_loss = features_loss
                # pred_loss.backward()
                # self.optimizer.step()
                # self.optimizer.zero_grad()
                # pred_loss = pred_loss
            else:
                print("wrong updating process")
                assert 1==0
                # pred_loss = pred_loss + alpha * (features_loss + distill_loss)
        #     pred_loss = pred_loss + alpha * distill_loss

        # a = [torch.mean(mod.bn_feature_mean) for mod in self.bn_feature_layers]
        # debug_item = torch.tensor(a, device=self.device)

        pred_loss.backward()
        
        # torch.nn.utils.clip_grad.clip_grad_value_(self.model.parameters(), 0.1)
        
        self.optimizer.step()
        
        # gradients = [param.grad.detach() for param in self.model.parameters() if param.grad is not None]
        gradients = []
        # tmp_param = [param for param in self.model.parameters()]
        self.optimizer.zero_grad()
        # self.scheduler.step(pred_loss.item())
        
        
        # set up the accuracy in pytorch
        if not no_record:
            self.accuracy.append((y.eq(predictions.max(dim=1).indices).sum() / y.shape[0]).item())
            # self.accuracy.update_state(y, F.softmax(predictions))
            self.mean_loss.append(pred_loss.item())
        if distillation:
            self.distill_loss.append(distill_loss.item())
            self.feature_loss.append(features_loss.item())
        
            training_result = [gradients, 
                               predictions.detach(), 
                               encoded_features.detach(),
                               encoded_feature_inputs.detach()]
            
                                # features_predictions.detach()]
        else:
            training_result = [gradients, 
                                predictions.detach(), 
                                encoded_features.detach(),
                                encoded_feature_inputs.detach()]
        return training_result

        
    def collect_logits_features(self, client_logit_vectors, 
                                   labels, logit_vectors, 
                                   count_for_labels, client_features,
                                   features_dict,
                                   features_dataset,
                                   client_feature_inputs=None):
        for idx, logit_y in enumerate(client_logit_vectors):
            label = labels[idx].item()
            if label not in logit_vectors:
                logit_vectors[label] = logit_y
                count_for_labels[label] = 1
                # client_features 0 is mean, 1 is var
                features_dict[label] = [client_features[idx]]
            else:
                logit_vectors[label] += logit_y
                count_for_labels[label] += 1
                features_dict[label].append(client_features[idx])
                
        assert client_feature_inputs.shape[0] == client_features.shape[0]
        # 0 is client_feature_inputs, 1 is client_features
        
        for idx in range(client_feature_inputs.shape[0]):
            # reduce the size of the feature dataset
            # the random number of rand between [0, 1]
            if np.random.rand() < 0.01:
                features_dataset.append([client_feature_inputs[idx, :], client_features[idx, :]])
            # else:
            #     del client_feature_inputs[idx, :]
            #     del client_features[idx, :]
        return logit_vectors, count_for_labels, features_dict, features_dataset

    def match_labels_to_vectors(self, 
                                server_vectors, 
                                labels):
        # labels = self.get_class_labels(one_hot_y)
        server_labeled_vectors = []
        # match logit vectors for each label
        for idx, label in enumerate(labels):
            server_labeled_vectors.append(server_vectors[label.item()])
        # a = np.array(server_labeled_vectors)
        return torch.stack(server_labeled_vectors, dim=0).to(self.device)

    def get_training_encoded_features(self, client_features_dataset, batch_size):
        idx_list = list(range(len(client_features_dataset)))
        random_list = random.sample(idx_list, batch_size)
        x = [val[0] for idx, val in enumerate(client_features_dataset) if idx in random_list]
        y = [val[1] for idx, val in enumerate(client_features_dataset) if idx in random_list]
        return torch.stack(x, dim=0).to(self.device), torch.stack(y, dim=0).to(self.device)

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
                 distillation=False,
                 **kwargs):
        """ Training n times (n=the epoch parameter)"""

        random_seed = kwargs["random_seed"]
        latent_size = kwargs["latent_size"]
        server_features = kwargs["server_features"]
        # server_generated_labels = kwargs["server_generated_labels"]
        # server_feature_logits = kwargs["server_feature_logits"]
        server_y_logits = kwargs["server_y_logits"]
        client_features_dataset = kwargs["client_features_dataset"]

        def list_add(a, b):
            if len(a) == 0:
                return b
            if len(b) == 0:
                return a
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
        accumulate_gradients = []
        server_logit_vectors, server_features_vectors = None, None
        client_y_logits = {}
        count_for_labels = {}
        self.mean_loss = []
        self.accuracy = []
        self.distill_loss, self.feature_loss = [], []
        features_dict = {}
        features_dataset = []
        self.model.train()
        # server_generated_labels = 0
        # server_features = 0
        # training
        # following the algorithm 1 of the paper dataset distillation.
        # dataset distillation, fixed parameters
        select_record_client_class_prob = []
        select_record_client_logits = []
        client_features_inputs = None
        encoded_features_inputs, encoded_features = None, None

        for i in range(self.clients_training_epoch):
            training_epoches = math.ceil(self.dataset.size // self.batch_size)
            for _ in range(training_epoches):
                x, y, data_idx = self.dataset.next_batch(self.batch_size)
                x, y = x.to(self.device), y.to(self.device)
                if distillation:
                    server_logit_vectors = \
                        self.match_labels_to_vectors(server_y_logits, 
                                                    y)
                    server_features_vectors = \
                        self.match_labels_to_vectors(server_features, 
                                                    y)
                    encoded_features_inputs, encoded_features = \
                        self.get_training_encoded_features(client_features_dataset, self.batch_size)


                # if i == 0:
                train_result = self.train_step(x,
                                                y,
                                                logit_vectors=server_logit_vectors,
                                                distillation=distillation,
                                                client_features_dataset=[encoded_features_inputs, encoded_features],
                                                server_features=server_features_vectors,)
                                                # server_feature_logits=server_feature_logits)

                tmp_gradients, client_logit_vectors, client_features = train_result[0], train_result[1], train_result[2]
                client_features_inputs = train_result[3]
                # if distillation:
                #     client_feature_logits = train_result[3]
                                
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



                # client_features_list.append([vae_mean, vae_var])
                client_y_logits, count_for_labels, features_dict, features_dataset = \
                    self.collect_logits_features(client_logit_vectors,
                                                y,
                                                client_y_logits,
                                                count_for_labels,
                                                client_features,
                                                client_feature_inputs=client_features_inputs, 
                                                features_dict=features_dict,
                                                features_dataset=features_dataset)


        mean_loss = np.mean(np.array(self.mean_loss))
        # self.scheduler.step(mean_loss)
        acc = np.mean(np.array(self.accuracy))

        # updated_gradients = list_divided_int(accumulate_gradients, self.clients_training_epoch)
        # self.optimizer.apply_gradients(zip(updated_gradients,
        #                                    self.model.trainable_variables))

        client_y_logits = self.sort_logits(client_y_logits, count_for_labels)

        if distillation:
            distill_loss = np.mean(np.array(self.distill_loss))
            feature_loss = np.mean(np.array(self.feature_loss))
            
            return {"CE_loss": mean_loss,
                    "distill_loss": distill_loss,
                    "feature_loss": feature_loss,
                    "Acc": acc,
                    # "gradients": list_divided_int(accumulate_gradients, self.clients_training_epoch),
                    "gradients": accumulate_gradients,
                    "y_logits": client_y_logits,
                    "features_dict": features_dict,
                    "client_features_dataset": features_dataset}
                    # "feature_logits": client_feature_logits}
        else:
            return {"CE_loss": mean_loss,
                    "Acc": acc,
                    # "gradients": list_divided_int(accumulate_gradients, self.clients_training_epoch),
                    "gradients": accumulate_gradients,
                    "y_logits": client_y_logits,
                    "features_dict": features_dict,
                    "client_features_dataset": features_dataset}

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

    # def get_weights(self):
    #     """ Return the model's parameters """
    #     return self.model.get_weights()

    def get_weights(self):
        with torch.no_grad():
            tmp_weights = [param.detach() for param in self.model.parameters()]
            return tmp_weights[0], tmp_weights[1], tmp_weights[-2], tmp_weights[-1]

    # def get_weights(self):
    #     with torch.no_grad():
    #         return self.model.parameters()
        
    # def set_weights(self, weights):
    #     """ Assign server model's parameters to this client """
    #     with torch.no_grad():
    #         new_weights = [param.detach() for param in weights]
    #         # print(new_weights[0][0][0])
    #         for idx, param in enumerate(self.model.parameters()):
    #             param.copy_(new_weights[idx])


    def set_weights(self, weights):
        """ Assign parameters to this model """
        with torch.no_grad():
            # last_weights = weights
            new_weights = [param for param in self.model.parameters()]
            new_weights[0].copy_(weights[0])
            new_weights[1].copy_(weights[1])
            new_weights[-2].copy_(weights[-2])
            new_weights[-1].copy_(weights[-1])
            # for idx, param in self.model.parameters():
                # param.copy_(new_weights[idx])
                # print(name, param.size())


    # def get_classifier_weights(self):
    #     with torch.no_grad():
    #         tmp_weights = [param.detach() for param in self.model.parameters()]
    #         return tmp_weights[-2], tmp_weights[-1]

    # def set_classifier_weights(self, weights, bias):
    #     """ Assign parameters to this model """
    #     with torch.no_grad():
    #         # last_weights = weights
    #         new_weights = [param for param in self.model.parameters()]
    #         new_weights[-2].copy_(weights)
    #         new_weights[-1].copy_(bias)
    #         # for idx, param in self.model.parameters():
    #             # param.copy_(new_weights[idx])
    #             # print(name, param.size())



