
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
from modules.ResnetModel import SmallResnet1, SmallResnet2, SmallResnet3, Resnet50, SmallResnet50



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

        self.input_shape = input_shape
        self.device = device

        if neural_network_shape["model_type"] == "resnet":
            # self.model = Resnet34(num_classes=num_classes)
            self.model = Resnet50(in_channels=input_shape[1], 
                                    num_classes=num_classes)
        elif neural_network_shape["model_type"] == "small_resnet":
            # self.model = Resnet18(num_classes=num_classes)
            self.model = SmallResnet50(in_channels=input_shape[1], 
                                        num_classes=num_classes)
        elif neural_network_shape["model_type"] == "small_resnet_1":
            # self.model = Resnet18(num_classes=num_classes)
            self.model = SmallResnet1(in_channels=input_shape[1], 
                                        num_classes=num_classes)
        elif neural_network_shape["model_type"] == "small_resnet_2":
            # self.model = Resnet18(num_classes=num_classes)
            self.model = SmallResnet2(in_channels=input_shape[1], 
                                        num_classes=num_classes)
        elif neural_network_shape["model_type"] == "small_resnet_3":
            # self.model = Resnet18(num_classes=num_classes)
            self.model = SmallResnet3(in_channels=input_shape[1], 
                                        num_classes=num_classes)


        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)
        

        
        total_epoches = args.epoch * args.clients_sample_ratio * args.clients_number
        
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        
        
        self.loss_func = F.cross_entropy
        self.distill_loss_func_logits = F.kl_div
        self.distill_loss_func_features = F.mse_loss
        self.mean_loss = []
        self.accuracy = []
        self.distill_loss, self.feature_loss = [], []
        self.bn_mean_loss = []
        self.bn_var_loss = []

        # initialize
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.cid = client_id
        self.dataset = dataset
        self.is_worker = is_worker
        self.distorted_data = distorted_data
        self.batch_size = batch_size
        self.clients_training_epoch = clients_training_epoch

    

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
        assert self.is_worker
        
        alpha = args.alpha
        predictions = 0
        features_predictions = 0
        if distillation:
            # assert logit_vectors
            if args.softmax_flag or args.KLD_flag:
                softmax_logit_vectors = F.softmax(logit_vectors)
                log_softmax_predictions = 0
        """ one training iteration """
        self.model.train()
        encoded_features = self.model.encode(x)
        predictions = self.model.decode(encoded_features)
        pred_loss = self.loss_func(predictions, y)
        if distillation:
            if args.softmax_flag or args.KLD_flag:
                log_softmax_predictions = F.log_softmax(predictions)
                distill_loss = self.distill_loss_func_logits(log_softmax_predictions, 
                                                             softmax_logit_vectors)
            else:
                distill_loss = self.distill_loss_func_logits(predictions, 
                                                             logit_vectors)
            feature_extractor_loss = self.distill_loss_func_features(encoded_features,
                                                                     server_features)
            features_loss = feature_extractor_loss
            if args.no_logits_flag:
                pred_loss = pred_loss + alpha * (features_loss)
            else:
                pred_loss = pred_loss + alpha * (features_loss + distill_loss)
            
        

        pred_loss.backward()
        
        
        
        self.optimizer.step()
        
        gradients = [param.grad.detach() for param in self.model.parameters()]
        self.optimizer.zero_grad()
        
        if not no_record:
            self.accuracy.append((y.eq(predictions.max(dim=1).indices).sum() / y.shape[0]).item())
            
            if distillation:
                self.mean_loss.append(pred_loss.item() - alpha * features_loss.item())
                
            else:
                self.mean_loss.append(pred_loss.item())
        if distillation:
            self.distill_loss.append(distill_loss.item())
            self.feature_loss.append(features_loss.item())
        
        training_result = [gradients, 
                            predictions.detach(), 
                            encoded_features.detach()]
        return training_result

    
        
        
    def collect_logits_features(self, client_logit_vectors, 
                                   labels, logit_vectors, 
                                   count_for_labels, client_features,
                                   features_dict):
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
        return logit_vectors, count_for_labels, features_dict
    
    def match_labels_to_vectors(self, 
                                server_vectors, 
                                labels):
        server_labeled_vectors = []
        # match logit vectors for each label
        for idx, label in enumerate(labels):
            server_labeled_vectors.append(server_vectors[label.item()])
        return torch.stack(server_labeled_vectors, dim=0)


    def get_logits_from_generator(self, generator, batch_size, y):
        noise = torch.randn(batch_size, self.num_classes, device=self.device)
        generated_logits = generator(noise, y)
        return generated_logits

    def get_features_from_generator(self, generator, 
                                    batch_size,
                                    latent_size, random_seed):
        # set a random seed manually, to reproduce the same features for all clients
        torch.manual_seed(random_seed)
        generated_label = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        noise = torch.randn(batch_size, latent_size, device=self.device)
        generated_features = generator.generate_data(noise, generated_label)
        # reset the random seed
        torch.seed()
        return generated_features, generated_label

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

    


    def training(self, 
                 server_generator=None, 
                 distillation=False,
                 **kwargs):
        """ Training n times (n=the epoch parameter)"""

        server_features = kwargs["server_features"]
        server_y_logits = kwargs["server_y_logits"]

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
        server_logit_vectors, server_feature_vectors = None, None
        client_y_logits = {}
        count_for_labels = {}
        self.mean_loss = []
        self.accuracy = []
        self.distill_loss, self.feature_loss = [], []
        features_dict = {}
        self.model.train()
        

        for i in range(self.clients_training_epoch):
            
            x, y, data_idx = self.dataset.next_batch(self.batch_size)
            x, y = x.to(self.device), y.to(self.device)
            if distillation:
                server_logit_vectors = self.match_labels_to_vectors(server_y_logits, y)
                server_feature_vectors = self.match_labels_to_vectors(server_features, y)

               
            train_result = self.train_step(x,
                                        y,
                                        logit_vectors=server_logit_vectors,
                                        distillation=distillation,
                                        server_features=server_feature_vectors)

            tmp_gradients, client_logit_vectors, client_features = train_result[0], train_result[1], train_result[2]

                            
            accumulate_gradients = list_add(accumulate_gradients,
                                            tmp_gradients)
            
            client_y_logits, count_for_labels, features_dict = \
                self.collect_logits_features(client_logit_vectors,
                                            y,
                                            client_y_logits,
                                            count_for_labels,
                                            client_features,
                                            features_dict)



        mean_loss = np.mean(np.array(self.mean_loss))
        
        acc = np.mean(np.array(self.accuracy))

        

        client_y_logits = self.sort_logits(client_y_logits, count_for_labels)

        if distillation:
            distill_loss = np.mean(np.array(self.distill_loss))
            feature_loss = np.mean(np.array(self.feature_loss))
            
            return {"CE_loss": mean_loss,
                    "distill_loss": distill_loss,
                    "feature_loss": feature_loss,
                    "Acc": acc,
                    "gradients": list_divided_int(accumulate_gradients, self.clients_training_epoch),
                    "y_logits": client_y_logits,
                    "features_dict": features_dict}
        else:
            return {"CE_loss": mean_loss,
                    "Acc": acc,
                    "gradients": list_divided_int(accumulate_gradients, self.clients_training_epoch),
                    "y_logits": client_y_logits,
                    "features_dict": features_dict}

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

    

    def get_weights(self):
        """ Return the model's parameters """
        return self.model.get_weights()



    def set_weights(self, weights):
        """ Assign server model's parameters to this client """
        self.model.set_weights(weights)



