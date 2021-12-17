
import math
import numpy as np
import copy
import os
from numpy.core.fromnumeric import mean

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib

from modules.Model import ClientModel
from modules.Model import ClientModel_2CNN
from parameters import args 
from modules.utils import get_model_mask

# # The definition of fed model
# FedModel = namedtuple('FedModel', 'X Y DROP_RATE train_op loss_op acc_op')


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
            self.model = ClientModel(input_shape, num_classes)
        else:
            if neural_network_shape["model_type"] == "2_layer_CNN":
                self.model = ClientModel_2CNN(input_shape, 
                                              num_classes, 
                                              layer1 = neural_network_shape["params"]["n1"],
                                              layer2 = neural_network_shape["params"]["n2"],
                                              dropout_rate = neural_network_shape["params"]["dropout_rate"])
            else:
                self.model = ClientModel(input_shape, 
                                         num_classes, 
                                         layer1 = neural_network_shape["params"]["n1"],
                                         layer2 = neural_network_shape["params"]["n2"],
                                         layer3 = neural_network_shape["params"]["n3"],
                                         dropout_rate = neural_network_shape["params"]["dropout_rate"])
        # # If you don't have this function, you would not have true weights in the network,
        # # because you only get your weights when data pass through the network.
        # self.create_weights_for_model(self.model)

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=learning_rate)
        
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
                   logit_vectors=None, 
                   distillation=False, 
                   no_record=False):
        assert self.is_worker
        if args.second_highest and distillation:
            second_highest_logits = logit_vectors[1]
            logit_vectors = logit_vectors[0]
        # alpha = args.alpha * (np.exp(current_epoches / max_epoches) - 1)
        alpha = args.alpha
        predictions = 0
        if distillation:
            # assert logit_vectors
            if args.softmax_flag or args.KLD_flag:
                softmax_logit_vectors = F.softmax(logit_vectors)
                softmax_predictions = 0

        """ one training iteration """
        predictions = self.model(x)
        pred_loss = self.loss_func(predictions, y)
        if distillation:
            if args.softmax_flag or args.KLD_flag:
                # if args.MSE_flag:
                #     distill_loss = self.distill_loss_func(softmax_logit_vectors, softmax_predictions)
                # else:
                softmax_predictions = F.softmax(predictions)
                distill_loss = self.distill_loss_func(softmax_predictions, softmax_logit_vectors)
            else:

                if args.second_highest:
                    pred_second_high_idx = torch.topk(predictions, k=2, dim=1).indices[:, 1].view(-1, 1)
                    new_server_logits = []
                    for idx, val in enumerate(pred_second_high_idx):
                        tmpval = second_highest_logits[idx][int(val.cpu())]
                        if isinstance(tmpval, int):
                            new_server_logits.append(logit_vectors[idx])
                        else:
                            new_server_logits.append(tmpval)
                    new_server_logits = torch.stack(new_server_logits, dim=0)
                    distill_loss = self.distill_loss_func(predictions, 
                                                          new_server_logits)
                elif args.Similarity:
                    new_server_logits = []
                    for idx, lg_val in enumerate(logit_vectors):
                        if isinstance(lg_val, int):
                            new_server_logits.append(predictions[idx])
                        else:
                            new_server_logits.append(lg_val)
                    if len(logit_vectors) > 0:
                        new_server_logits = torch.stack(new_server_logits, dim=0)
                        distill_loss = self.distill_loss_func(predictions, 
                                                            new_server_logits)
                    else:
                        distill_loss = self.distill_loss_func(predictions, 
                                                              predictions)

                else:
                    distill_loss = self.distill_loss_func(predictions, 
                                                        logit_vectors)
            if distill_loss.item() == 0:
                pred_loss = pred_loss
            else:
                pred_loss = (1 - alpha) * pred_loss + alpha * distill_loss

        # a = [torch.mean(mod.bn_feature_mean) for mod in self.bn_feature_layers]
        # debug_item = torch.tensor(a, device=self.device)
        
        pred_loss.backward()
        self.optimizer.step()
        gradients = [copy.deepcopy(param.grad.detach()) for param in self.model.parameters()]
        self.optimizer.zero_grad()
        
        # set up the accuracy in pytorch
        if not no_record:
            self.accuracy.append((y.eq(predictions.max(dim=1).indices).sum() / y.shape[0]).item())
            # self.accuracy.update_state(y, F.softmax(predictions))
            self.mean_loss.append(pred_loss.item())
        if distillation:
            self.distill_loss.append(distill_loss.item())
        if args.softmax_flag or args.KLD_flag:
            return gradients, \
                   F.softmax(predictions).detach()
        else:
            return gradients, \
                   predictions.detach()

    # def get_class_labels(self, one_hot_y):
    #     labels = [np.argmax(yi) for yi in one_hot_y]
    #     return labels

    def collect_logits(self, client_logit_vectors, labels, logit_vectors, count_for_labels):
        # one_hots to class labels
        # labels = self.get_class_labels(one_hot_y)
        for idx, logit_y in enumerate(client_logit_vectors):
            if labels[idx].item() not in logit_vectors:
                logit_vectors[labels[idx].item()] = [logit_y]
                count_for_labels[labels[idx].item()] = 1
            else:
                logit_vectors[labels[idx].item()].append(logit_y)
                count_for_labels[labels[idx].item()] += 1
        return logit_vectors, count_for_labels

    # def match_labels_to_logits(self, server_y_logits, labels):
    #     # labels = self.get_class_labels(one_hot_y)
    #     server_logit_vectors = []
    #     # match logit vectors for each label
    #     for idx, label in enumerate(labels):
    #         server_logit_vectors.append(np.array(server_y_logits[label.item()].cpu()))
    #     # a = np.array(server_logit_vectors)
    #     return np.array(server_logit_vectors)


    def get_all_server_logits(self, generator):
        all_server_logits = []
        for key, val in generator["client_per_data_logits"].items():
            if len(val) > 0:
                torch_val = torch.cat(val, dim=0)
                all_server_logits.extend(torch_val[:, :-1])
        all_server_logits = torch.stack(all_server_logits, dim=0)
        return all_server_logits


    def get_logits_from_generator(self, generator, 
                                  y, current_precent, 
                                  sample_idx = None,
                                  all_server_logits = None):
        # generate logits followed the normal distribution with mean and var from the server collected logits
        generated_logits = []
        server_second_high_logits = []
        # mean_list = []
        # var_list = []
        if args.Similarity:
            topK_number = args.topK
            # for cid, val in generator["dict_client_per_data_logits"].items():
            # bool(dict) is false if a dict is empty.
            if bool(generator["dict_client_per_data_logits"][self.cid]):
                for idx in sample_idx:
                    if idx in generator["dict_client_per_data_logits"][self.cid]:
                        lg_val = generator["dict_client_per_data_logits"][self.cid][idx]
                        tmp_res = torch.sum(torch.pow((lg_val - all_server_logits), 2), dim=1)
                        tmp_topK_idx = torch.topk(tmp_res, topK_number, largest=False)
                        a = torch.mean(all_server_logits[tmp_topK_idx.indices, :], dim=0)
                        generated_logits.append(a)
                    else:
                        generated_logits.append(0)



            # for idx in sample_idx:
            #     if idx in generator:
            #         generated_logits.append(generator[idx])
            #     else:
            #         generated_logits.append(0)
        else:
            if args.second_highest:
                second_high_generator = generator[1]
                logit_generator = generator[0]
            else:
                logit_generator = generator
            for label in y:
                generated_logits.append(logit_generator[int(label.cpu())])
                # var_list.append(args.var_para * statistics["var"])
                if args.second_highest:
                    server_second_high_logits.append(second_high_generator[int(label.cpu())])



        # if np.isclose(current_precent, 0.5):
        #     tmp_y = [5 for _ in range(10)]

        #     del mean_list
        #     mean_list = []
        #     var_list = []

            # for label in tmp_y:
            #     statistics = generator[label]
            #     mean_list.append(statistics["mean"])
            #     var_list.append(args.var_para * statistics["var"])


            # draw_generated_logits = torch.distributions.multivariate_normal.MultivariateNormal(
            #                             loc=torch.stack(mean_list, dim=0), 
            #                             covariance_matrix=torch.stack(var_list, dim=0),
            #                         ).sample()

            # draw_generated_logits = torch.normal(mean=torch.stack(mean_list, dim=0), 
            #                                      std=torch.stack(var_list, dim=0))


            # for idx, label in enumerate(tmp_y):
            #     if idx == 10:
            #         break
            #     plt.plot(draw_generated_logits[idx].cpu(), label=str(label))
            # plt.title("Generated samples")
            # plt.legend()
            # model_mask = get_model_mask(args)
            # pic_save_path = "pic/{}/{}/alpha{}_lr{}/".format(args.dataset, model_mask, args.alpha, args.learning_rate)
            # makedirs(pic_save_path)
            # plt.savefig("{}generated_samples.png".format(pic_save_path))
            # plt.close()
                
        if args.second_highest:
            return generated_logits, server_second_high_logits
        else:
            return generated_logits

    def sort_logits(self, 
                    client_y_logits, 
                    count_for_labels):

        for label, logit in client_y_logits.items():
            client_y_logits[label] = logit / count_for_labels[label]
        return client_y_logits

        
    # def tmp_save_pic(self, save_path):
    #     for i in range(self.distilled_x.shape[0]):
    #         tf.keras.preprocessing.image.save_img("{}/{}.png".format(save_path, i), self.distilled_x[i])

    def training(self, 
                 current_percent,
                 server_logits_generator=None, 
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
        data_idx_list, client_per_data_logits_list = [], []
        self.mean_loss = []
        self.accuracy = []
        self.distill_loss = []
        self.model.train()

        if args.Similarity and distillation:
            all_server_logits = self.get_all_server_logits(server_logits_generator)

        for i in range(self.clients_training_epoch):
            x, y, data_idx = self.dataset.next_batch(self.batch_size)
            data_idx_list.extend(data_idx)
            x, y = x.to(self.device), y.to(self.device)
            if distillation:
                if args.second_highest:
                    server_logit_vectors \
                        = self.get_logits_from_generator(server_logits_generator, y, current_percent)
                elif args.Similarity:
                    server_logit_vectors \
                        = self.get_logits_from_generator(server_logits_generator, 
                                                         y, 
                                                         current_percent, 
                                                         data_idx,
                                                         all_server_logits)
                else:
                    server_logit_vectors = self.get_logits_from_generator(server_logits_generator, y, current_percent)

            if i == 0:
                accumulate_gradients, client_logit_vectors = self.train_step(x,
                                                                             y,
                                                                             logit_vectors=server_logit_vectors,
                                                                             distillation=distillation)
            else:
                tmp_gradients, client_logit_vectors = self.train_step(x,
                                                                      y,
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



            client_per_data_logits_list.extend(client_logit_vectors)

            client_y_logits, count_for_labels = self.collect_logits(client_logit_vectors,
                                                                    y,
                                                                    client_y_logits,
                                                                    count_for_labels)



        mean_loss = np.mean(np.array(self.mean_loss))
        acc = np.mean(np.array(self.accuracy))
        client_per_data_logits_list = torch.stack(client_per_data_logits_list, dim=0)

        # updated_gradients = list_divided_int(accumulate_gradients, self.clients_training_epoch)
        # self.optimizer.apply_gradients(zip(updated_gradients,
        #                                    self.model.trainable_variables))

        # client_y_logits = self.sort_logits(client_y_logits, count_for_labels)


        if distillation:
            distill_loss = np.mean(np.array(self.distill_loss))
            return [mean_loss,
                    distill_loss,
                    acc,
                    list_divided_int(accumulate_gradients, self.clients_training_epoch),
                    client_y_logits,
                    client_per_data_logits_list,
                    data_idx_list]
        else:
            return [mean_loss,
                    acc,
                    list_divided_int(accumulate_gradients, self.clients_training_epoch),
                    client_y_logits,
                    client_per_data_logits_list,
                    data_idx_list]

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



