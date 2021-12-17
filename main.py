
import os
from collections import namedtuple
import torch

from parameters import args


if args.Var or args.second_highest or args.Similarity:
    from modules.Server_Var import Server as server
elif args.VAE:
    from modules.Server_VAE import Server as server
else:
    from modules.Server import Server as server

def main(device):
    parameters = namedtuple('parameters', ['clients_number',
                                           'clients_sample_ratio',
                                           'epoch',
                                           'learning_rate',
                                           'decay_rate',
                                           'num_input',
                                           'num_input_channel',
                                           'num_classes',
                                           'batch_size',
                                           'clients_training_epoch',
                                           'distorted_data',
                                           'dataset',
                                           'device' ])
    
    clients_number = args.clients_number
    clients_sample_ratio = args.clients_sample_ratio
    epoch = args.epoch
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    num_input_channel = 3
    num_classes = args.num_classes
    batch_size = args.batch_size
    clients_training_epoch = args.clients_training_epoch
    distorted_data = args.distorted_data
    dataset = args.dataset
    if distorted_data:        
        num_input = 24
    else:
        num_input = 32
    if dataset == "mnist":
        num_input = 28
        distorted_data = False
        num_input_channel = 1
    parameters = parameters(clients_number,
                            clients_sample_ratio,
                            epoch,
                            learning_rate,
                            decay_rate,
                            num_input,
                            num_input_channel,
                            num_classes,
                            batch_size,
                            clients_training_epoch,
                            distorted_data,
                            dataset,
                            device)

    Server = server(parameters)
    Server.run()




if __name__ == '__main__':

    # prohibit GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_num)
    print(torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    main(dev)
