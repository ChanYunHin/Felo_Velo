import argparse


parser = argparse.ArgumentParser(description='parameters')

# -- Model parameters --
parser.add_argument('--clients_number', type=int, default=10,
                    help='default: 10 clients')
parser.add_argument('--clients_sample_ratio', type=float, default=1,
                    help='default: 0.1')
parser.add_argument('--epoch', type=int, default=100,
                    help='default: 500')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='default: 0.001')
parser.add_argument('--decay_rate', type=float, default=1,
                    help='default: 1')
parser.add_argument('--batch_size', type=int, default=50,
                    help='default: 50')
parser.add_argument('--clients_training_epoch', type=int, default=10,
                    help='default: 3')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='default: 0.5')
parser.add_argument('--beta', type=float, default=10,
                    help='default: 10')
parser.add_argument('--cita', type=float, default=1,
                    help='default: 1')
# The default setting should be changed to run this file in sh files.
parser.add_argument('--moment_flag', action='store_true', 
                    help='Moment gradient descent or not')
parser.add_argument('--personalized_flag', action='store_true', 
                    help='Personalized y_logits or not')


# dataset parameters
parser.add_argument('--dataset', default="cifar10",
                    help='default: cifar10')
parser.add_argument('--num_classes', type=int, default=10,
                    help='default: 10')

# If you set --distorted_data in your command, it means that you distort data.
parser.add_argument("--distorted_data", action='store_true', help='distort data or not')
parser.add_argument("--aggregate_flag", action='store_true', help='aggregate the logits or not')
# parser.add_argument("--aggregate_flag", action='store_true', default=True, help='aggregate the logits or not')
parser.add_argument("--delayed_gradients_flag", action='store_true', help='deal with delayed gradients with Hessian or not')
parser.add_argument("--softmax_flag", action='store_true', help='compute the distillation loss in softmax or not')
parser.add_argument("--KLD_flag", action='store_true', help='distillation loss computed in KL divergence or not')
parser.add_argument("--delayed_gradients_divided_flag", action='store_true', help='deal with delayed gradients with devided gradients or not')
parser.add_argument("--noniid_flag", action='store_true', help='Non-iid for the training datasets or not')
# parser.add_argument("--noniid_flag", action='store_true', default=True, help='Non-iid for the training datasets or not')
parser.add_argument("--public_flag", action='store_true', help='Public datasets or not')
# parser.add_argument("--public_flag", action='store_true', default=True, help='Public datasets or not')



# debug parameters
parser.add_argument("--homo_flag", action='store_true', help='all the client models are homogeneous or not')
# parser.add_argument("--homo_flag", action='store_true', default=True, help='all the client models are homogeneous or not')
parser.add_argument("--VAE", action='store_true', help='Using VAE to generate features or not')
# parser.add_argument("--VAE", action='store_true', default=True, help='Using VAE to generate features or not')
parser.add_argument("--resnet_flag", action='store_true', 
                    help='Using Resnet or not')
# parser.add_argument("--resnet_flag", action='store_true', 
                    # default=True, help='Using Resnet or not')
parser.add_argument('--y_distillation_flag', action='store_true',
                    help='distilled y logits or not')
# parser.add_argument('--y_distillation_flag', action='store_true', default=True,
                    # help='distilled y logits or not')
parser.add_argument('--asyn_FL_flag', action='store_true',
                    help='Federated Learning or not')
# parser.add_argument('--asyn_FL_flag', action='store_true', default=True,
                    # help='Federated Learning or not')
parser.add_argument('--old_CNN', action='store_true',
                    help='Use old CNN or not')






parser.add_argument("--middle_flag", action='store_true', help='get the outputs from middle layer or not')
# parser.add_argument("--middle_flag", action='store_true', default=True, help='get the outputs from middle layer or not')
parser.add_argument("--baseline_middle_flag", action='store_true', help='block the outputs from middle layer or not')
parser.add_argument("--GPU_num", type=int, default=0, help='default: 1')
parser.add_argument("--BN_features", action='store_true', help='Using bn features or not')
parser.add_argument("--Var", action='store_true', help='Using bn features or not')
# parser.add_argument("--Var", action='store_true', default=True, help='Using bn features or not')
parser.add_argument("--var_para", type=float, default=1., help='The trade-off param for the logit variance')
parser.add_argument("--gradient_penalty", action='store_true', help='pre training or not')
parser.add_argument("--pre_train", action='store_true', help='pre training or not')
# parser.add_argument("--pre_train", action='store_true', default=True, help='pre training or not')
parser.add_argument("--GAN_moment_flag", action='store_true', help='Using the multivariate Gaussian to generate logits or not')
parser.add_argument("--second_highest", action='store_true', help='Using the second_highest logits to cluster or not')
# parser.add_argument("--second_highest", action='store_true', default=True, help='Using the second_highest logits to cluster or not')


parser.add_argument("--Similarity", action='store_true', help='Using the similar logits or not')
# parser.add_argument("--Similarity", action='store_true', default=True, help='Using the similar logits or not')
# parser.add_argument("--topK", type=int, default=2, help='default: 2')


parser.add_argument("--latent_size", type=int, default=1024, help='default: 1024')
# parser.add_argument("--one_classifer_flag", action="store_true", help="Using Linear classifier or not")
parser.add_argument("--one_classifer_flag", action="store_true", default=True, help="Using Linear classifier or not")
parser.add_argument("--half_flag", action="store_true", help="training in a half of the whole procedure or not")
# parser.add_argument("--half_flag", action="store_true", default=True, help="training in a half of the whole procedure or not")


args = parser.parse_args()
