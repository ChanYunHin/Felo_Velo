import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os

# # fed_distill_pytorch/result/cmp/rn0.2_cifar10_50/Adamsr0.2_rn_MD_alpha0.5_lr0.001.csv
# df =pd.read_csv("result/cmp/rn0.2_cifar10_50/Adamsr0.2_rn_MD_alpha0.5_lr0.001.csv")


def method_result(fpath):
    df =pd.read_csv(fpath)
    max_val = 0
    max_idx = 0
    # print(len(df.columns))
    for idx in range(1, len(df.columns)):
        # print(df.columns[idx][20:30])
        if int(df.columns[idx][20:30]) > max_val:
            max_val = int(df.columns[idx][20:30])
            max_idx = idx

    # print(int(df.columns[2][20:30]))

    # print(max_val)
    # print(df.columns[max_idx])
    return list(df[df.columns[max_idx]])

def read_heteroFL(fpath):
    df =pd.read_csv(fpath)
    acc_list = []
    for idx, val in enumerate(df['Value']):
        if idx % 4 == 0:
            acc_list.append(val/100)
    print(len(acc_list))
    return acc_list


# acc_HeteroFL_list = read_heteroFL("result/cifar10HeteroFL-test_Global-Accuracy.csv")
acc_HeteroFL_list = read_heteroFL("result/FMNISTHeteroFL-test_Global-Accuracy.csv")

# print(acc_HeteroFL_list)


dataset_name = 'Fmnist'
# dataset_name = 'cifar10'
batch_size_num = '50'
exp_type = 'heter'

if exp_type == 'heter':
    exp_path = 'cmp'
    method_path = ''
    model_num = '5'
elif exp_type == 'homo':
    exp_path = 'cmp_homo'
    method_path = 'homo_'
    model_num = ''


acc_MD_list = method_result("result/{}/rn0.2_{}_{}/{}Adam{}sr0.2_rn_MD_alpha1.0_lr0.001.csv".format(exp_path, dataset_name, batch_size_num, method_path, model_num))
acc_GKT_list = method_result("result/{}/rn0.2_{}_{}/{}Adam{}sr0.2_rn_GKT_KLD_y_logits_alpha1.0_lr0.001.csv".format(exp_path, dataset_name, batch_size_num, method_path, model_num))
# acc_Private_list = method_result("result/{}/rn0.2_{}_{}/{}Adam{}sr0.2_rn_y_logits_alpha0.0_lr0.001.csv".format(exp_path, dataset_name, batch_size_num, method_path, model_num))
acc_FedHe_list = method_result("result/{}/rn0.2_{}_{}/{}Adam{}sr0.2_rn_y_logits_alpha1.0_lr0.001.csv".format(exp_path, dataset_name, batch_size_num, method_path, model_num))
acc_Felo_list = method_result("result/{}/rn0.2_{}_{}/{}Adam{}sr0.2_rn_op1024_KLD_y_logits_alpha1.0_lr0.001.csv".format(exp_path, dataset_name, batch_size_num, method_path, model_num))
acc_Velo_list = method_result("result/{}/rn0.2_{}_{}/{}Adam{}sr0.2_rn_new_VAE1024_KLD_y_logits_alpha1.0_lr0.001.csv".format(exp_path, dataset_name, batch_size_num, method_path, model_num))


print("FedMD top-1 acc: {}".format(max(acc_MD_list)))
print("GKT top-1 acc: {}".format(max(acc_GKT_list)))
# print("Private top-1 acc: {}".format(max(acc_Private_list)))
print("FedHe top-1 acc: {}".format(max(acc_FedHe_list)))
print("Felo top-1 acc: {}".format(max(acc_Felo_list)))
print("Velo top-1 acc: {}".format(max(acc_Velo_list)))
print("HeteroFL top-1 acc: {}".format(max(acc_HeteroFL_list)))


# plt.plot(acc_MD_list, label="FedMD")
# plt.plot(acc_GKT_list, label="GKT")
# plt.plot(acc_HeteroFL_list, label="HeteroFL")
# plt.plot(acc_FedHe_list, label="FedHe")
# plt.plot(acc_Felo_list, label="Felo")
# plt.plot(acc_Velo_list, label="Velo")
# # plt.plot(evaluate_acc_FedAvg_list, label="FedAvg")
# # plt.plot(evaluate_acc_FedHe01_list, label="FedHe 0.001")
# # plt.plot(evaluate_acc_FedHe001_list, label="FedHe")
# # plt.plot(evaluate_acc_FedHe07_list, label="FedHe 0.7")
# # plt.plot(evaluate_acc_FedMD, label="FedMD")
# plt.title("Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Number of iterations")
# if dataset_name == 'cifar10':
#     plt.ylim(0.4, 0.7)
# elif dataset_name == 'Fmnist':
#     plt.ylim(0.0, 0.95)
# plt.legend()
# plt.grid(linestyle="dotted")
# plt.savefig("compared_pic/acc_{}_{}_{}.eps".format(exp_type, dataset_name, batch_size_num))
# plt.savefig("compared_pic/acc_{}_{}_{}.png".format(exp_type, dataset_name, batch_size_num))
# plt.close()




# def read_felosize(fpath):
#     df =pd.read_csv(fpath)
#     acc_list = []
#     for idx, val in enumerate(df['Value']):
#         acc_list.append(val)
#     print(len(acc_list))
#     return acc_list

# acc_1024_list = read_felosize("result/ftsize/run-Adam5sr0.2_rn_op{}_KLD_y_logits_alpha1.0_lr0.001_test_accuracy_beta10_acc-tag-accuracy.csv".format(1024))
# acc_4096_list = read_felosize("result/ftsize/run-Adam5sr0.2_rn_op{}_KLD_y_logits_alpha1.0_lr0.001_test_accuracy_beta10_acc-tag-accuracy.csv".format(4096))
# acc_16384_list = read_felosize("result/ftsize/run-Adam5sr0.2_rn_op{}_KLD_y_logits_alpha1.0_lr0.001_test_accuracy_beta10_acc-tag-accuracy.csv".format(16384))

# plt.plot(acc_1024_list, label="Felo_1024")
# plt.plot(acc_4096_list, label="Felo_4096")
# plt.plot(acc_16384_list, label="Felo_16384")
# # plt.plot(evaluate_acc_FedAvg_list, label="FedAvg")
# # plt.plot(evaluate_acc_FedHe01_list, label="FedHe 0.001")
# # plt.plot(evaluate_acc_FedHe001_list, label="FedHe")
# # plt.plot(evaluate_acc_FedHe07_list, label="FedHe 0.7")
# # plt.plot(evaluate_acc_FedMD, label="FedMD")
# plt.title("Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Number of iterations")
# plt.ylim(0.5, 0.7)
# plt.legend()
# plt.grid(linestyle="dotted")
# # plt.savefig("compared_pic/acc_{}_{}_{}.eps".format(exp_type, dataset_name, batch_size_num))
# plt.savefig("compared_pic/acc_ft_felo.png")
# plt.close()


# def read_felosr(fpath):
#     df =pd.read_csv(fpath)
#     acc_list = []
#     for idx, val in enumerate(df['Value']):
#         acc_list.append(val)
#     print(len(acc_list))
#     return acc_list

# acc_02_list = read_felosr("result/sr/Felo0{}_cifar10.csv".format(2))
# acc_04_list = read_felosr("result/sr/Felo0{}_cifar10.csv".format(4))
# acc_06_list = read_felosr("result/sr/Felo0{}_cifar10.csv".format(6))
# acc_08_list = read_felosr("result/sr/Felo0{}_cifar10.csv".format(8))
# acc_10_list = read_felosr("result/sr/Felo{}_cifar10.csv".format(10))

# plt.plot(acc_02_list, label="Sample ratio=0.2")
# plt.plot(acc_04_list, label="Sample ratio=0.4")
# plt.plot(acc_06_list, label="Sample ratio=0.6")
# plt.plot(acc_08_list, label="Sample ratio=0.8")
# plt.plot(acc_10_list, label="Sample ratio=1.0")
# # plt.plot(evaluate_acc_FedAvg_list, label="FedAvg")
# # plt.plot(evaluate_acc_FedHe01_list, label="FedHe 0.001")
# # plt.plot(evaluate_acc_FedHe001_list, label="FedHe")
# # plt.plot(evaluate_acc_FedHe07_list, label="FedHe 0.7")
# # plt.plot(evaluate_acc_FedMD, label="FedMD")
# plt.title("Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Number of iterations")
# plt.ylim(0, 0.7)
# plt.legend()
# plt.grid(linestyle="dotted")
# # plt.savefig("compared_pic/acc_{}_{}_{}.eps".format(exp_type, dataset_name, batch_size_num))
# plt.savefig("compared_pic/acc_sr_felo.png")
# plt.close()


