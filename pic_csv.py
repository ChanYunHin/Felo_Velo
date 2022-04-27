import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os

# fed_distill_pytorch/result/cmp/rn0.2_cifar10_50/Adamsr0.2_rn_MD_alpha0.5_lr0.001.csv
df =pd.read_csv("result/cmp/rn0.2_cifar10_50/Adamsr0.2_rn_MD_alpha0.5_lr0.001.csv")


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


dataset_name = 'Fmnist'
batch_size_num = '50'
exp_type = 'homo'

if exp_type == 'heter':
    exp_path = 'cmp'
    method_path = ''
elif exp_type == 'homo':
    exp_path = 'cmp_homo'
    method_path = 'homo_'

acc_MD_list = method_result("result/{}/rn0.2_{}_{}/{}Adamsr0.2_rn_MD_alpha1.0_lr0.001.csv".format(exp_path, dataset_name, batch_size_num, method_path))
acc_GKT_list = method_result("result/{}/rn0.2_{}_{}/{}Adamsr0.2_rn_GKT_KLD_y_logits_alpha1.0_lr0.001.csv".format(exp_path, dataset_name, batch_size_num, method_path))
acc_Private_list = method_result("result/{}/rn0.2_{}_{}/{}Adamsr0.2_rn_y_logits_alpha0.0_lr0.001.csv".format(exp_path, dataset_name, batch_size_num, method_path))
acc_FedHe_list = method_result("result/{}/rn0.2_{}_{}/{}Adamsr0.2_rn_y_logits_alpha1.0_lr0.001.csv".format(exp_path, dataset_name, batch_size_num, method_path))
acc_Felo_list = method_result("result/{}/rn0.2_{}_{}/{}Adamsr0.2_rn_op_KLD_y_logits_alpha1.0_lr0.001.csv".format(exp_path, dataset_name, batch_size_num, method_path))
acc_Velo_list = method_result("result/{}/rn0.2_{}_{}/{}Adamsr0.2_rn_new_VAE_KLD_y_logits_alpha1.0_lr0.001.csv".format(exp_path, dataset_name, batch_size_num, method_path))

plt.plot(acc_MD_list, label="FedMD")
plt.plot(acc_GKT_list, label="GKT")
plt.plot(acc_Private_list, label="Private")
plt.plot(acc_FedHe_list, label="FedHe")
plt.plot(acc_Felo_list, label="Felo")
plt.plot(acc_Velo_list, label="Velo")
# plt.plot(evaluate_acc_FedAvg_list, label="FedAvg")
# plt.plot(evaluate_acc_FedHe01_list, label="FedHe 0.001")
# plt.plot(evaluate_acc_FedHe001_list, label="FedHe")
# plt.plot(evaluate_acc_FedHe07_list, label="FedHe 0.7")
# plt.plot(evaluate_acc_FedMD, label="FedMD")
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Number of iterations")
if dataset_name == 'cifar10':
    plt.ylim(0.5, 0.7)
elif dataset_name == 'Fmnist':
    plt.ylim(0.8, 0.92)
plt.legend()
plt.grid(linestyle="dotted")
plt.savefig("compared_pic/acc_{}_{}_{}.eps".format(exp_type, dataset_name, batch_size_num))
plt.savefig("compared_pic/acc_{}_{}_{}.png".format(exp_type, dataset_name, batch_size_num))
plt.close()
