import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# from numpy.core.defchararray import array
def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
# FedMD
# file_name = "/home/cyh/fed/FedMD_clean/result_CIFAR_balanced_heter/col_performance.pkl"
# f = open(file_name, 'rb')
# col_performance = pickle.load(f)
# client_number = []
# performance = []
# avg_performance = []
# for key, val in col_performance.items():
#     client_number.append(key)
#     performance.append(val)

# for idx, val in enumerate(performance):
#     plt.plot(val, label="client{}".format(idx))
# plt.ylabel("acc")
# plt.xlabel("Number of iterations")
# plt.legend()
# plt.savefig("compared_pic/acc_FedMD_individual.png")
# plt.close()
# a = np.array(performance)
# evaluate_acc_FedMD = np.average(a, axis=0)


dataset_name = "cifar10"

# file_name = 'txt_result/{}/{}/evaluate_client_acc.pkl'.format(dataset_name, "Adam_notDealDG_y_logits/alpha1.0_lr0.001")
# f = open(file_name, 'rb')
# evaluate_client_acc = pickle.load(f)
# performance = []
# avg_performance = []
# performance_baseline = []
# avg_performance_baseline = []
# for key, val in evaluate_client_acc.items():
#     performance.append(val)

# file_name = 'txt_result/{}/{}/evaluate_client_acc.pkl'.format(dataset_name, "Adam_notDealDG_y_logits/alpha0.0_lr0.001")
# f = open(file_name, 'rb')
# evaluate_client_acc = pickle.load(f)
# for key, val in evaluate_client_acc.items():
#     performance_baseline.append(val)

# avg_performance.append(np.mean(np.array(performance[0:5]), axis=0))
# avg_performance.append(np.mean(np.array(performance[5:10]), axis=0))

# avg_performance_baseline.append(np.mean(np.array(performance_baseline[0:5]), axis=0))
# avg_performance_baseline.append(np.mean(np.array(performance_baseline[5:10]), axis=0))

# for idx, val in enumerate(avg_performance):
#     plt.plot(val, label="FedHe-client group {}".format(idx))
# for idx, val in enumerate(avg_performance_baseline):
#     plt.plot(val, label="Private-client group {}".format(idx))
# plt.ylabel("Accuracy")
# plt.xlabel("Number of iterations")
# plt.legend()
# plt.grid(linestyle="dotted")
# plt.savefig("compared_pic/acc_compared_individual.eps")
# plt.savefig("compared_pic/acc_compared_individual.png")
# plt.close()

# for idx, val in enumerate(performance):
#     plt.plot(val, label="client{}".format(idx))
# plt.ylabel("Accuracy")
# plt.xlabel("Number of iterations")
# plt.legend()
# plt.grid(linestyle="dotted")
# plt.savefig("compared_pic/acc_individual.eps")
# plt.savefig("compared_pic/acc_individual.png")
# plt.close()
# a = np.array(performance)
# evaluate_acc_FedMD = np.average(a, axis=0)

# evaluate_acc_asynFL_DG_divided_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "Adam_DGDivided_asynFL/alpha0.5_lr0.001"))
# evaluate_acc_asynFL_DG_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "Adam_DG_asynFL/alpha0.5_lr0.001"))
# evaluate_acc_asynFL_baseline_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "Adam_notDealDG_asynFL/alpha0.5_lr0.001"))

# # evaluate_loss_asynFL_list = np.loadtxt('txt_result/{}/{}/evaluate_loss.txt'.format(dataset_name, "Adam_asynFL/alpha0.5_lr0.0002"))
# # evaluate_loss_asynFL_baseline_list = np.loadtxt('txt_result/{}/{}/evaluate_loss.txt'.format(dataset_name, "Adam_notDealDG_asynFL/alpha0.5_lr0.0001"))
# evaluate_loss_asynFL_DG_divided_list = np.loadtxt('txt_result/{}/{}/evaluate_loss.txt'.format(dataset_name, "Adam_DGDivided_asynFL/alpha0.5_lr0.001"))
# evaluate_loss_asynFL_DG_list = np.loadtxt('txt_result/{}/{}/evaluate_loss.txt'.format(dataset_name, "Adam_DG_asynFL/alpha0.5_lr0.001"))
# evaluate_loss_asynFL_baseline_list = np.loadtxt('txt_result/{}/{}/evaluate_loss.txt'.format(dataset_name, "Adam_notDealDG_asynFL/alpha0.5_lr0.001"))






# evaluate_acc_y_logits_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "Adam_noAgg_notDealDG_y_logits/alpha0.5_lr0.001"))
# evaluate_acc_noAgg_y_logits_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "homo_Adam_noAgg_notDealDG_y_logits/alpha1.0_lr0.001"))
# evaluate_acc_preTrain_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "homo_Adam_preTrain_notDealDG_y_logits/alpha1.0_lr0.001"))
# evaluate_acc_y_logits1_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "Adam_notDealDG_y_logits/alpha1.0_lr0.001"))
# evaluate_acc_preTrain1_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "Adam_preTrain_notDealDG_y_logits/alpha1.0_lr0.001"))
# evaluate_acc_preTrain_noAgg_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "homo_Adam_preTrain_noAgg_notDealDG_y_logits/alpha1.0_lr0.001"))
# evaluate_acc_public_preTrain_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "homo_Adam_public_preTrain_noAgg_notDealDG_y_logits/alpha1.0_lr0.001"))
# evaluate_acc_public_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "homo_Adam_public_noAgg_notDealDG_y_logits/alpha1.0_lr0.001"))
# evaluate_acc_KLD_noAgg_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "Adam_KLD_noAgg_notDealDG_y_logits/alpha0.5_lr0.001"))
# evaluate_acc_KLD_Agg05_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "Adam_KLD_preTrain_notDealDG_y_logits/alpha0.5_lr0.001"))
# evaluate_acc_KLD_Agg1_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "Adam_KLD_preTrain_notDealDG_y_logits/alpha1.0_lr0.001"))
# evaluate_acc_softmax_noAgg_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "Adam_softmaxLg_noAgg_notDealDG_y_logits/alpha1.0_lr0.001"))
# evaluate_acc_softmax_Agg_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "Adam_softmaxLg_notDealDG_y_logits/alpha0.5_lr0.001"))
evaluate_acc_baseline1_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "Adam_noAgg_notDealDG_y_logits/alpha0.0_lr0.001"))
# evaluate_acc_FedHe02_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "Adam_noAgg_notDealDG_y_logits/alpha0.2_lr0.001"))
evaluate_acc_FedHe01_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "Adam_noAgg_notDealDG_y_logits/alpha0.5_lr0.001"))
evaluate_acc_FedHe001_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "Adam_noAgg_notDealDG_y_logits/alpha0.5_lr0.0001/clientEpoches50"))
# evaluate_acc_baseline2_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "homo_Adam_noAgg_notDealDG_y_logits/alpha0.0_lr0.001"))
# evaluate_acc_new_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "Adam_BN_noAgg_notDealDG_y_logits/alpha1.0_lr0.001"))
# evaluate_acc_asynFL_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "Adam_asynFL/alpha0.5_lr0.001"))
# evaluate_acc_dc_list = np.loadtxt('txt_result/evaluate_acc_no_decay.txt')
# makedirs('txt_result/{}/{}/'.format(dataset_name, "Adam_Var0.0001_noAgg_notDealDG_y_logits/alpha1.0_lr0.001"))
# evaluate_acc_SC_list = np.loadtxt('txt_result/{}/{}/evaluate_acc.txt'.format(dataset_name, "Adam_Var0.0001_noAgg_notDealDG_y_logits/alpha0.1_lr0.001"))
evaluate_acc_SimK1_list = np.loadtxt('txt_result/{}/{}/clientEpoches{}/topK{}/evaluate_acc.txt'.format(dataset_name, "Adam_Sim_noAgg_notDealDG_y_logits/alpha0.5_lr0.0001", 50, 1))
evaluate_acc_SimK2_list = np.loadtxt('txt_result/{}/{}/clientEpoches{}/topK{}/evaluate_acc.txt'.format(dataset_name, "Adam_Sim_noAgg_notDealDG_y_logits/alpha0.5_lr0.0001", 50, 2))
evaluate_acc_SimK5_list = np.loadtxt('txt_result/{}/{}/clientEpoches{}/topK{}/evaluate_acc.txt'.format(dataset_name, "Adam_Sim_noAgg_notDealDG_y_logits/alpha0.5_lr0.0001", 50, 5))
evaluate_acc_SimK10_list = np.loadtxt('txt_result/{}/{}/clientEpoches{}/topK{}/evaluate_acc.txt'.format(dataset_name, "Adam_Sim_noAgg_notDealDG_y_logits/alpha0.5_lr0.0001", 50, 10))


# evaluate_loss_y_logits_list = np.loadtxt('txt_result/{}/{}/evaluate_loss.txt'.format(dataset_name, "Adam_notDealDG_y_logits/alpha0.5_lr0.001"))
# evaluate_loss_preTrain_list = np.loadtxt('txt_result/{}/{}/evaluate_loss.txt'.format(dataset_name, "Adam_preTrain_notDealDG_y_logits/alpha0.5_lr0.001"))
# evaluate_loss_preTrain_noAgg_list = np.loadtxt('txt_result/{}/{}/evaluate_loss.txt'.format(dataset_name, "Adam_preTrain_noAgg_notDealDG_y_logits/alpha1.0_lr0.001"))
# evaluate_loss_KLD_noAgg_list = np.loadtxt('txt_result/{}/{}/evaluate_loss.txt'.format(dataset_name, "Adam_KLD_noAgg_notDealDG_y_logits/alpha0.5_lr0.001"))
# evaluate_loss_KLD_Agg05_list = np.loadtxt('txt_result/{}/{}/evaluate_loss.txt'.format(dataset_name, "Adam_KLD_preTrain_notDealDG_y_logits/alpha0.5_lr0.001"))
# # evaluate_loss_KLD_Agg1_list = np.loadtxt('txt_result/{}/{}/evaluate_loss.txt'.format(dataset_name, "Adam_KLD_preTrain_notDealDG_y_logits/alpha1.0_lr0.001"))
# evaluate_loss_softmax_noAgg_list = np.loadtxt('txt_result/{}/{}/evaluate_loss.txt'.format(dataset_name, "Adam_softmaxLg_noAgg_notDealDG_y_logits/alpha1.0_lr0.001"))
# evaluate_loss_softmax_Agg_list = np.loadtxt('txt_result/{}/{}/evaluate_loss.txt'.format(dataset_name, "Adam_softmaxLg_notDealDG_y_logits/alpha0.5_lr0.001"))
# evaluate_loss_baseline1_list = np.loadtxt('txt_result/{}/{}/evaluate_loss.txt'.format(dataset_name, "Adam_notDealDG_y_logits/alpha0.0_lr0.001"))
# evaluate_loss_baseline2_list = np.loadtxt('txt_result/{}/{}/evaluate_loss.txt'.format(dataset_name, "Adam_noAgg_notDealDG_y_logits/alpha0.0_lr0.001"))
# evaluate_loss_baseline3_list = np.loadtxt('txt_result/{}/{}/evaluate_loss.txt'.format(dataset_name, "fedavg/alpha0.0_lr0.0001"))
# tmp_evaluated_loss = []
# tmp_train_loss = []
#
# for idx in range(evaluate_loss_list.shape[0]):
#     if evaluate_loss_list[idx] <= 3.5:
#         tmp_evaluated_loss.append(evaluate_loss_list[idx])
#         tmp_train_loss.append(train_loss_list[idx])




# # plt.plot(evaluate_loss_asynFL_DG_divided_list, label="Divided")
# # plt.plot(evaluate_loss_asynFL_DG_list, label="DG")
# # plt.plot(evaluate_loss_asynFL_baseline_list, label="baseline")
# plt.plot(evaluate_loss_y_logits_list, label="Agg")
# plt.plot(evaluate_loss_preTrain_list, label="preTrain_Agg")
# # plt.plot(evaluate_loss_preTrain_noAgg_list, label="preTrain_noAgg")
# # plt.plot(evaluate_loss_KLD_noAgg_list, label="KLD_noAgg")
# # plt.plot(evaluate_loss_KLD_Agg05_list, label="KLD_Agg")
# # plt.plot(evaluate_loss_KLD_Agg1_list, label="KLD_Agg1")
# # plt.plot(evaluate_loss_softmax_noAgg_list, label="softmax_noAgg")
# # plt.plot(evaluate_loss_softmax_Agg_list, label="softmax_Agg")
# plt.plot(evaluate_loss_baseline1_list, label="baseline")
# plt.plot(evaluate_loss_baseline2_list, label="baseline2")
# plt.plot(evaluate_loss_baseline3_list, label="FedAvg")
# plt.title("loss")
# plt.ylabel("loss")
# plt.xlabel("Number of iterations")
# plt.legend()
# # plt.savefig("pic/asyn/loss_no_decay.eps")
# plt.savefig("compared_pic/loss_hetero.png")
# plt.close()



# plt.plot(evaluate_acc_asynFL_DG_divided_list, label="Divided")
# plt.plot(evaluate_acc_asynFL_DG_list, label="DG")
# plt.plot(evaluate_acc_asynFL_baseline_list, label="baseline")
# plt.plot(evaluate_acc_var00001_list, label="Var*0.0001")
# plt.plot(evaluate_acc_noAgg_y_logits_list, label="noAgg")
# plt.plot(evaluate_acc_preTrain_list, label="preTrain_Agg")
# plt.plot(evaluate_acc_preTrain_noAgg_list, label="preTrain_noAgg")
# plt.plot(evaluate_acc_public_preTrain_list, label="public_Pre")
# plt.plot(evaluate_acc_public_list, label="public")
# plt.plot(evaluate_acc_KLD_Agg1_list, label="KLD_Agg1")
# plt.plot(evaluate_acc_softmax_noAgg_list, label="softmax_noAgg")
# plt.plot(evaluate_acc_softmax_Agg_list, label="softmax_Agg")
plt.plot(evaluate_acc_SimK1_list, label="Sim topK1")
plt.plot(evaluate_acc_SimK2_list, label="Sim topK2")
plt.plot(evaluate_acc_SimK5_list, label="Sim topK5")
plt.plot(evaluate_acc_SimK10_list, label="Sim topK10")
plt.plot(evaluate_acc_baseline1_list, label="Private")
# plt.plot(evaluate_acc_FedHe01_list, label="FedHe 0.001")
plt.plot(evaluate_acc_FedHe001_list, label="FedHe")
# plt.plot(evaluate_acc_FedHe07_list, label="FedHe 0.7")
# plt.plot(evaluate_acc_FedMD, label="FedMD")
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Number of iterations")
plt.ylim(0.5, 0.70)
plt.legend()
plt.grid(linestyle="dotted")
plt.savefig("compared_pic/acc_var.eps")
plt.savefig("compared_pic/acc_var.png")
plt.close()


