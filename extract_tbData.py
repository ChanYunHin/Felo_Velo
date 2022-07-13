import os
import numpy as np
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]


    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = []

    for tag in tags:
        
        steps = [i for i in range(100)]

        
        tmp_events = []
        tmp_names = []
        for acc in summary_iterators:
            tmp_events.append(acc.Scalars(tag))
            a = os.path.split(acc.path)
            tmp_names.append(a[1])
            # print(a[1])

        tmp_events = [acc.Scalars(tag) for acc in summary_iterators]
        select_events = []
        select_name = []
        for idx, event in enumerate(tmp_events):
            if len(event) == 100:
                select_events.append(event)
                select_name.append(tmp_names[idx])
            # else:
            #     event.extend([0 for _ in range(100-len(event))])
            #     select_events.append(event)

        # b = zip(*[acc.Scalars(tag) for acc in summary_iterators])
        
        # for tmp in b:
        #     print(tmp)


        for events in zip(*select_events):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps, select_name


def to_csv(dpath):
    # dirs = os.listdir(dpath)

    d, steps, select_name = tabulate_events(dpath)
    tags, values = zip(*d.items())
    np_values = np.array(values)

    for index, tag in enumerate(tags):
        df = pd.DataFrame(np_values[index], index=steps, columns=select_name)
        # tmppath = get_file_path(dpath, tag)
        # folder_path = os.path.join(dpath, 'csv')
        # a = 'fed_distill_pytorch/tb_logs/cmp_homo/rn0.2_Fmnist_50/homo_Adamsr0.2_rn_GKT_KLD_y_logits_alpha1.0_lr0.001/test/accuracy_beta10_acc/csv/accuracy.csv'
        tmppath = dpath.split('/')
        save_path = os.path.join('result/', *tmppath[1:3])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_save_path = save_path + "/" + tmppath[3] + ".csv"
        df.to_csv(file_save_path)


def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(dpath, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


if __name__ == '__main__':
    # path = "tb_logs/cmp_homo/rn0.2_Fmnist_50/homo_Adamsr0.2_rn_GKT_KLD_y_logits_alpha1.0_lr0.001/test/accuracy_beta10_acc"
    # path = "fed_distill_pytorch/tb_logs/cmp_homo/rn0.2_Fmnist_50/homo_Adamsr0.2_rn_GKT_KLD_y_logits_alpha1.0_lr0.001/test/accuracy_beta10_acc"
    path = "tb_logs/cmp/rn0.2_cifar10_50"
    # path = "fed_distill_pytorch/tb_logs/cmp/rn0.2_Fmnist_16"
    methods = os.listdir(path)
    for method_name in methods:
        if 'Adam5' in method_name:
            method_path = "{}/{}/test/".format(path, method_name)
            folder_name = os.listdir(method_path)
            for fname in folder_name:
                if fname[0:3] == 'acc':
                    # print(fname)
                    method_path = method_path + fname
                    # if method_path == 'tb_logs/cmp/rn0.2_Fmnist_16/Adamsr0.2_rn_MD_alpha1.0_lr0.001/test/accuracy_FedHe_acc':
                    #     continue
                    to_csv(method_path)
                
        # print(folder_name)
        # to_csv(method_path)
    