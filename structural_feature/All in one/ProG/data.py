import torch
import pickle as pk
from random import shuffle

from torch_geometric.data import Batch
from collections import defaultdict


def multi_class_NIG(dataname, num_class,shots=100):
    """
    NIG: node induced graphs
    :param dataname: e.g. CiteSeer 数据集名称
    :param num_class: e.g. 6 (for CiteSeer) 任务数量
    :return: a batch of training graphs and testing graphs
    """
    #创建字典 (key, list)
    statistic = defaultdict(list)

    # load training NIG (node induced graphs)加载训练集的节点诱导图
    train_list = []
    for task_id in range(num_class):
        data_path1 = './Dataset/{}/induced_graphs/task{}.meta.train.support'.format(dataname, task_id)
        data_path2 = './Dataset/{}/induced_graphs/task{}.meta.train.query'.format(dataname, task_id)

        with (open(data_path1, 'br') as f1, open(data_path2, 'br') as f2):
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots] #0--shots 行
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                train_list.append(g)# 将每个任务的图存储到训练列表中
    shuffle(train_list)#打乱顺序
    train_data = Batch.from_data_list(train_list)

    test_list = []
    for task_id in range(num_class):
        data_path1 = './Dataset/{}/induced_graphs/task{}.meta.test.support'.format(dataname, task_id)
        data_path2 = './Dataset/{}/induced_graphs/task{}.meta.test.query'.format(dataname, task_id)

        with (open(data_path1, 'br') as f1, open(data_path2, 'br') as f2):
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]

            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                test_list.append(g)
    print(test_list)
    shuffle(test_list)
    test_data = Batch.from_data_list(test_list)

    for key, value in statistic.items():
        print("{}ing set (class_id, graph_num): {}".format(key, value))

    return train_data, test_data, train_list,test_list


if __name__ == '__main__':
    pass
