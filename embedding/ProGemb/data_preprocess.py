from collections import defaultdict
import pickle as pk
from torch_geometric.utils import subgraph, k_hop_subgraph
import torch
import numpy as np
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.data import Data, Batch
import random
import warnings
from ProG.utils import mkdir
from random import shuffle

# this file has been tested applicable on PubMed and CiteSeer.
# next, we will further make it complicable with Cora and Reddits2
# 为元学习准备数据集
def nodes_split(data: Data, dataname: str = None, node_classes=3):
    if dataname is None:
        raise KeyError("dataname is None!")
    #路径创建： 构建保存索引文件的路径
    index_path = './dataset/{}/index/'.format(dataname)
    mkdir(index_path)

    node_labels = data.y#获取节点标签信息。

    # step1: split/sample nodes for meta-training support | meta-training query | meta-test support | meta-test query |
    #接下来的步骤将节点分割用于元训练的支持集、元训练的查询集、元测试的支持集和元测试的查询集。
    #对每个节点类别执行以下操作。
    for i in range(0, node_classes):
        #正样本节点选择： 找到属于当前类别的节点，然后随机排列它们。
        pos_nodes = torch.argwhere(node_labels == i)  # torch.squeeze(torch.argwhere(node_labels == i))
        pos_nodes = pos_nodes[torch.randperm(pos_nodes.shape[0])]
        # TODO: ensure each label contain more than 400 nodes
        # 如果节点数量少于 400，发出运行时警告，否则选择前 400 个节点。
        if pos_nodes.shape[0] < 400:
            warnings.warn("label {} only has {} nodes but it should be larger than 400!".format(i, pos_nodes.shape[0]),
                          RuntimeWarning)
        else:
            pos_nodes = pos_nodes[0:400]
        # print(pos_nodes.shape)

        #将节点分为四部分，每部分用于元训练的支持集、元训练的查询集、元测试的支持集和元测试的查询集。
        # 1:1:1:1 split shuffled nodes for meta-training support | meta-training query | meta-test support | meta-test query
        pos_split = int(pos_nodes.shape[0] / 4)
        #分区创建： 根据当前循环迭代，将节点分配到不同的分区。
        for p in range(4):  # p=0,1,2,3
            partition_dic_list = defaultdict(torch.Tensor)
            if p < 3:
                partition_dic_list['pos'] = pos_nodes[p * pos_split:(p + 1) * pos_split, :]
            else:
                partition_dic_list['pos'] = pos_nodes[p * pos_split:, :]
            #确定名字
            if p == 0:
                dname = 'task{}.meta.train.support'.format(i)
            elif p == 1:
                dname = 'task{}.meta.train.query'.format(i)
            elif p == 2:
                dname = 'task{}.meta.test.support'.format(i)
            elif p == 3:
                dname = 'task{}.meta.test.query'.format(i)
            #使用 Pickle 将分区数据保存到二进制文件。
            pk.dump(partition_dic_list, open(index_path + dname, 'bw'))

#用于处理边任务的数据集准备
def edge_split(data, dataname: str = None, node_classes=3):
    """
    edge task:
    接下来的任务是为每个节点类别创建边任务，其中每个节点类别都与自己相连
    label1, label1
    label2, label2
    label3, label3
    """
    if dataname is None:
        raise KeyError("dataname is None!")

    index_path = './dataset/{}/index/'.format(dataname)
    mkdir(index_path)

    node_labels = data.y
    edge_index = data.edge_index

    for n_label in range(node_classes):
        """
        说明任务 ID 的范围，获取当前节点类别的子图。
        node-task: [0, num_node_classes)
        edge-task: [num_node_classes, 2*num_node_classes)
        """
        task_id = node_classes + n_label#变为边的任务

        subset = torch.argwhere(node_labels == n_label)  # (num, 1)
        #标签等于当前节点类别 n_label 的节点索引
        print("label {} total num subset {}".format(n_label, subset.shape[0]))
        #输出显示当前节点类别为 n_label 的节点子集的数量。
        sub_edges, _ = subgraph(subset, edge_index)#使用 subgraph 函数从原始图中提取与当前节点子集相关的子图
        print("label {} total sub_edges {}".format(n_label, sub_edges.shape[1]))

        # TODO: you can also sample even more edges (larger than 400)
        #随机排列子图中的边，并选择前 400 条。
        edge_index_400_shot = sub_edges[:, torch.randperm(sub_edges.shape[1])][:, 0:400]
        # print(edge_index_400_shot.shape)

        pos_split = int(edge_index_400_shot.shape[1] / 4)
        #将边分为四部分，每部分用于元训练的支持集、元训练的查询集、元测试的支持集和元测试的查询集。
        for p in range(4):  # p=0,1,2,3
            partition_dic_list = defaultdict(torch.Tensor)
            if p < 3:
                partition_dic_list['pos'] = edge_index_400_shot[:, p * pos_split:(p + 1) * pos_split]
            else:
                partition_dic_list['pos'] = edge_index_400_shot[:, p * pos_split:]
            if p == 0:
                dname = 'task{}.meta.train.support'.format(task_id)
            elif p == 1:
                dname = 'task{}.meta.train.query'.format(task_id)
            elif p == 2:
                dname = 'task{}.meta.test.support'.format(task_id)
            elif p == 3:
                dname = 'task{}.meta.test.query'.format(task_id)

            pk.dump(partition_dic_list,
                    open(index_path + dname, 'bw'))

#生成图
def induced_graphs_nodes(data, dataname: str = None, num_classes=3, smallest_size=100, largest_size=300):
    """
    node-level: [0,num_classes)
    edge-level: [num_classes,num_classes*2)
    图任务的id范围
    graph-level: [num_classes*2,num_classes*3)
    """
    if dataname is None:
        raise KeyError("dataname is None!")
    #存储路径
    induced_graphs_path = './dataset/{}/induced_graphs/'.format(dataname)
    mkdir(induced_graphs_path)

    edge_index = data.edge_index
    ori_x = data.x

    fnames = []#存储文件名的空列表。
    #生成每个任务的文件名列表，用于加载感知子图。
    for i in range(0, num_classes):  # TODO: remember to reset to num_classies!
        for t in ['train', 'test']:
            for d in ['support', 'query']:
                fname = './dataset/{}/index/task{}.meta.{}.{}'.format(dataname, i, t, d)
                fnames.append(fname)

    for fname in fnames:
        induced_graph_dic_list = defaultdict(list)#创建一个默认字典以存储感知子图。
        # aa = torch.load(fname)
        #解析文件名： 从文件名中提取任务ID、数据集类型和数据集子类型。
        sp = fname.split('.')
        prefix_task_id, t, d = sp[-4], sp[-2], sp[-1]
        i = prefix_task_id.split('/')[-1][4:]
        # print("task{}.meta.{}.{}...".format(i, t, d))

        a = pk.load(open(fname, 'br'))
        # print('*****************')
        #从文件中加载数据，其中 value 是正类别的节点集，label 是用于感知子图的标签。
        value = a['pos']
        label = torch.tensor([1]).long()
        induced_graph_list = []
        # for r in range(value.shape[0]):

        value = value[torch.randperm(value.shape[0])]
        iteration = 0
        #对节点进行随机排列，并且开始迭代处理每个节点。
        for node in torch.flatten(value):
            iteration = iteration + 1
            #如果生成的子图小于指定的最小大小且当前跳数小于5，逐步增加跳数，直到满足大小要求。
            subset, _, _, _ = k_hop_subgraph(node_idx=node.item(), num_hops=2,
                                             edge_index=edge_index, relabel_nodes=True)
            current_hop = 2
            while len(subset) < smallest_size and current_hop < 5:
                # print("subset smaller than {} explore higher hop...".format(smallest_size))
                current_hop = current_hop + 1
                subset, _, _, _ = k_hop_subgraph(node_idx=node.item(), num_hops=current_hop,
                                                 edge_index=edge_index)
            #调整子图大小： 如果子图超过最大大小，则随机截取，并确保节点唯一性。然后通过 subgraph 函数获取子图的边索引。
            if len(subset) > largest_size:
                subset = subset[torch.randperm(subset.shape[0])][0:largest_size - 1]
                subset = torch.unique(torch.cat([torch.LongTensor([node.item()]), torch.flatten(subset)]))

            sub_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True)

            x = ori_x[subset]
            induced_graph = Data(x=x, edge_index=sub_edge_index, y=label)
            induced_graph_list.append(induced_graph)
            print('graph size {} at {:.2f}%...'.format(induced_graph.x.shape[0], iteration * 100.0 / value.shape[0]))

        induced_graph_dic_list['pos'] = induced_graph_list

        if len(induced_graph_dic_list['pos']) < 100:
            # raise ValueError("candidate graphs should be at least 400")
            warnings.warn("===task{}.meta.{}.{} has not enough graphs "
                          "(should be 100 but got {})".format(i, t, d, len(induced_graph_dic_list['pos'])),
                          RuntimeWarning)

        pk.dump(induced_graph_dic_list,
                open('{}task{}.meta.{}.{}'.format(induced_graphs_path, i, t, d), 'bw'))

        print('node-induced graphs saved!')


def induced_graphs_edges(data, dataname: str = None, num_classes=3, smallest_size=100, largest_size=300):
    """
        node-level: [0,num_classes)
        edge-level: [num_classes,num_classes*2)
        graph-level: [num_classes*2,num_classes*3)
    """
    if dataname is None:
        raise KeyError("dataname is None!")
    #定义感知图保存的路径
    induced_graphs_path = './dataset/{}/induced_graphs/'.format(dataname)
    mkdir(induced_graphs_path)

    edge_index = data.edge_index
    ori_x = data.x#初始化文件名列表。

    fnames = []
    #根据任务、训练/测试和支持/查询，生成文件名列表。
    for task_id in range(num_classes, 2 * num_classes):
        for t in ['train', 'test']:
            for d in ['support', 'query']:
                fname = './dataset/{}/index/task{}.meta.{}.{}'.format(dataname, task_id, t, d)
                fnames.append(fname)


    # 1-hop edge induced graphs
    #遍历文件名列表，加载任务元信息，初始化标签和位置信息。
    for fname in fnames:
        induced_graph_dic_list = defaultdict(list)

        sp = fname.split('.')
        prefix_task_id, t, d = sp[-4], sp[-2], sp[-1]
        task_id = int(prefix_task_id.split('/')[-1][4:])
        # print("task{}.meta.{}.{}...".format(task_id, t, d))

        n_label = task_id - num_classes

        # same_label_edge_index, _ = subgraph(torch.squeeze(torch.argwhere(node_labels == n_label)),
        #                                     edge_index,
        #                                     relabel_nodes=False)  # attention! relabel_nodes=False!!!!!!
        # # I previously use the following to construct graph but most of the baselines ouput 1.0 acc.
        a = pk.load(open(fname, 'br'))
        label = torch.tensor([1]).long()
        value = a['pos']
        induced_graph_list = []
        #对于每个位置，通过广度优先搜索获取感知图的子图，确保子图大小在指定范围内。
        for c in range(value.shape[1]):
            src_n, tar_n = value[0, c].item(), value[1, c].item()

            subset, _, _, _ = k_hop_subgraph(node_idx=[src_n, tar_n], num_hops=1,
                                             edge_index=edge_index)

            temp_hop = 1
            while len(subset) < smallest_size and temp_hop < 3:
                # print("subset smaller than {} explore higher hop...".format(smallest_size))
                temp_hop = temp_hop + 1
                subset, _, _, _ = k_hop_subgraph(node_idx=[src_n, tar_n], num_hops=temp_hop,
                                                 edge_index=edge_index)


            if len(subset) > largest_size:
                subset = subset[torch.randperm(subset.shape[0])][0:largest_size]
                centered_paris = torch.LongTensor([src_n, tar_n])
                subset = torch.unique(torch.cat([centered_paris, subset]))

            x = ori_x[subset]
            sub_edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True)

            induced_graph = Data(x=x, edge_index=sub_edge_index, y=label)

            # if not(smallest_size <= induced_graph.x.shape[0] <= largest_size):
            #     print(induced_graph.x.shape[0])

            induced_graph_list.append(induced_graph)

        induced_graph_dic_list['pos'] = induced_graph_list
        #将生成的感知图保存到指定路径。
        pk.dump(induced_graph_dic_list,
                open('{}task{}.meta.{}.{}'.format(induced_graphs_path, task_id, t, d), 'bw'))


def induced_graphs_graphs(data, dataname: str = None, num_classes=3, smallest_size=100,
                          largest_size=300):
    """
        node-level: [0,num_classes)
        edge-level: [num_classes,num_classes*2)
        graph-level: [num_classes*2,num_classes*3)

    可否这样做graph induced graph？
    metis生成多个graph
    然后对这些graph做扰动变成更多的graphs
    这部分是用于生成感知图的图版本，其中包括对图进行划分和扰动的思想。
    """
    if dataname is None:
        raise KeyError("dataname is None!")

    induced_graphs_path = './dataset/{}/induced_graphs/'.format(dataname)
    mkdir(induced_graphs_path)

    node_labels = data.y
    edge_index = data.edge_index
    ori_x = data.x
    num_nodes = data.x.shape[0]

    #遍历节点标签，初始化任务ID和节点列表。
    for n_label in range(num_classes):
        task_id = 2 * num_classes + n_label
        #选择了与当前类别标签相匹配的节点。
        nodes = torch.squeeze(torch.argwhere(node_labels == n_label))
        nodes = nodes[torch.randperm(nodes.shape[0])]#进一步筛选，产生随机数
        # print("there are {} nodes for label {} task_id {}".format(nodes.shape[0],n_label,task_id))


        # # I previouly use the following to construct graph but most of the baselines ouput 1.0 acc.
        # same_label_edge_index, _ = subgraph(nodes, edge_index, num_nodes=num_nodes,
        #                                     relabel_nodes=False)  # attention! relabel_nodes=False!!!!!!
        same_label_edge_index=edge_index#将边索引初始化为与同类别相关的边索引。

        #拆分节点
        split_size = max(5, int(nodes.shape[0] / 400))
        seeds_list = list(torch.split(nodes, split_size))
        #如果节点拆分数不足400，则进行重复拆分，确保拆分数达到400。
        if len(seeds_list) < 400:
            print('len(seeds_list): {} <400, start overlapped split'.format(len(seeds_list)))
            seeds_list = []
            while len(seeds_list) < 400:
                split_size = random.randint(3, 5)
                seeds_list_1 = torch.split(nodes, split_size)
                seeds_list = seeds_list + list(seeds_list_1)
                nodes = nodes[torch.randperm(nodes.shape[0])]

        shuffle(seeds_list)
        seeds_list = seeds_list[0:400]#随机打乱子列表顺序，并选择其中的前400个子列表

        #这个循环迭代了4种不同类型的任务。
        for p in range(4):  # p=0,1,2,3
            if p == 0:
                dname = 'task{}.meta.train.support'.format(task_id)
            elif p == 1:
                dname = 'task{}.meta.train.query'.format(task_id)
            elif p == 2:
                dname = 'task{}.meta.test.support'.format(task_id)
            elif p == 3:
                dname = 'task{}.meta.test.query'.format(task_id)

            induced_graph_dic_list = defaultdict(list)
            seeds_part_list = seeds_list[p * 100:(p + 1) * 100]
            #生成一个空的字典列表，用于存储生成的子图，然后根据任务类型选择相应数量的种子节点。

            #对于每个种子节点集合，生成一个k跳子图，并确保节点标签的重新标记。
            for seeds in seeds_part_list:

                subset, _, _, _ = k_hop_subgraph(node_idx=seeds, num_hops=1, num_nodes=num_nodes,
                                                 edge_index=same_label_edge_index, relabel_nodes=True)

                # regularize its size

                temp_hop = 1
                while len(subset) < smallest_size and temp_hop < 5:
                    temp_hop = temp_hop + 1
                    subset, _, _, _ = k_hop_subgraph(node_idx=seeds, num_hops=temp_hop, num_nodes=num_nodes,
                                                     edge_index=same_label_edge_index, relabel_nodes=True)

                #如果子图大小超过最大限制，则随机截取一部分节点。然后，确保子图节点的唯一性。
                if len(subset) > largest_size:
                    # directly downmsample
                    subset = subset[torch.randperm(subset.shape[0])][0:largest_size - len(seeds)]
                    subset = torch.unique(torch.cat([seeds, subset]))

                #根据子图节点和原始边索引，生成子图的边索引。
                sub_edge_index, _ = subgraph(subset, same_label_edge_index, num_nodes=num_nodes, relabel_nodes=True)

                #基于子图节点和边索引，构建图数据对象，并添加到字典列表中。
                x = ori_x[subset]
                graph = Data(x=x, edge_index=sub_edge_index)
                induced_graph_dic_list['pos'].append(graph)

            pk.dump(induced_graph_dic_list,
                    open('{}{}'.format(induced_graphs_path, dname), 'bw'))

            print("{} saved! len {}".format(dname, len(induced_graph_dic_list['pos'])))


def induced_graph_2_K_shot(t1_dic, t2_dic, dataname: str = None,
                           K=None, seed=None):
    if dataname is None:
        raise KeyError("dataname is None!")
    #根据是否提供了K值，选择相应数量的正类别和负类别数据。
    if K:
        t1_pos = t1_dic['pos'][0:K]
        t2_pos = t2_dic['pos'][0:K]  # treat as neg
    else:
        t1_pos = t1_dic['pos']
        t2_pos = t2_dic['pos']  # treat as neg

    #将正类别的图数据标记为1，负类别的图数据标记为0，并添加到任务数据列表中。
    task_data = []
    for g in t1_pos:
        g.y = torch.tensor([1]).long()
        task_data.append(g)

    for g in t2_pos:
        g.y = torch.tensor([0]).long()
        task_data.append(g)

    #如果提供了种子值，则使用种子对数据进行随机化。然后，将任务数据列表转换为批次数据。
    if seed:
        random.seed(seed)
    random.shuffle(task_data)

    batch = Batch.from_data_list(task_data)

    return batch


def load_tasks(meta_stage: str, task_pairs: list, dataname: str = None, K_shot=None, seed=0):
    #加载元学习任务的数据
    if dataname is None:
        raise KeyError("dataname is None!")

    """
    :param meta_stage: 'train', 'test'
    :param task_id_list:
    :param K_shot:  default: None.
                    if K_shot is None, load the full data to train/test meta.
                    Else: K-shot learning with 2*K graphs (pos:neg=1:1)
    :param seed:
    :return: iterable object of (task_id, support, query)


    # 从序列中取2个元素进行排列
        for e in it.permutations('ABCD', 2):
            print(''.join(e), end=', ') # AB, AC, AD, BA, BC, BD, CA, CB, CD, DA, DB, DC,

    # 从序列中取2个元素进行组合、元素不允许重复
        for e in it.combinations('ABCD', 2):
            print(''.join(e), end=', ') # AB, AC, AD, BC, BD, CD,

    """

    max_iteration = 100

    i = 0
    while i < len(task_pairs) and i < max_iteration:
        task_1, task_2 = task_pairs[i]

        task_1_support = './dataset/{}/induced_graphs/task{}.meta.{}.support'.format(dataname, task_1, meta_stage)
        task_1_query = './dataset/{}/induced_graphs/task{}.meta.{}.query'.format(dataname, task_1, meta_stage)
        task_2_support = './dataset/{}/induced_graphs/task{}.meta.{}.support'.format(dataname, task_2, meta_stage)
        task_2_query = './dataset/{}/induced_graphs/task{}.meta.{}.query'.format(dataname, task_2, meta_stage)

        with (open(task_1_support, 'br') as t1s,
              open(task_1_query, 'br') as t1q,
              open(task_2_support, 'br') as t2s,
              open(task_2_query, 'br') as t2q):
            #使用pickle库加载任务1支持集的字典数据。
            t1s_dic, t2s_dic = pk.load(t1s), pk.load(t2s)
            support = induced_graph_2_K_shot(t1s_dic, t2s_dic, dataname, K=K_shot, seed=seed)

            t1q_dic, t2q_dic = pk.load(t1q), pk.load(t2q)
            query = induced_graph_2_K_shot(t1q_dic, t2q_dic, dataname, K=K_shot, seed=seed)

        i = i + 1
        yield task_1, task_2, support, query, len(task_pairs)
        #用生成器yield返回任务对应的支持集、查询集以及任务对数目。


if __name__ == '__main__':
    dataname = 'CiteSeer'  # 'CiteSeer'  # 'PubMed' 'Cora'

    if dataname in ['CiteSeer', 'PubMed', 'Cora']:
        dataset = Planetoid(root='./dataset/', name=dataname)
    elif dataname=='Computers':
        dataset = Amazon(root='./dataset/', name=dataname)

    data = dataset.data
    # # this is legitimate on Cora, CiteSeer, and PubMed. but it refers to graph num classes for ENZYMES
    node_classes = dataset.num_classes#获取数据集的节点类别数。
    #
    # # step1 use SVD to reduce input-dim as 100 (PubMed: from 500 to 100 | CiteSeer from 3,703 to 100. )
    # # TODO: next, we will try to make Cora from 1433 to 100, and Reddit from 602 to 100,
    # #  then we can further study transfer issues across different datasets.
   #初始化一个特征降维的模块，使用SVD将输入维度降至100。
    feature_reduce = SVDFeatureReduction(out_channels=100)
    data = feature_reduce(data)
    pk.dump(data, open('./dataset/{}/feature_reduced.data'.format(dataname), 'bw'))

    data = pk.load(open('./dataset/{}/feature_reduced.data'.format(dataname), 'br'))
    #node_classes=10
    # # step2 split node and edge

    nodes_split(data, dataname=dataname, node_classes=node_classes)
    edge_split(data, dataname=dataname, node_classes=node_classes)

    # step3: induced graphs
    induced_graphs_nodes(data, dataname=dataname, num_classes=node_classes, smallest_size=100,
                         largest_size=300)
    induced_graphs_edges(data, dataname=dataname, num_classes=node_classes, smallest_size=100,
                         largest_size=300)
    induced_graphs_graphs(data, dataname=dataname, num_classes=node_classes, smallest_size=100,
                          largest_size=300)

    # aa=pk.load(open('./dataset/PubMed/induced_graphs/task4.meta.test.query', 'br'))
    # g=aa['pos'][0]
    # print(g.x)
    # print(g.edge_index)


    pass
