import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dcg(scores):
    """
    compute the DCG value based on the given score
    :param scores: a score list of documents
    :return v: DCG value
    """
    v = 0
    for i in range(len(scores)):
        v += (np.power(2, scores[i]) - 1) / np.log2(i+2)  # i+2 is because i starts from 0
    return v


def idcg(scores):
    """
    compute the IDCG value (best dcg value) based on the given score
    :param scores: a score list of documents
    :return:  IDCG value
    """
    best_scores = sorted(scores)[::-1]
    return dcg(best_scores)


def ndcg(scores):
    """
    compute the NDCG value based on the given score
    :param scores: a score list of documents
    :return:  NDCG value
    """
    return dcg(scores)/idcg(scores)

def single_dcg(scores, i, j):
    """
    compute the single dcg that i-th element located j-th position
    :param scores:
    :param i:
    :param j:
    :return:
    """
    return (np.power(2, scores[i]) - 1) / np.log2(j+2)


# def delta_ndcg(scores, p, q, single_dcgs):
#     """
#     swap the i-th and j-th doucment, compute the absolute value of NDCG delta
#     :param scores: a score list of documents
#     :param p, q: the swap positions of documents
#     :return: the absolute value of NDCG delta
#     """
#     delta = single_dcgs[(p,q)] + single_dcgs[(q,p)] - single_dcgs[(p,p)] -single_dcgs[(q,q)]
#     s2 = scores.copy()  # new score list
#     s2[p], s2[q] = s2[q], s2[p]  # swap
#     return abs(ndcg(s2) - ndcg(scores))


def ndcg_k(scores, k):
    scores_k = scores[:k]
    dcg_k = dcg(scores_k)
    idcg_k = dcg(sorted(scores)[::-1][:k])
    if idcg_k == 0:
        return np.nan
    return dcg_k/idcg_k




def group_by(data, qid_index):
    """
    :param data: input_data
    :param qid_index: the column num where qid locates in input Fold1
    :return: a dict group by qid
    """
    qid_doc_map = {}
    idx = 0
    for record in data:
        qid_doc_map.setdefault(record[qid_index], [])
        qid_doc_map[record[qid_index]].append(idx)
        idx += 1
    return qid_doc_map


def get_pairs(scores):
    """
    :param scores: given score list of documents for a particular query
    :return: the documents pairs whose firth doc has a higher value than second one.
    """
    pairs = []
    for i in range(len(scores)):
        for j in range(len(scores)):
            if scores[i] > scores[j]:
                pairs.append((i, j))
    return pairs


def compute_lambda(true_scores, temp_scores, order_pairs, qid):
    """
    :param true_scores: the score list of the documents for the qid query
    :param temp_scores: the predict score list of the these documents
    :param order_pairs: the partial oder pairs where first document has higher score than the second one
    :param qid: specific query id
    :return:
        lambdas: changed lambda value for these documents
        w: w value
        qid: query id
    """
    doc_num = len(true_scores)
    lambdas = np.zeros(doc_num)
    w = np.zeros(doc_num)
    IDCG = idcg(true_scores)
    single_dcgs ={}
    for i, j in order_pairs:
        if (i, i) not in single_dcgs:
            single_dcgs[(i, i)] = single_dcg(true_scores, i, i)
        if (j, j) not in single_dcgs:
            single_dcgs[(j, j)] = single_dcg(true_scores, j, j)
        single_dcgs[(i, j)] = single_dcg(true_scores, i, j)
        single_dcgs[(j, i)] = single_dcg(true_scores, j, i)

    for i, j in order_pairs:
        delta = abs(single_dcgs[(i,j)] + single_dcgs[(j,i)] - single_dcgs[(i,i)] -single_dcgs[(j,j)])/IDCG
        if -(temp_scores[i] - temp_scores[j])>=0:
            rho = 1 / (1 + np.exp(temp_scores[i] - temp_scores[j]))
        else:
            rho = np.exp(-(temp_scores[i] - temp_scores[j]))/(1+np.exp(-(temp_scores[i] - temp_scores[j])))
        lambdas[i] += rho * delta
        lambdas[j] -= rho * delta

        rho_complement = 1.0 - rho
        w[i] += rho * rho_complement * delta
        w[j] -= rho * rho_complement * delta


    return lambdas, w, qid


def load_data(file_path):
    with open(file_path, 'r') as f:
        data = []
        for line in f.readlines():
            new_arr = []
            line_split = line.split(' ')
            score = float(line_split[0])
            qid = int(line_split[1].split(':')[1])
            new_arr.append(score)
            new_arr.append(qid)
            for ele in line_split[2:102]:

                new_arr.append(float(ele.split(':')[1]))
            data.append(new_arr)
    data_np = np.array(data)
    return data_np


class Net(nn.Module):
    def __init__(self, n_feature, h1_units, h2_units):
        super(Net, self).__init__()
        self.h1 = nn.Linear(n_feature, h1_units)

        self.h2 = nn.Linear(h1_units, h2_units)

        self.out = nn.Linear(h2_units, 1)

    def forward(self, x):
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        x = self.out(x)
        return x


class LambdaRank:

    def __init__ (self, training_data, n_feature, h1_units, h2_units, epoch, lr=0.0001):
        self.training_data = training_data
        self.n_feature = n_feature
        self.h1_units = h1_units
        self.h2_units = h2_units
        self.epoch = epoch
        self.lr = lr
        self.trees = []
        self.model = Net(n_feature, h1_units, h2_units)
        # self.test_data=test_data
        # self.k=k
        # for para in self.model.parameters():
        #     print(para[0])

    def fit(self):
        """
        train the model to fit the train dataset
        """
        #按查询ID分组，得到一个查询ID到文档索引的映射
        qid_doc_map = group_by(self.training_data, 1)
        #获取所有查询的索引。
        query_idx = qid_doc_map.keys()
        # true_scores is a matrix, different rows represent different queries
        #获取所有查询对应的真实得分，存储在true_scores列表中。
        true_scores = [self.training_data[qid_doc_map[qid], 0] for qid in query_idx]

        order_paris = []#初始化一个空列表，用于存储每个查询对应的排序对。

        #遍历所有查询的真实得分，计算每个查询的排序对，并将其添加到order_paris列表中。
        for scores in true_scores:
            order_paris.append(get_pairs(scores))

        sample_num = len(self.training_data)#获取训练样本的数量。

        print('Training .....\n')

        #使用循环迭代训练过程，其中epoch是训练的迭代次数。
        for i in range(self.epoch):
            #使用训练数据的特征部分作为输入，通过模型进行预测，得到预测得分。
            predicted_scores = self.model(torch.from_numpy(self.training_data[:, 2:].astype(np.float32)))
            #将预测得分转换为NumPy数组。
            predicted_scores_numpy = predicted_scores.data.numpy()
            print('predicted_scores_numpy:')
            print(predicted_scores_numpy)
            #初始化一个长度为样本数的零数组，用于存储每个样本的lambda值。
            lambdas = np.zeros(sample_num)
            # w = np.zeros(sample_num)

            #遍历每个查询的真实得分、预测得分和排序对，并计算每个样本的lambda值。
            pred_score = [predicted_scores_numpy[qid_doc_map[qid]] for qid in query_idx]

            #使用zip函数将真实得分、预测得分、排序对和查询索引进行打包，以便进行迭代。
            zip_parameters = zip(true_scores, pred_score, order_paris, query_idx)
            #使用zip函数迭代每个查询的真实得分、预测得分、排序对和查询索引。
            for ts, ps, op, qi in zip_parameters:
                #计算每个查询的lambda值、w值和查询ID。
                sub_lambda, sub_w, qid = compute_lambda(ts, ps, op, qi)
                #将计算得到的lambda值存储在lambdas数组中，使用查询ID作为索引。
                lambdas[qid_doc_map[qid]] = sub_lambda
                # print('lambdas:')
                # print(lambdas)
                # w[qid_doc_map[qid]] = sub_w
            # update parameters
            self.model.zero_grad()#将模型的梯度置零。
            # 将lambda值转换为PyTorch张量，并调整形状以匹配预测得分。
            lambdas_torch = torch.Tensor(lambdas).view((len(lambdas), 1))
            #根据lambda值计算梯度，这一步是反向传播的关键。
            predicted_scores.backward(lambdas_torch, retain_graph=True)  # This is very important. Please understand why?
            with torch.no_grad():
                #使用梯度下降算法更新模型参数
                for param in self.model.parameters():
                    param.data.add_(param.grad.data * self.lr)


            #计算NDCG
            if i % 1 == 0:
                qid_doc_map = group_by(self.training_data, 1)
                ndcg_list = []
                for qid in qid_doc_map.keys():
                    subset = qid_doc_map[qid]

                    X_subset = torch.from_numpy(self.training_data[subset, 2:].astype(np.float32))
                    sub_pred_score = self.model(X_subset).data.numpy().reshape(1, len(X_subset)).squeeze()

                    # calculate the predicted NDCG
                    true_label = self.training_data[qid_doc_map[qid], 0]
                    k = len(true_label)
                    pred_sort_index = np.argsort(sub_pred_score)[::-1]
                    true_label = true_label[pred_sort_index]
                    ndcg_val = ndcg_k(true_label, k)
                    ndcg_list.append(ndcg_val)
                print('Epoch:{}, Average NDCG : {}'.format(i, np.nanmean(ndcg_list)))

    def validate(self,file, data, k):
        """
        validate the NDCG metric
        :param data: given th testset
        :param k: used to compute the NDCG@k
        :return:
        """
        qid_doc_map = group_by(data, 1)
        ndcg_list = []
        predicted_scores = np.zeros(len(data))
        hit10 = 0
        hit3 = 0
        hit1 = 0
        mrr = 0
        file = open("./resultdata/LambRank/example_lambdaRank_1v1.txt", 'w', encoding='utf8')
        for qid in qid_doc_map.keys():
            h1=0
            h3=0
            h10=0
            mr=0
            index_label={}
            index_score={}
            subset = qid_doc_map[qid]
            X_subset = torch.from_numpy(data[subset, 2:].astype(np.float32))
            sub_pred_score = self.model(X_subset).data.numpy().reshape(1, len(X_subset)).squeeze()
            for i,s in zip(subset,sub_pred_score):
                index_score[i]=s


            # calculate the predicted NDCG
            true_label = data[qid_doc_map[qid], 0]
            for s,t in zip(subset,true_label):
                if t>0:
                    index_label[s]=t
            rank=sorted(index_score.items(), key=lambda item:item[1], reverse=True)
            example = list(dict(rank).keys())[:10]
            lst = []
            for e in example:
                lst.append(str(e))
            file.write(str(int(qid)) + "\t" + ",".join(lst) + "\n")

            for v in (index_label):
                ylist = list(dict(rank).keys())
                mr += 1 / (int(ylist.index(v)) + 1)
                if v in ylist[0:1]:
                    h1 += 1
                if v in ylist[:3]:
                    h3 += 1
                if v in ylist[:10]:
                    h10 += 1

            num = len(index_label)
            sum1=0
            if(num!=0):
                hit10+=h10/num
                hit3+=h3/num
                hit1+=h1/num
                mrr+=mr/num
            else:
                sum1=sum1+1

            k = len(true_label)
            pred_sort_index = np.argsort(sub_pred_score)[::-1]
            true_label = true_label[pred_sort_index]
            ndcg_val = ndcg_k(true_label, k)
            ndcg_list.append(ndcg_val)
        sum=len(qid_doc_map.keys())-sum1
        print("total test:", sum)
        print("hit@10:", hit10 / sum)
        print("hit@3:", hit3 / sum)
        print("hit@1:", hit1 / sum)
        print("mrr:", mrr / sum)
        mymetric = [hit10 / sum, hit3 / sum, hit1 / sum, mrr / sum]
        return np.nanmean(ndcg_list),mymetric

