# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_data(file_loc):#读取文件（input）
    f = open(file_loc, 'r')
    data = []## 初始化一个空列表，用于存储读取的数据
    for line in f:
        new_arr = []# 用于存储每行数据的新列表
        arr = line.split(' #')[0].split()
        score = arr[0]
        q_id = arr[1].split(':')[1]## 获取查询ID（q_id），按冒号分割并取第二部分
        new_arr.append(int(score))
        new_arr.append(int(q_id))
        arr = arr[2:77]#取剩余的部分作为特征值，范围为索引2到101
        for el in arr:
            new_arr.append(float(el.split(':')[1]))
        data.append(new_arr)
    f.close()
    return np.array(data)

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

def extract_numbers_by_id(file_path, target_id):
    numbers_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            columns = line.strip().split(' ')
            if int(columns[0]) == int(target_id):
                numbers = columns[1].split(',')
                numbers_list.extend([int(num) for num in numbers])
                break  # 找到匹配行后立即停止遍历
    return numbers_list

if __name__ == '__main__':
    test_data = 'D:\\yanjiu\\DrugRepSS\\LTR\\mydata2\\output测试集gnnrecover.txt'
    test_data = get_data(test_data)

    qid_doc_map = group_by(test_data, 1)
    hit10 = 0
    hit3 = 0
    hit1 = 0
    mrr = 0
    for qid in qid_doc_map.keys():
        h1 = 0
        h3 = 0
        h10 = 0
        mr = 0
        index_label = {}
        index_score = {}
        subset = qid_doc_map[qid]
        X_subset = torch.from_numpy(test_data[subset, 2:].astype(np.float32))

        # calculate the predicted NDCG
        true_label = test_data[qid_doc_map[qid], 0]
        for s, t in zip(subset, true_label):
            if t > 0:
                index_label[s] = t

        test_pre = './output-gnncover/100epoch_PL_rank_3.txt'
        result = extract_numbers_by_id(test_pre, qid)
        # print(result)
        num = len(index_label)
        sum = len(qid_doc_map.keys())
        for v in (index_label):
            ylist = result
            if(v in (ylist)):
                mr += 1 / (int(ylist.index(v)) + 1)
                if v in ylist[0:1]:
                    h1 += 1
                if v in ylist[:3]:
                    h3 += 1
                if v in ylist[:10]:
                    h10 += 1
            else:
                sum=sum-1
        if(num!=0):
            hit10 += h10 / num
            hit3 += h3 / num
            hit1 += h1 / num
            mrr += mr / num
    print("total test:", sum)
    print("hit@10:", hit10 / sum)
    print("hit@3:", hit3 / sum)
    print("hit@1:", hit1 / sum)
    print("mrr:", mrr / sum)