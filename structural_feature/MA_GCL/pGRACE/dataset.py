import os.path as osp
import pandas as pd
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T
import torch
from ogb.nodeproppred import PygNodePropPredDataset

def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code']
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~/datasets')
    print (root_path)
    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
#        return Coauthor(root=path, name='cs')
    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())
#        return Amazon(root=path, name='computers')

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
#        return Amazon(root=path, name='photo')

    if name.startswith('ogbn'):
        return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())


def get_path(base_path, name):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return base_path
    else:
        return osp.join(base_path, name)

def get_features(fea_type,input_file):
    feature = None
    if fea_type == 'CMap features':
        df_dis_drug = pd.read_excel(input_file, sheet_name='CMap_fea')
        dis_drug_fea = [i[1: -1].split(', ') for i in list(df_dis_drug['Gene_vect'])]
        dis_drug_fea_init = []
        for dr in dis_drug_fea:
            dis_drug_fea_init.append(list(map(float, dr)))

        fin_dis_drug_fea_init = []
        for dis_drug_fea_init_list in dis_drug_fea_init:
            one_dis_drug_fea_init = []
            for dis_drug_fea in dis_drug_fea_init_list:
                one_dis_drug_fea_init.append(dis_drug_fea * 0.01)
            fin_dis_drug_fea_init.append(one_dis_drug_fea_init)
        feature = torch.Tensor(fin_dis_drug_fea_init)


    elif fea_type == 'random features':
        feature = torch.normal(0, 0.1, size=(625, 978))

    else:
        print(f"Warning: Unsupported feature type '{fea_type}'. Returning default value.")

    return feature



def get_rel(input_file):
    df_dis_drug = pd.read_excel(input_file,sheet_name='dis_drug_rel')
    dis_list = df_dis_drug['dis_ID'].to_list()
    drug_list = df_dis_drug['drug_ID'].to_list()
    edge_index_list = [dis_list,drug_list]
    edge_index = torch.tensor(edge_index_list)

    return edge_index

