import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import SVDFeatureReduction
from sklearn.decomposition import PCA
from torch import nn, optim
import torch
from ProG.prompt import FrontAndHead, GNN ,Pipeline
from ProG.meta import MAML



def model_components():
  """
  input_dim, dataname, gcn_layer_num=2, hid_dim=16, num_classes=2,
               task_type="multi_label_classification",
               token_num=10, cross_prune=0.1, inner_prune=0.3, gnn_type='TransformerConv'

  :param args:
  :param round:
  :param pre_train_path:
  :param gnn_type:
  :param project_head_path:
  :return:
  """
  adapt_lr = 0.01
  meta_lr = 0.001
  # 创建一个FrontAndHead模型对象，设置输入维度为100、隐藏维度为100、类别数为2，任务类型为多标签分类，令牌数量为10，交叉剪枝率为0.1，内部剪枝率为0.3。
  # 重提示
  model = FrontAndHead(input_dim=100, hid_dim=100, num_classes=2,  # 0 or 1
                       task_type="multi_label_classification",
                       token_num=10, cross_prune=0.1, inner_prune=0.3)

  # load pre-trained GNN
  gnn = GNN(input_dim=100, hid_dim=100, out_dim=100, gcn_layer_num=2, gnn_type='TransformerConv')
  pre_train_path = './pre_trained_gnn/{}.GraphCL.{}.pth'.format('CiteSeer', 'TransformerConv')
  gnn.load_state_dict(torch.load(pre_train_path))  # 权重加载到gnn中
  print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))
  # 将GNN模型的参数设置为不需要梯度计算
  for p in gnn.parameters():
    p.requires_grad = False

  maml = MAML(model, lr=adapt_lr, first_order=False, allow_nograd=True)

  opt = optim.Adam(filter(lambda p: p.requires_grad, maml.parameters()), meta_lr)

  lossfn = nn.CrossEntropyLoss(reduction='mean')

  return maml, gnn, opt, lossfn


if __name__ == '__main__':
  import pandas as pd


  table_path = 'D:\\yanjiu\\DrugRepSS\\tructural_feature\\dataset\\disease_drug_vector.xlsx'  # 请替换为实际文件路径
  df = pd.read_excel(table_path)

  # 提取第三列到最后一列的数据
  x_data = df.iloc[:, 2:].values

  pca = PCA(n_components=100)

  # 执行 PCA 降维
  x_reduced = pca.fit_transform(x_data)
  # 将数据转换为 tensor
  x_tensor = torch.tensor(x_reduced)

  file_path = 'D:\\yanjiu\\DrugRepSS\\tructural_feature\\dataset\\dis_drug_list(all).csv'  # 请替换为你的实际文件路径

  # 读取表格数据
  edge_index = pd.read_csv(file_path)

  # 将两列数据转换为 PyTorch Tensor
  tensor_edge_index = torch.tensor(edge_index.values)
  tensor_edge_index = torch.transpose(tensor_edge_index, 0, 1)

  y = torch.zeros_like(x_tensor, dtype=torch.int32)

  # 将前 100 个元素设置为 0
  y[:264] = 0

  # 将剩下的元素设置为 1
  y[264:] = 1

  # 构建 induced_graph_list
  induced_graph_list = {
    'x': x_tensor,
    'edge_index': tensor_edge_index,  # 使用原始示例中的 edge_index，这里暂时设置为 None
    'y': y
  }



  # 将字典转换为列表
  induced_graph_list = [(k, v) for k, v in induced_graph_list.items()]


  # 将列表转换为 Data 对象
  data = Data(**dict(induced_graph_list))

  # 将 Data 对象转换为 Batch 对象
  batch_data = Batch.from_data_list([data])

  maml, gnn, opt, lossfn = model_components()

  support_batch_preds = maml(batch_data, gnn)

  print(support_batch_preds)

###########################处理数据###########################
###################1.
# # 读取表格数据
# table_path = 'D:\\yanjiu\\图神经网络\\gnncover数据\\drug_vector.xlsx'  # 表格文件路径
# table_df = pd.read_excel(table_path)
#
# # 遍历每行数据，将所有列的数据合并到第一列中
# for index, row in table_df.iterrows():
#     merged_data = ' '.join([f"{i+1}:{value}" for i, value in enumerate(row)])
#     table_df.at[index, 'Merged Column'] = merged_data
#
# # 删除原始列
# table_df = table_df.drop(columns=table_df.columns[:-1])
#
# # 对表格中的一列数据进行处理
# table_df['Merged Column'] = '"' + table_df['Merged Column'].astype(str) + ' "'
# # 将结果保存到新的表格文件中
# output_path = 'D:\\yanjiu\\图神经网络\\gnncover数据\\drug_vector_合并.xlsx'
# table_df.to_excel(output_path, index=False)
#
# print("合并完成，结果已保存到:", output_path)
#

#
# # 读取表格1数据
# table1_path = 'C:\\university\\yanjiu\\图神经网络\\gnncover数据\\drug_vector_合并.xlsx'  # 表格1文件路径
# table1_df = pd.read_excel(table1_path)
#
# # 读取表格2数据
# table2_path = 'C:\\university\\yanjiu\\大模型数据集\\token为10\\训练集.xlsx'  # 表格2文件路径
# table2_df = pd.read_excel(table2_path)
#
# # 进行表格匹配
# merged_df = pd.merge(table2_df, table1_df, left_on=table2_df.columns[0], right_on=table1_df.columns[0], how='left')
#
# # 将匹配到的数据写入表格2的第四列
# table2_df[table2_df.columns[2]] = merged_df[table1_df.columns[1]]
#
# # 将结果保存到新的表格文件中
# output_path = 'C:\\university\\yanjiu\\图神经网络\\gnncover数据\\output训练集.xlsx'
# table2_df.to_excel(output_path, index=False)
#
# print("匹配并写入完成，结果已保存到:", output_path)
#
# import pandas as pd
#
# # 读取表格数据
# table_path = 'C:\\university\\yanjiu\\图神经网络\\gnncover数据\\output测试集.xlsx'  # 表格文件路径
# table_df = pd.read_excel(table_path)
#
# output_path = 'C:\\university\\yanjiu\\图神经网络\\gnncover数据\\output测试集.txt'
#
# # 打开 txt 文件准备写入数据
# with open(output_path, 'w') as file:
#     # 遍历表格的每一行数据
#     for index, row in table_df.iterrows():
#         # 将第一列的数据直接写入 txt 文件
#         file.write(str(row[table_df.columns[0]]) + ' ')
#
#         # 在第二列前面加上 'qid:'，然后写入 txt 文件
#         file.write('qid:' + str(row[table_df.columns[1]]) + ' ')
#
#         # 写入第三列的数据
#         file.write(str(row[table_df.columns[2]]))
#
#         # 写入换行符
#         file.write('\n')
#
# print("处理完成，结果已保存到:", output_path)
#
#
#
#
#
#
#


