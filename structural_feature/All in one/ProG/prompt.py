import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv
from torch_geometric.data import Batch, Data
from .utils import act
import warnings
from deprecated.sphinx import deprecated
import pandas as pd


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, pool=None, gnn_type='GAT'):
        super().__init__()
        #  # 选择图卷积层的类型
        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and TransformerConv')
        self.gnn_type = gnn_type
        # 初始化隐藏层维度和输出维度
        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim
        # 验证图卷积层数是否合理
        #创建的图卷积网络将包含两层图卷积层
        #当层数为2时，直接构建两个图卷积层；当层数大于2时，使用循环构建多个隐藏层和一个输出层。
        if gcn_layer_num < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(gcn_layer_num))
        elif gcn_layer_num == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, hid_dim)]
            for i in range(gcn_layer_num - 2):
                layers.append(GraphConv(hid_dim, hid_dim))
            layers.append(GraphConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

        if pool is None:
            # Returns batch-wise graph-level-outputs by averaging node features across the node dimension.
            # 通过对节点维度上的节点特征进行平均，返回逐批的图级输出。
            self.pool = global_mean_pool
        else:
            self.pool = pool

    #定义了一个前向传播的方法，接受输入参数 x（节点特征）、edge_index（图的边索引）和 batch（图中节点所属的批次信息）。
    def forward(self, x, edge_index, batch):
        for conv in self.conv_layers[0:-1]:
            x = conv(x.float(), edge_index) # 对输入特征 x 进行卷积操作，使用图的边索引 edge_index 进行图卷积计算。
            x = act(x) # 对卷积后的结果应用激活函数，这里默认使用 Leaky ReLU 激活函数。
            # 在训练过程中，使用伯努利分布中的样本，以概率p（默认为0.5）随机将输入张量的一些元素归零。
            x = F.dropout(x, training=self.training)
            #在训练过程中，使用伯努利分布中的样本，以概率 p（默认为0.5）随机将输入张量的一些元素归零，以防止过拟合。

        node_emb = self.conv_layers[-1](x, edge_index) # 使用最后一层 进行卷积
        numpy_data = node_emb.detach().numpy()

        # 将 NumPy 数组转换为 DataFrame
        df = pd.DataFrame(numpy_data)

        # 指定要保存的 Excel 文件路径
        excel_file_path = 'output_feature.xlsx'

        df = df.iloc[10:]
        # 将 DataFrame 数据保存到 Excel 文件中
        df.to_excel(excel_file_path, index=False, header=False)
        #使用模型中的最后一层卷积层对经过前面卷积和激活处理后的结果进行最终的卷积操作，得到节点嵌入 (node_emb)。
        graph_emb = self.pool(node_emb, batch.long()) # 进行池化
        return graph_emb


class LightPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group, group_num=1, inner_prune=None):
        """
        :param token_dim:
        :param token_num_per_group:
        :param group_num:   the total token number = token_num_per_group*group_num, in most cases, we let group_num=1.
                            In prompt_w_o_h mode for classification, we can let each class correspond to one group.
                            You can also assign each group as a prompt batch in some cases.

        :param prune_thre: if inner_prune is None, then all inner and cross prune will adopt this prune_thre
        :param isolate_tokens: if Trure, then inner tokens have no connection.
        :param inner_prune(剪枝): if inner_prune is not None, then cross prune adopt prune_thre whereas inner prune adopt inner_prune
        :param token_dim: 令牌的维度
        :param token_num_per_group: 每个组中的令牌数量
         :param group_num: 总令牌数量 = token_num_per_group * group_num，在大多数情况下，我们让 group_num=1。
              在分类的 prompt_w_o_h 模式中，我们可以让每个类对应一个组。
              在某些情况下，您还可以将每个组分配为提示批次。
         :param inner_prune(剪枝): 如果 inner_prune 不为 None，则交叉剪枝采用 prune_thre，而内部剪枝采用 inner_prune。
        """
        super(LightPrompt, self).__init__()

        self.inner_prune = inner_prune

        # 这段代码创建了一个torch.nn.ParameterList对象，其中包含了多个torch.nn.Parameter对象。
        # 每个torch.nn.Parameter对象都是一个可训练参数，它包含了一个空的torch.Tensor，其形状为(token_num_per_group, token_dim)。
        # 具体来说，以下是代码的解释：
        # group_num表示要创建的torch.nn.Parameter对象的数量。
        # token_num_per_group表示每个torch.nn.Parameter对象中的令牌数目。
        # token_dim表示每个令牌的特征维度。
        # 通过使用torch.nn.Parameter，可以将这些参数添加到模型的参数列表中，并且这些参数会自动进行梯度计算和更新。这对于模型的训练和优化非常重要。
        # 请注意，这段代码只是创建了一个参数列表，并且每个参数对象都是一个空的torch.Tensor，需要在模型训练过程中使用具体的数值进行初始化。
        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for i in range(group_num)])

        self.token_init(init_method="kaiming_uniform")

    def token_init(self, init_method="kaiming_uniform"):#Kaiming 均匀初始化方法
        if init_method == "kaiming_uniform":
            #Fills the input Tensor with values using a uniform distribution
            for token in self.token_list:
                torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")

    def inner_structure_update(self):
        return self.token_view()

    def token_view(self, ):
        """
        each token group is viewed as a prompt sub-graph.
        turn the all groups of tokens as a batch of prompt graphs.
        每个令牌组被视为一个提示子图。
        将所有令牌组转换为一个提示图批次。
        :return:
        """
        pg_list = []
        for i, tokens in enumerate(self.token_list):
            # inner link: token-->token
            # 内部连接: 令牌 --> 令牌
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1

            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)

            # 代码使用inner_adj.nonzero()得到了inner_adj中非零元素的索引，
            # 并通过.t().contiguous()对结果进行转置和连续化处理，最终得到了edge_index。
            edge_index = inner_adj.nonzero().t().contiguous()
            # edge_index通常表示图的边的索引，它是一个大小为2×E的张量，其中E是边的数量。
            # 在这个张量中，每一列表示一条边，两行分别表示这条边连接的两个节点的索引。
            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))
        # 将生成的提示图数据组成一个新的图批次返回
        pg_batch = Batch.from_data_list(pg_list)
        return pg_batch


class HeavyPrompt(LightPrompt):
    def __init__(self, token_dim, token_num, cross_prune=0.1, inner_prune=0.01):
        super(HeavyPrompt, self).__init__(token_dim, token_num, 1, inner_prune)  # only has one prompt graph.
        self.cross_prune = cross_prune

    def forward(self, graph_batch: Batch):
        """
        TODO: although it recieves graph batch, currently we only implement one-by-one computing instead of batch computing
        TODO: we will implement batch computing once we figure out the memory sharing mechanism within PyG
        :param graph_batch:
        :return:
        """
        # device = torch.device("cuda")
        # device = torch.device("cpu")

        pg = self.inner_structure_update()  # batch of prompt graph (currently only 1 prompt graph in the batch)

        inner_edge_index = pg.edge_index
        token_num = pg.x.shape[0]

        re_graph_list = []
        for g in Batch.to_data_list(graph_batch):
            g_edge_index = g.edge_index + token_num
            # pg_x = pg.x.to(device)
            # g_x = g.x.to(device)
            
            cross_dot = torch.mm(pg.x.float(), torch.transpose(g.x, 0, 1).float())
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)
            
            cross_edge_index = cross_adj.nonzero().t().contiguous()
            cross_edge_index[1] = cross_edge_index[1] + token_num
            
            x = torch.cat([pg.x, g.x], dim=0)
            y = g.y

            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            data = Data(x=x, edge_index=edge_index, y=y)
            re_graph_list.append(data)

        graphp_batch = Batch.from_data_list(re_graph_list)
        return graphp_batch

class FrontAndHead(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3):

        super().__init__()

        self.PG = HeavyPrompt(token_dim=input_dim, token_num=token_num, cross_prune=cross_prune,
                              inner_prune=inner_prune)

        if task_type == 'multi_label_classification':
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, num_classes),
                torch.nn.Softmax(dim=1))
        else:
            raise NotImplementedError

    def forward(self, graph_batch, gnn):
        prompted_graph = self.PG(graph_batch)

        graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        print(graph_emb)

        return graph_emb


@deprecated(version='1.0', reason="Pipeline is deprecated, use FrontAndHead instead")
class Pipeline(torch.nn.Module):
    def __init__(self, input_dim, dataname, gcn_layer_num=2, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3, gnn_type='TransformerConv'):
        warnings.warn("deprecated", DeprecationWarning)

        super().__init__()
        # load pre-trained GNN
        self.gnn = GNN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=gcn_layer_num, gnn_type=gnn_type)
        pre_train_path = './pre_trained_gnn/{}.GraphCL.{}.pth'.format(dataname, gnn_type)
        self.gnn.load_state_dict(torch.load(pre_train_path))
        print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))
        for p in self.gnn.parameters():
            p.requires_grad = False

        self.PG = HeavyPrompt(token_dim=input_dim, token_num=token_num, cross_prune=cross_prune,
                              inner_prune=inner_prune)

        if task_type == 'multi_label_classification':
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, num_classes),
                torch.nn.Softmax(dim=1))
        else:
            raise NotImplementedError

    def forward(self, graph_batch: Batch):
        prompted_graph = self.PG(graph_batch)
        graph_emb = self.gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)

        pre = self.answering(graph_emb)

        return pre



if __name__ == '__main__':
    pass
