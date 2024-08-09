from torch import nn, optim
import torch
from copy import deepcopy
from ProG.utils import seed, seed_everything
from random import shuffle
from ProG.meta import MAML
from torch_geometric.loader import DataLoader

seed_everything(seed)

from ProG.prompt import GNN, FrontAndHead


def meta_test_adam(meta_test_task_id_list,
                   dataname,
                   K_shot,
                   seed,
                   maml, gnn,
                   adapt_steps_meta_test,
                   lossfn):
    # meta-testing
    if len(meta_test_task_id_list) < 2:
        raise AttributeError("\ttask_id_list should contain at leat two tasks!")

    shuffle(meta_test_task_id_list)

    task_pairs = [(meta_test_task_id_list[i], meta_test_task_id_list[i + 1]) for i in
                  range(0, len(meta_test_task_id_list) - 1, 2)]

    for task_1, task_2, support, query, _ in load_tasks('test', task_pairs, dataname, K_shot, seed):

        test_model = deepcopy(maml.module)
        test_opi = optim.Adam(filter(lambda p: p.requires_grad, test_model.parameters()),
                              lr=0.001,
                              weight_decay=0.00001)
        #打开训练模式
        test_model.train()

        support_loader = DataLoader(support.to_data_list(), batch_size=10, shuffle=True)
        query_loader = DataLoader(query.to_data_list(), batch_size=10, shuffle=True)

        for _ in range(adapt_steps_meta_test):
            running_loss = 0.
            for batch_id, support_batch in enumerate(support_loader):
                support_preds = test_model(support_batch, gnn)
                support_loss = lossfn(support_preds, support_batch.y)
                test_opi.zero_grad()
                support_loss.backward()
                test_opi.step()
                running_loss += support_loss.item()

                if batch_id == len(support_loader) - 1:
                    last_loss = running_loss / len(support_loader)  # loss per batch
                    print('{}/{} training loss: {:.8f}'.format(_, adapt_steps_meta_test, last_loss))
                    running_loss = 0.

        #评估模式
        test_model.eval()
        # acc_f1_over_batches(query_loader, test_model.PG, gnn, test_model.answering, 2, 'multi_class_classification','cpu')
        ## DO NOT DELETE the following content!
        # metric = torchmetrics.classification.Accuracy(task="binary")  # , num_labels=2)
        # for batch_id, query_batch in enumerate(query_loader):
        #     query_preds = test_model(query_batch,gnn)
        #     pre_class = torch.argmax(query_preds, dim=1)
        #     acc = metric(pre_class, query_batch.y)
        #     # print(f"Accuracy on batch {batch_id}: {acc}")
        #
        # acc = metric.compute()
        # print("""\ttask pair ({}, {}) | Acc: {:.4} """.format(task_1, task_2, acc))
        # metric.reset()

#基于MAML算法进行元训练
def meta_train_maml(epoch, maml, gnn, lossfn, opt, meta_train_task_id_list, dataname, adapt_steps=2, K_shot=100):
    if len(meta_train_task_id_list) < 2:
        raise AttributeError("\ttask_id_list should contain at leat two tasks!")

    shuffle(meta_train_task_id_list)
    #将meta_train_task_id_list中的元素两两配对，得到任务对的列表
    task_pairs = [(meta_train_task_id_list[i], meta_train_task_id_list[i + 1]) for i in
                  range(0, len(meta_train_task_id_list) - 1, 2)]

    # meta-training
    #循环进行meta训练的epoch次数
    for ep in range(epoch):
        meta_train_loss = 0.0
        pair_count = 0
        PrintN = 10

        for task_1, task_2, support, query, total_num in load_tasks('train', task_pairs, dataname, K_shot, seed):
            pair_count = pair_count + 1

            learner = maml.clone()

            #创建一个DataLoader对象，用于加载支持集数据，设置批大小为10，并不打乱数据顺序
            support_loader = DataLoader(support.to_data_list(), batch_size=10, shuffle=False)
            query_loader = DataLoader(query.to_data_list(), batch_size=10, shuffle=False)
            #循环进行自适应步骤
            for j in range(adapt_steps):  # adaptation_steps
                running_loss = 0.
                support_loss = 0.
                #循环遍历支持集数据的批次ID和对应的批次数据
                for batch_id, support_batch in enumerate(support_loader):
                    #将支持集数据和gnn传递给learner进行前向传播，得到支持集数据的预测结果
                    support_batch_preds = learner(support_batch, gnn)
                    #计算支持集数据的损失，将预测结果和真实标签传递给lossfn函数
                    support_batch_loss = lossfn(support_batch_preds, support_batch.y)
                    # learner.adapt(support_batch_loss)
                    running_loss += support_batch_loss.item()
                    support_loss += support_batch_loss
                    if (batch_id + 1) % PrintN == 0:  # report every PrintN updates
                        last_loss = running_loss / PrintN  # loss per batch
                        print('adapt {}/{} | batch {}/{} | loss: {:.8f}'.format(j + 1, adapt_steps,
                                                                                batch_id + 1,
                                                                                len(support_loader),
                                                                                last_loss))

                        running_loss = 0.

                support_loss = support_loss / len(support_loader)
                learner.adapt(support_loss)

            running_loss, query_loss = 0., 0.
            for batch_id, query_batch in enumerate(query_loader):
                query_batch_preds = learner(query_batch, gnn)
                query_batch_loss = lossfn(query_batch_preds, query_batch.y)
                query_loss += query_batch_loss
                running_loss += query_batch_loss
                if (batch_id + 1) % PrintN == 0:
                    last_loss = running_loss / PrintN
                    print('query loss batch {}/{} | loss: {:.8f}'.format(batch_id + 1,
                                                                         len(query_loader),
                                                                         last_loss))

                    running_loss = 0.

            query_loss = query_loss / len(query_loader)
            meta_train_loss += query_loss

        print('meta_train_loss @ epoch {}/{}: {}'.format(ep, epoch, meta_train_loss.item()))
        meta_train_loss = meta_train_loss / len(meta_train_task_id_list)
        opt.zero_grad()
        meta_train_loss.backward()
        opt.step()


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
    #创建一个FrontAndHead模型对象，设置输入维度为100、隐藏维度为100、类别数为2，任务类型为多标签分类，令牌数量为10，交叉剪枝率为0.1，内部剪枝率为0.3。
    #重提示
    model = FrontAndHead(input_dim=100, hid_dim=100, num_classes=2,  # 0 or 1
                         task_type="multi_label_classification",
                         token_num=0, cross_prune=0.1, inner_prune=0.3)

    # load pre-trained GNN
    gnn = GNN(input_dim=100, hid_dim=100, out_dim=100, gcn_layer_num=2, gnn_type='TransformerConv')
    pre_train_path = './pre_trained_gnn/{}.GraphCL.{}.pth'.format(dataname, 'TransformerConv')
    gnn.load_state_dict(torch.load(pre_train_path))#权重加载到gnn中
    print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))
    #将GNN模型的参数设置为不需要梯度计算
    for p in gnn.parameters():
        p.requires_grad = False

    maml = MAML(model, lr=adapt_lr, first_order=False, allow_nograd=True)

    opt = optim.Adam(filter(lambda p: p.requires_grad, maml.parameters()), meta_lr)

    lossfn = nn.CrossEntropyLoss(reduction='mean')

    return maml, gnn, opt, lossfn


if __name__ == '__main__':
    dataname = 'CiteSeer'
    # node-level: 0 1 2 3 4 5
    # edge-level: 6 7 8 9 10 11
    # graph-level: 12 13 14 15 16 17
    meta_train_task_id_list = [0, 1, 2, 3]
    meta_test_task_id_list = [4, 5]

    pre_train_method = 'GraphCL'
    gnn_type = ['TransformerConv']

    maml, gnn, opt, lossfn = model_components()

    # meta training on source tasks
    meta_train_maml(20, maml, gnn, lossfn, opt, meta_train_task_id_list,
                    dataname, adapt_steps=2, K_shot=100)

    # meta testing on target tasks
    adapt_steps_meta_test = 2  # 00  # 50
    meta_test_adam(meta_test_task_id_list, dataname, 100, seed, maml, gnn,
                   adapt_steps_meta_test, lossfn)
