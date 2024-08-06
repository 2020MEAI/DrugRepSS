import numpy as np

# 这个图是一个有向无环图(DAG)
# -1 代表什么也不连
# 图用二维列表，第一行代表节点编号，第二行为对应节点的指向节点
graph = [
    [0,0,1,2,3],
    [1,2,3,3,4]
]

# 定义5个节点的初始特征值
embeddings = [
    [1,2,3],
    [2,6,5],
    [2,3,7],
    [7,8,6],
    [1,0,0]
]

# 定义聚合的权重w全为1
w = [1,1,1,1,1]

# 下面开始图神经网络的聚合过程（训练过程）
# 在这里每个节点只按照方向聚合一层
for i in range(len(graph[0])):  # 每个节点
    # 先寻找指向节点i的节点们
    temp_roots = []
    for j, eve in enumerate(graph[1]):
        if eve == i:
            temp_roots.append(graph[0][j])
    temp_roots.append(i)
    # 此时temp_roots存储了节点i的根节点以及节点i自己的编号
    around = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    # 将temp_roots中的几点对应的around替换成当前的embedding
    for every_node_id in temp_roots:
        around[every_node_id] = embeddings[every_node_id]
    # 开始更新节点i的特征：自己之前的特征+周围节点特征的平均
    embeddings[i] = np.matmul(np.array(w),np.array(around))

# 输出更新一层后的embeddings
print(embeddings)
