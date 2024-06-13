import sys
import numpy as np
import torch

sys.path.extend(['../'])
from graph import tools

num_node = 25
# 每个节点和自身的连接，可以叫做自连接
self_link = [(i, i) for i in range(num_node)]
# 人体骨架节点自身的连接，就叫内连接把，参照NTU RGB+D骨架图可理解
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]

# 这个操作的目的是将索引值从以 1 开始的形式转换为以 0 开始的形式。在计算机编程中，索引通常是从 0 开始的，而不是从 1 开始的。
# 这样做可以使索引值与实际的节点在列表或数组中的位置对应起来，方便后续的处理和操作。
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]

# 人体内连接的反方向连接，outward可以叫外连接
outward = [(j, i) for (i, j) in inward]

# 邻居节点，无论外连接还是内连接，这两个节点必定是邻居节点
neighbor = inward + outward

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
# class Graph_original:
#     def __init__(self, labeling_mode='spatial'):
#         self.num_node = num_node
#         self.self_link = self_link
#         self.inward = inward
#         self.outward = outward
#         self.neighbor = neighbor
#         self.A = self.get_adjacency_matrix(labeling_mode)
#
#     def get_adjacency_matrix(self, labeling_mode=None):
#         if labeling_mode is None:
#             return self.A
#         if labeling_mode == 'spatial':
#             A = tools.get_spatial_graph(num_node, self_link, inward, outward)
#         else:
#             raise ValueError()
#         return A

# 以中间为界限，左边有10个节点，右边有10个节点
num_node_left = 10
joint_part_body_left = [
    #  head  1
    np.array([3,4])-1,
    # left arm  2
    np.array([9,10,11,12])-1,
    # right arm  3
    np.array([5,6,7,8])-1,
    # left hand  4
    np.array([24,25])-1,
    # right hand  5
    np.array([22,23])-1,
    # qugan     6
    np.array([21, 2, 1]) - 1,
    # left leg  7
    np.array([17,18,19])-1,
    # right leg  8
    np.array([13,14,15])-1,
    # left foot 9
    np.array([20])-1,
    # right foot    0
    np.array([16])-1
]
# 这里的每个节点i其实是 一个group，比如1代表头部的节点，上面定义好的
self_link_left = [ (i,i) for i in range(num_node_left) ]
# 同理这里的(1,6)代表的就是头部和躯干的连接，依次往后 这里可以参照 cross-level的图
left_link_ori = [(1,6),(2,6),(3,6),(4,2),(5,3),(7,6),(8,6),(9,7),(10,8)]
# 从0开始与索引匹配，上面有解释
left_link = [(i-1,j-1) for (i,j) in left_link_ori] # 哪些身体部分是相互连接的
# 与内连接反向，形成外连接
left_link_outward = [(j,i) for (i,j) in left_link]


# 左右中间关于身体部位的定义为何不同，例如腿和脚，为什么分手臂的上部分和下部分？
num_node_middle = 12
joint_part_body_middle = [
    # head 1
    np.array([3,4])-1,
    # left arm up 2
    np.array([9,10])-1,
    # right arm up 3
    np.array([5,6])-1,
    # left arm down 4
    np.array([11,12])-1,
    # right arm down 5
    np.array([7,8])-1,
    # left hand 6
    np.array([24,25])-1,
    # right hand 7
    np.array([22,23])-1,
    # qugan 8
    np.array([21, 2, 1]) - 1,
    # left leg 9
    np.array([17,18,19])-1,
    # right leg 10
    np.array([13,14,15])-1,
    # left foot 11
    np.array([20])-1,
    # right foot 12
    np.array([16])-1
]
self_link_middle = [ (i,i) for i in range(num_node_middle) ]
# 这里的每个节点i其实是 一个group，比如1代表头部的节点，上面定义好的
self_link_middle = [ (i,i) for i in range(num_node_left) ]
# 同理这里的(1,6)代表的就是头部和躯干的连接，依次往后
middle_link_ori = [(1,8),(2,8),(3,8),(6,4),(4,2),(7,5),(5,3),(9,8),(10,8),(11,9),(12,10)]
# 从0开始与索引匹配，上面有解释
middle_link = [(i-1,j-1) for (i,j) in middle_link_ori] # 哪些身体部分是相互连接的
# 与内连接反向，形成外连接
middle_link_outward = [(j,i) for (i,j) in middle_link]



# 右连接的一个实现
num_node_right = 10
joint_part_body_right = [
    # head 1
    np.array([3,4])-1,
    # left arm 2
    np.array([9,10,11,12])-1,
    # right arm 3
    np.array([5,6,7,8])-1,
    # left hand 4
    np.array([24,25])-1,
    # right hand 5
    np.array([22,23])-1,
    # qugan 6
    np.array([21, 2, 1]) - 1,
    # left leg 7
    np.array([17,18])-1,
    # right leg 8
    np.array([13,14])-1,
    # left foot 9
    np.array([19,20])-1,
    # right foot 10
    np.array([15,16])-1
]
# 这里的每个节点i其实是 一个group，比如1代表头部的节点，上面定义好的
self_link_right = [ (i,i) for i in range(num_node_right) ]
# 同理这里的(1,6)代表的就是头部和躯干的连接，依次往后
right_link_ori = [(1,6),(2,6),(3,6),(4,2),(5,3),(7,6),(8,6),(9,7),(10,8)]
# 从0开始与索引匹配，上面有解释
right_link = [(i-1,j-1) for (i,j) in right_link_ori] # 哪些身体部分是相互连接的
# 与内连接反向，形成外连接
right_link_outward = [(j,i) for (i,j) in right_link]





class Graph_left:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            # 三种基本的邻接矩阵
            A1,A2,A3 = tools.get_spatial_graph(num_node, self_link, inward, outward)
            # 这里分别可以从左连接、右连接、中间连接分别建立邻接矩阵 再进行融合,得到一个新的邻接矩阵A4
            A4 = tools.get_left(num_node_left,self_link_left,left_link,left_link_outward,joint_part_body_left)
            # A4 = tools.get_left(num_node_middle, self_link_middle, middle_link, middle_link_outward, joint_part_body_middle)
            # A4 = tools.get_left(num_node_right, self_link_right, right_link, right_link_outward, joint_part_body_right)
            # size mismatch for l1.gcn1.PA: copying a param with shape torch.Size([3, 25, 25]) from checkpoint,
            # the shape in current model is torch.Size([4, 25, 25])
            # A的维度如果加上A4，会变成3，如果想用预训练权重，维度必须为3
            A = np.stack((A1,A2,A3,A4)).reshape(-1,num_node,num_node)
        else:
            raise ValueError()
        return A

class Graph_middle:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            # 三种基本的邻接矩阵
            A1,A2,A3 = tools.get_spatial_graph(num_node, self_link, inward, outward)
            # 这里分别可以从左连接、右连接、中间连接分别建立邻接矩阵 再进行融合,得到一个新的邻接矩阵A4
            # A4 = tools.get_left(num_node_left,self_link_left,left_link,left_link_outward,joint_part_body_left)
            A4 = tools.get_left(num_node_middle, self_link_middle, middle_link, middle_link_outward, joint_part_body_middle)
            # A4 = tools.get_left(num_node_right, self_link_right, right_link, right_link_outward, joint_part_body_right)
            # size mismatch for l1.gcn1.PA: copying a param with shape torch.Size([3, 25, 25]) from checkpoint,
            # the shape in current model is torch.Size([4, 25, 25])
            # A的维度如果加上A4，会变成3，如果想用预训练权重，维度必须为3
            A = np.stack((A1,A2,A3,A4)).reshape(-1,num_node,num_node)
        else:
            raise ValueError()
        return A
class Graph_right:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            # 三种基本的邻接矩阵
            A1,A2,A3 = tools.get_spatial_graph(num_node, self_link, inward, outward)
            # 这里分别可以从左连接、右连接、中间连接分别建立邻接矩阵 再进行融合,得到一个新的邻接矩阵A4
            # A4 = tools.get_left(num_node_left,self_link_left,left_link,left_link_outward,joint_part_body_left)
            # A4 = tools.get_left(num_node_middle, self_link_middle, middle_link, middle_link_outward, joint_part_body_middle)
            A4 = tools.get_left(num_node_right, self_link_right, right_link, right_link_outward, joint_part_body_right)
            # size mismatch for l1.gcn1.PA: copying a param with shape torch.Size([3, 25, 25]) from checkpoint,
            # the shape in current model is torch.Size([4, 25, 25])
            # A的维度如果加上A4，会变成3，如果想用预训练权重，维度必须为3
            A = np.stack((A1,A2,A3,A4)).reshape(-1,num_node,num_node)
        else:
            raise ValueError()
        return A

