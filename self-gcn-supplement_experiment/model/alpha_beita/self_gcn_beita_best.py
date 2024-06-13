"""
    修改的地方：
        1、 Bi_inter 中 conv_c 进行了分组卷积，groups = self.inter_channels
        2、对 V 也采用了压缩激励
"""
import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'
        # in_channels == out_channels
        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

class Bi_Inter(nn.Module):
    def __init__(self,in_channels):
        super(Bi_Inter, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        self.conv_c = nn.Sequential(
            nn.Conv2d(self.in_channels,self.inter_channels ,kernel_size=1,groups=self.inter_channels),
            nn.BatchNorm2d(self.inter_channels),
            nn.GELU(),
            nn.Conv2d(self.inter_channels,self.in_channels,kernel_size=1,groups=self.inter_channels)
        )
        self.conv_s = nn.Conv2d(self.in_channels, 1, kernel_size=1)
        # self.conv_v = nn.Sequential(
        #     nn.Conv2d(25,5,kernel_size=1,groups=5),
        #     nn.BatchNorm2d(5),
        #     nn.GELU(),
        #     nn.Conv2d(5,25,kernel_size=1)
        # )

        self.conv_t = nn.Conv2d(self.in_channels, 1, kernel_size=(9,1), padding=(4,0))

        self.sigmoid = nn.Sigmoid()

    def forward(self,x,mode):
        N,C,T,V = x.size()
        if mode == 'channel':
            x_res = x.mean(-1, keepdim=True).mean(-2, keepdim=True) # N,C,1,1
            x_res = self.sigmoid(self.conv_c(x_res))
        elif mode == 'spatial':
            x_res = x.mean(2, keepdim=True)  # N,C,1,V
            x_res = self.sigmoid(self.conv_s(x_res))
        else:
            x_res = x.mean(-1,keepdim=True) # N,C,T,1
            x_res = self.sigmoid(self.conv_t(x_res))
        return x_res


class SelfGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(SelfGCN_Block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv1_1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2_2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels,self.out_channels,kernel_size=1)
        self.att_t = nn.Conv2d(self.rel_channels,self.out_channels,kernel_size=1)
        self.Bi_inter = Bi_Inter(self.out_channels)

        # self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(2 * self.rel_channels, self.rel_channels, kernel_size=1, groups=self.rel_channels)
        self.beita = torch.tensor(0.391797364, requires_grad=False)
        self.tanh = nn.Tanh()
        self.bn = nn.BatchNorm2d(self.out_channels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        N,C,T,V = x.size()
        x1, x2 = self.conv1(x), self.conv2(x)
        x1_mean, x2_mean = x1.mean(-2), x2.mean(-2)
        x1_max, _ = x1.max(-2)
        x2_max, _ = x2.max(-2)
        x_max_res = x1_max.unsqueeze(-1) - x2_max.unsqueeze(-2)
        x1_mean_res = x1_mean.unsqueeze(-1) - x2_mean.unsqueeze(-2)
        N1, C1, V1, V2 = x1_mean_res.shape
        x_result = torch.cat((x1_mean_res, x_max_res), 2)  # (N,rel_channels,2V,V)
        x_result = x_result.reshape(N1, 2 * C1, V1, V2)
        x_1 = self.tanh(self.conv5(x_result)) # (N,rel_channels,V,V)
        x_1 = self.conv4(x_1) * alpha # out_channels

        x3 = self.conv3(x) # out_channels
        # A.unsqueeze(0).unsqueeze(0).shape : (1,1,25,25)

        x1_res = x_1 + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,out_channels,V,V

        q , k = x1, x2 # rel_channels
        att = self.tanh(torch.einsum('nctu,nctv->ncuv',[q,k])/T) # rel_channels
        att = self.att_t(att) # out_channels
        global_res = torch.einsum('nctu,ncuv->nctv',x3,att)
        # 根据 self-attention提取的特征做通道交互
        c_att = self.Bi_inter(global_res,'channel')
        x1_res = c_att * x1_res

        x1_res = torch.einsum('ncuv,nctv->nctu', x1_res, x3) # out_channels
        s_att = self.Bi_inter(x1_res,'spatial')
        global_res = global_res * s_att
        x1_res = x1_res + global_res



        # N, C, T, V = x1_1.size()
        x1_1, x2_2 = self.conv1_1(x), self.conv2_2(x)
        x1_trans = x1_1.permute(0, 2, 3, 1).contiguous()  # (N,T,V,C)
        x2_trans = x2_2.permute(0, 2, 1, 3).contiguous()  # (N,T,C,V)
        xt = self.tanh(torch.einsum('ntvc,ntcu->ntvu',x1_trans,x2_trans)/self.rel_channels) #(N,T,V,V)
        t_res = torch.einsum('nctu,ntuv->nctv',x3,xt) # 特定于时间的结果(N,C,T,V)
        res = t_res.permute(0, 2, 3, 1).contiguous() * self.beita + x1_res.permute(0, 2, 3, 1).contiguous()  # (N,T,V,C)
        res = res.permute(0, 3, 1, 2).contiguous()

        return res

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        # for i in range(3)
        for i in range(self.num_subset):
            self.convs.append(SelfGCN_Block(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = torch.tensor(0.00000156613, requires_grad=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        # self.num_subset=A.shape[0] = 3
        for i in range(self.num_subset):
            # self.convs[0](x,A[0],self.alpha)   A[0] 是 I
            # self.convs[1](x,A[1],self.alpha)   A[1] 是inward
            # self.convs[2](x,A[2],self.alpha)   A[2] 是outward
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        #
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride) # 如果输入的通道跟输出的通道不一致的话，那么残差的通道要与输出的通道一致，则使用1x1的卷积

    def forward(self, x):

        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            # graph: graph.ntu_rgb_d.Graph
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25 包含了 I，inward,outward

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        # (N,M*V*C,T)
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        # (N*M,C,T,V)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        # N,M,C_new,T*V
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1) # N,M,C_new -- > N,C_new
        x = self.drop_out(x)

        return self.fc(x)
