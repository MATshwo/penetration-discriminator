import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch_geometric.nn import EdgeConv,GCNConv, global_max_pool,GATConv
from torch_geometric.utils import dropout_edge, get_laplacian
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch_geometric.transforms import FaceToEdge
from torch_geometric.data import Data
import time
from src.obj_parser import Mesh_obj
# from obj_parser import Mesh_obj  # 单独运行w_models_sim时使用本行，注释上一行

# 此处为w_models_copy模型的简化版,感觉简化后的没有之前的效果好

class Penetration_D(nn.Module):

    """
        给定人体网格和任意顶点,检测顶点是否在人体网格外部(未穿模):在外则判定为1,在内判定为0
        输入: 当前人体网格,任意布料顶点
        输出: 任意布料顶点是否在人体网格外部的结果向量,长度=输入的顶点规模
        核心思路: 基于人体动作训练一个广义的超平面笼罩在人体,用于分离人体和布料
    """    
    def __init__(self,cloth_features=3,linear_dim=[16,64,1],motion_size=32,device="cuda:0"):
        
        super(Penetration_D, self).__init__()

        # body_numverts=20813
        # cloth_numverts=26718

        # 1.body_mesh 使用  conv 网络结构

        self.device = device
        self.motion_weights = Motion_conv(out_features=motion_size)
        self.motion_size = motion_size
        linear_dim.insert(0,cloth_features) # 布料网格的特征使用三维坐标

        # 2.Fusion_net for mixing the motion_weights to discriminator
        self.fusion_depth = len(linear_dim)-2
        self.fusion = nn.ModuleList()
        self.fus_bn = nn.ModuleList()
        for idx in range(self.fusion_depth):
            tmp = nn.ModuleList()
            for i in range(motion_size):
                tmp.append(nn.Linear(linear_dim[idx],linear_dim[idx+1]))
            self.fusion.append(tmp)
            self.fus_bn.append(nn.BatchNorm1d(linear_dim[idx+1]))    

        self.fus_ac = nn.PReLU()

        # 3.布料网格使用线性层提取特征,因为是逐顶点判断,不需要对整个布料网格
        self.cloth_features = cloth_features

        # 4.最后使用线性层将动作权重与布料特征混合,并输出预测概率值
        self.out_linear = nn.Sequential(
            nn.Linear(linear_dim[-2],linear_dim[-1]),
            nn.Sigmoid()
        )

    def forward(self,vertex_pos,body_mesh):

        batchsize = vertex_pos.shape[0] 
        body_mesh = body_mesh.view(batchsize,-1,3).float()
        motion_weights = self.motion_weights(body_mesh).float()
        w_sum = torch.sum(motion_weights,dim=1) #[2,1]
        x = vertex_pos.view(batchsize,3).float()

        for idx in range(self.fusion_depth):
            sums = 0
            for j in range(self.motion_size):
                # print((motion_weights[:,j] / w_sum).shape) # [batchsize]
                # print(self.fusion[idx][j](x).shape)        # [batchsize,*]
                wx = self.fusion[idx][j](x) * (motion_weights[:,j] / w_sum).unsqueeze(-1)       
                sums = sums + wx
            x = self.fus_bn[idx](sums)
            x = self.fus_ac(x)
            # print(x.shape)
        p = self.out_linear(x).view(batchsize,-1) # [bachsize,1]
        return p  
    
class Motion_conv(nn.Module):
    """
        ------
        Net: 
        input -- conv1 -- conv2 -- adaptiveavepool -- linear -- output
        ------
        input: body_mesh:[numverts,3]
        out: features[out_features,1]
        ------
        params:
        
        numverts:顶点个数 --> adaptiveavepool结构允许任意规模的网格顶点输入
        numfeatures: 卷积后提取的特征向量的第一维度,可更改,预设为1024
        in_features: 1024
        channels: 指定两个卷积网络的channels通道数

        out_features: 输出特征向量的维度
        ------        
        网络的输出:
        conv的输出结果:       
            [batch,numverts,3] >> conv1 >> conv2 >> SPP >> [batch,1024,1] >> linear >> [batch,32]

    """
    def __init__(self,num_features=1024,in_features=1024,channels=[1,3,1,3],out_features=32,dropout=0.2,device="cuda:2"):
        
        super(Motion_conv, self).__init__()

        # numverts=20813
        self.blockNum = 2
        self.device = device
        self.num_features = num_features

        #  编码器设定: 两个卷积块+激活层,每个卷积块包含卷积层、BN、pool层,其中针对网格数据采用了长宽不一的矩形conv核
        # [20809~20813,3] >conv> [1024,1] > linear > [32,1]  
        h_kernel = [409,330]
        w_kernel = 3
        stride_list = [6,1]
        
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=channels[0],out_channels=channels[1],kernel_size=(h_kernel[0],w_kernel),stride=(stride_list[0],1)),
            nn.BatchNorm2d(channels[1]),
            # nn.MaxPool2d(kernel_size=(3,1),stride=(1,1),padding=(1,0))
        )
        
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels= channels[2],out_channels=channels[3],kernel_size=(h_kernel[1],w_kernel),stride=(stride_list[1],1)),
            nn.BatchNorm2d(channels[3]),
            # 添加池化层导致提取的特征向量相邻维度的值重复(图像领域池化有效,但网格任务考虑弃用池化层)
            #nn.MaxPool2d(kernel_size=(3,1),stride=(1,1),padding=(1,0)) 
        )
        self.admaxpool = nn.AdaptiveAvgPool2d((1024,1))
        
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features, out_features),
            # nn.BatchNorm1d(in_features),
            nn.ELU(),
            nn.Dropout(p=dropout) # 添加dropout后输出预测概率值不同batch之间差异增大
        )
        

    def mesh_encoder(self,mesh):
        
        mesh = mesh.unsqueeze(1).float()
        batchsize = mesh.shape[0]
        x = self.convBlock1(mesh).view(batchsize,1,-1,3)
        x = self.convBlock2(x).view(batchsize,1,-1,3)
        # 自适应池化到[1024,1],就可以接受任意大小的网格输入了
        x = self.admaxpool(x).view(batchsize,-1) # 输出[batchsize,1024]
        return x
    
    def forward(self,mesh,mask=None):
        
        # batch_size = mesh.size(0)
        x = self.mesh_encoder(mesh)
        output = self.feed_forward(x)
        # print(output.shape)
        return output   



if __name__ == "__main__":
    # ctrl + / 实现批量注释
    device = "cuda:2"
    #cloth = Mesh_obj("garment.obj")

    # 穿模判别器网络测试
    net = Penetration_D()
    # print(net)
    a = torch.randn(5,3)
    b = torch.randn(5,20813,3)
    start = time.time()*1000
    c = net(a,b)
    end = time.time()*1000
    print("最近邻计算时间:%f ms"%(end-start))
    print(c)
    print(c.shape) #[batchsize,1]