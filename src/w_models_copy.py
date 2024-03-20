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
#from obj_parser import Mesh_obj


class Penetration_D(nn.Module):

    """
        给定人体网格和任意顶点,检测顶点是否在人体网格外部(未穿模):在外则判定为1,在内判定为0
        输入: 当前人体网格,任意布料顶点
        输出: 任意布料顶点是否在人体网格外部的结果向量,长度=输入的顶点规模
        核心思路: 基于人体动作训练一个广义的超平面笼罩在人体,用于分离人体和布料
    """    
    def __init__(self,cloth_features=3,linear_dim=[32,128,1],motion_size=48,device="cuda:0"):
        
        super(Penetration_D, self).__init__()

        # body_numverts=20813
        # cloth_numverts=26718

        # 1.body_mesh 使用 transformer + conv 网络结构
        # why transformer? 
        # -- 碰撞超平面的动态变化特征(随时间变化)、全局性(决定布料正反之类的)和稀疏性(布料某个区域是否发生穿模取决于少部分顶点组成的面决定)
        self.device = device
        self.motion_weights = GMW_Net(final_features=motion_size)
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
                # dropout?
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
        motion_weights = self.get_motion_weights(body_mesh).float() #[batchsize,motion_size]
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
    
    def get_motion_weights(self,body_mesh):
        
        motion_weights = self.motion_weights(body_mesh)
        
        return motion_weights

class MultiHeadAttention_conv(nn.Module):
    """
        网格检测中的多头注意力层: 采取类似NLP中的多头注意力层
        ------
        Net: 
        input -- conv1 -- conv2 -- adaptiveavepool -- multiHead -- output
        ------
        input: body_mesh:[numverts,3]
        out: features[out_features,1]
        ------
        params:
        
        numverts:顶点个数 --> adaptiveavepool结构允许任意规模的网格顶点输入
        numfeatures: 卷积后提取的特征向量的第一维度,可更改,预设为1024
        input_size: 输入多头网络的特征的属性维度 
                    -- 原计划将顶点数据直接输入多头网络[numverts,3],input_size=3三维顶点坐标;
                    -- 但顶点规模太大影响,故先用一个卷积网络提取原始网格特征,并将特征向量作为多头网络输入
                    -- 特征向量维度人为设置:[numfeatures=1024,9],为了适应下一步的多头网络输入
                        -- 但由于网格三维坐标特征,第二个维度一般取3,所以adaptive网络固定输出[1024*(9/3),3],再reshape为[1024,9]
        channels: 指定两个卷积网络的channels通道数
                    
        numheads:多头的header个数
        hidden_size:多头注意力的总的隐藏层特征个数 = heads_num * head_size(多头的个数*每个头对应的隐藏特征个数)
        out_features: 输出特征向量的维度
        ------        
        网络的输出:
        conv的输出结果:       
                    -- [batchsize,3,numfeatures*(input_size/3),1] 对应 [batchsize,1,numverts,3]经过两个卷积网络后输出的维度
                    -- 最终输出reshape为[batchsize,numfeatures,input_size]作为多头层的输入
        multihead的输出:
                    -- 输入[batchsize,numfeatures,input_size],将input_size使用多头机制作用+全连接层
                    -- 输出[batchsize,out_features=1024],对应输入人体网格的特征向量
                    
    """
    def __init__(self,num_features=1024,input_size=9,channels=[1,3,1,3],num_heads=2,hidden_size=4,out_features=1024,dropout=0.2,device="cuda:0"):
        
        super(MultiHeadAttention_conv, self).__init__()

        # numverts=20813
        self.blockNum = 2
        self.device = device
        self.num_features = num_features
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        
        #  编码器设定: 两个卷积块+激活层,每个卷积块包含卷积层、BN、pool层,其中针对网格数据采用了长宽不一的矩形conv核
        # [20809~20813,3] >conv> [3072,3] >reshape> [1024,9]  
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
        self.admaxpool = nn.AdaptiveAvgPool2d((num_features*input_size//3,3))
        self.relu = nn.PReLU()
        
        # transformer不直接以网格数据作为输入的原因: 顶点规模通常上w,直接用线性层参数规模量太大
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(num_features*hidden_size, out_features) #压缩成向量,提取特征
        
    def split_heads(self, x, batch_size):

        x = x.view(batch_size, -1, self.num_heads, self.head_size)
        return x.permute(0, 2, 1, 3) # x.shape:[batchsize,num_heads,num_features,head_size]
    
    def mesh_encoder(self,mesh):
        
        mesh = mesh.unsqueeze(1).float()
        batchsize = mesh.shape[0]
        x = self.convBlock1(mesh).view(batchsize,1,-1,3)
        x = self.convBlock2(x).view(batchsize,1,-1,3)
        # 自适应池化到[3072,3],就可以接受任意大小的网格输入了
        x = self.admaxpool(x)
        x = self.relu(x.view(batchsize,self.num_features,-1)) #[3072,3]>>[1024,9]
        return x
    
    def forward(self,mesh,mask=None):
        
        batch_size = mesh.size(0)
        x = self.mesh_encoder(mesh)
        
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / self.head_size**0.5
        
        if mask is not None:
            # mask==0的部位,对应在score的地方使用非零微小值填充,表示这部分权重很小,对输出没什么影响
            scores = scores.masked_fill(mask == 0, float("-1e20"))
        
        # attention_weights:[batchsize,numheads,num_features,num_features]
        attention_weights = torch.softmax(scores, dim=-1)
        # attention_weights:[batchsize,numheads,num_features,head_size]
        attended_values = self.dropout(torch.matmul(attention_weights, value)) # dropout避免过拟合
        
        #contiguous()实现张量语义和内存顺序的一致性维护
        # [batchsize,num_features,numheads,head_size]
        attended_values = attended_values.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.head_size) #[batchsize,1024,4]
        output = self.fc_out(attended_values.view(batch_size,-1))  # [batchsize,1024*4]>>[batchsize,1024]
        
        return output   

class GMW_Net(nn.Module):
    """
        GMW:get_motion_weights Net,使用一个“conv+多头注意力”网络块提取人体网格特征,得到一组权重向量
    """
    def __init__(self,numverts=20813,num_features=1024,input_size=9,channels=[1,3,1,3],num_heads=2,hidden_size=4,head_out_features=1024,final_features=48,dropout=0.2,device="cuda:0"):
        super(GMW_Net, self).__init__()

        # params:
        self.device = device
        
        self.multihead_attention = MultiHeadAttention_conv(num_features=num_features,input_size=input_size,channels=channels,
                                                           num_heads=num_heads,hidden_size=hidden_size,out_features=head_out_features,dropout=0.2,device=device)
        self.feed_forward = nn.ModuleList()
        self.feed_forward.append(
            nn.Sequential(
            nn.Linear(head_out_features, head_out_features),
            #nn.LayerNorm(head_out_features), 
            nn.BatchNorm1d(head_out_features), # 网格任务的话还是用BN会更好些?
            nn.ELU(),
            nn.Dropout(p=dropout)) # 添加dropout后输出预测概率值不同batch之间差异增大
        )
        self.feed_forward.append(nn.Linear(head_out_features,final_features))

        
    def forward(self,mesh,mask=None):

        x = self.multihead_attention(mesh)  #[batchsize,out_features]

        feed_forward_output = self.feed_forward[0](x)
        x = x + feed_forward_output
        x = self.feed_forward[1](x)  # [batchsize,final_features]

        return x


if __name__ == "__main__":
    # ctrl + / 实现批量注释
    device = "cuda:2"
    #cloth = Mesh_obj("garment.obj")

    # 穿模判别器网络测试
    net = Penetration_D()
    a = torch.randn(2,3)
    b = torch.randn(2,20813,3)
    start = time.time()*1000
    c = net(a,b)
    end = time.time()*1000
    print("最近邻计算时间:%f ms"%(end-start))
    #print(c)
    print(c.shape) #[batchsize,1]