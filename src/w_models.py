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

from src.obj_parser import Mesh_obj

class ResBlock(nn.Module):

    """
        input:[batchsize,in_channel,h,w] >> output:[batchsize,out_channel,h,w]
    """

    def __init__(self,in_channel,out_channel):

        super(ResBlock,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(in_channel,out_channel,3,1,1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_channel,out_channel,3,1,1)
        self.bn2 = nn.BatchNorm2d(out_channel)
    
    def forward(self,x0):
        
        residual = self.conv1(x0)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x0 + residual

class UpsampleBlock(nn.Module):
    
    """
        Upsample: input:[channels,H,W] >> out:[channels,H*r,W*r]
    """

    def __init__(self,in_channel=64,Upscale=2):
        super(UpsampleBlock,self).__init__()
        
        self.Upscale = Upscale
        self.conv = nn.Conv2d(in_channel,in_channel*(Upscale**2),kernel_size=3,padding=1)
        self.Upsample = nn.PixelShuffle(Upscale)
        self.prelu = nn.PReLU()
    
    def forward(self,x0):
        x = self.conv(x0)
        x = self.Upsample(x)
        x = self.prelu(x)
        
        return x

class MeshRefineNet_G(nn.Module):
    """
        建立 平滑后的网格隐向量 >> 原始网格隐向量 的映射生成
        生成器接受平滑网格隐空间的向量输入: 64*1
        输出原始网格隐空间的向量: 1024*1
        判别器判别对象分别是这两个隐向量基于两个平滑/原始预训练网络生成的布料网格模型对象,并将结果用于指导生成器训练
        input: [batchsize,64] 64 >> 256 >> 16*16*1 >> 16*16*3 >> 16*16*64 >> 32*32*64 >> 64*64*64 >> 64*64*1 
        block:{conv,bn,prelu}
        nn.conv2d():输入数据的格式定义[batchsize,channels,height,weight],注意channels非最后一位
    """
    def __init__(self,dimList=[64,256,4096],Res_channelList=[1,3,64,64,64,64,64,64],Upscale=[2,2],device="cuda:0"):
    
        super(MeshRefineNet_G,self).__init__()

        self.device = device
        self.Res_channelList = Res_channelList

        self.linear = nn.Linear(dimList[0],dimList[1])
        self.inRes = nn.Sequential(
            nn.Conv2d(in_channels=Res_channelList[0],out_channels=Res_channelList[1],kernel_size=5,padding=2),
            nn.PReLU()
        )

        ResNet = []
        for i in range(len(Res_channelList)-3):
            ResNet.append(ResBlock(in_channel=Res_channelList[i+1],out_channel=Res_channelList[i+2]))
        self.ResNet = nn.Sequential(*ResNet)

        self.outRes = nn.Sequential(

            nn.Conv2d(in_channels=Res_channelList[-2],out_channels=Res_channelList[-1],kernel_size=3,padding=1),
            nn.BatchNorm2d(Res_channelList[-1])
        )

        UpNet = []
        for i in range(len(Upscale)):
            UpNet.append(UpsampleBlock(Res_channelList[-1],Upscale=Upscale[i]))
        self.UpNet = nn.Sequential(*UpNet)

        self.finalOut = nn.Sequential(

            nn.Conv2d(in_channels=Res_channelList[-1],out_channels=1,kernel_size=1),
            nn.PReLU() # 可以测试下不加激活函数的效果
        )

    def forward(self,x0):
        
        # x0:[batchsize,num_features]
        batchsize = x0.shape[0]
        x = self.linear(x0)
        size = x.shape[-1] ** 0.5
        x = x.reshape(batchsize,1,size,size)
        res1 = self.inRes(x)
        res2 = self.ResNet(res1)
        res3 = self.outRes(res2)
        res4 = self.UpNet(res3 + res1)
        res = self.finalOut(res4).reshape(batchsize,-1)
        
        return res

class MeshRefineNet_D(nn.Module):
    """
        判别器判断对平滑前后的网格进行鉴别:更像是在对拉普拉斯网格平滑的逆向操作
        input: mesh[batchsize,numverts,3]
    """
    def __init__(self,numverts,channelList=[3,5,7,9,1024],reluRate=0.2,device="cuda:0"):
    
        super(MeshRefineNet_D,self).__init__()
        
        self.device = device
        self.reluRate = reluRate
        self.channelList = channelList
        self.numverts = numverts
        self.sqrtdim = int(torch.sqrt(numverts)) + 1

        self.linear = nn.Linear(numverts,self.sqrtdim**2)

        self.net = nn.Sequential(

            nn.Conv2d(channelList[0], channelList[1], kernel_size=3, padding=1),
            nn.LeakyReLU(reluRate),  

            nn.Conv2d(channelList[1], channelList[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channelList[1]),
            nn.LeakyReLU(reluRate),

            nn.Conv2d(channelList[1], channelList[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channelList[2]),
            nn.LeakyReLU(reluRate),

            nn.Conv2d(channelList[2], channelList[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channelList[2]),
            nn.LeakyReLU(reluRate),

            nn.Conv2d(channelList[2], channelList[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channelList[3]),
            nn.LeakyReLU(reluRate),

            nn.Conv2d(channelList[3], channelList[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channelList[3]),
            nn.LeakyReLU(reluRate),

            # 自适应平均池化层，将输入特征图转换为大小为1x1的特征图
            nn.AdaptiveAvgPool2d(16),
            nn.Conv2d(channelList[3], channelList[4], kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.LeakyReLU(reluRate),
            nn.Conv2d(channelList[4], 1, kernel_size=1)
        )

    def forward(self, x0):
        # 输入批次的大小
        batchsize = x0.shape[0]
        x = self.linear(x0).reshape(batchsize,3,self.sqrtdim,self.sqrtdim)
        x = self.net(x).view(batchsize)

        # 使用torch.sigmoid函数将特征图映射到0到1之间，表示输入图像为真实图像的概率。
        return torch.sigmoid(x)
      
class GarmentNet_HF(nn.Module):
    
    """
        基于卷积网络CNN结构重构高精度的布料网格: 输入网格经过两个卷积块提取特征,得到[4096,1]的特征向量
    """
#region 初始卷积层定义: 计算很慢,采取小步长和大卷积核导致参数规模很大,弃用
#     def __init__(self,numverts=283344,num_features=4096,channels=[1,3,1,4],device="cuda:0"):
        
#         super(GarmentNet_HF,self).__init__()
        
#         self.blockNum = 2
#         self.device = device
#         self.channels = channels
#         self.numverts = numverts
#         self.num_features = num_features
        
#         #  编码器设定: 两个卷积块+激活层,每个卷积块包含卷积层、BN、pool层,其中针对网格数据采用了长宽不一的矩形conv核
#         h_kernel = [1000,623]
#         w_kernel = 3
#         stride_list = [16,8]
        
#         self.convBlock1 = nn.Sequential(
#             nn.Conv2d(in_channels=self.channels[0],out_channels=self.channels[1],kernel_size=(h_kernel[0],w_kernel),stride=(stride_list[0],1)),
#             nn.BatchNorm2d(self.channels[1]),
#             nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))
#         )
        
#         self.convBlock2 = nn.Sequential(
#             nn.Conv2d(in_channels=self.channels[2],out_channels=self.channels[3],kernel_size=(h_kernel[1],w_kernel),stride=(stride_list[1],1)),
#             nn.BatchNorm2d(self.channels[3]),
#             nn.MaxPool2d(kernel_size=(3,1),stride=(1,1))
#         )
        
#         self.relu = nn.PReLU()

#         #解码器设定: 两种思路 -- 1) 基于多个线性层; 2) 基于反卷积层 -- 观察哪个更加有效？
# #         self.decodeLinear = nn.Sequential(
# #             nn.Linear(num_features,num_features*4),
# #             nn.PReLU(),
# #             nn.Linear(num_features*4,numverts),  # 这里内存直接爆炸，
# #         )
        
#         channel_list = [1,1]
#         scale = int(numverts/num_features)+1  # 扩大规模
#         self.decodeConv = nn.Sequential(
#             nn.Conv2d(in_channels=1,out_channels=scale**2,kernel_size=1),
#             nn.BatchNorm2d(scale**2),
#             nn.PixelShuffle(scale),
#             nn.PReLU(),
#             #nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(3377,68),stride=1,padding=(0,0)),
#             nn.Conv2d(in_channels=1,out_channels=channel_list[1],kernel_size=(601,11)),
#             nn.BatchNorm2d(channel_list[1]),
#             nn.PReLU(),
#             nn.Conv2d(in_channels=channel_list[1],out_channels=channel_list[1],kernel_size=(601,11)),
#             nn.BatchNorm2d(channel_list[1]),
#             nn.PReLU(),
#             nn.Conv2d(in_channels=channel_list[1],out_channels=channel_list[1],kernel_size=(601,11)),
#             nn.BatchNorm2d(channel_list[1]),
#             nn.PReLU(),
#             nn.Conv2d(in_channels=channel_list[1],out_channels=channel_list[1],kernel_size=(601,11)),
#             nn.BatchNorm2d(channel_list[1]),
#             nn.PReLU(),
#             nn.Conv2d(in_channels=channel_list[1],out_channels=channel_list[1],kernel_size=(601,11)),
#             nn.BatchNorm2d(channel_list[1]),
#             nn.PReLU(),
#             nn.Conv2d(in_channels=channel_list[1],out_channels=1,kernel_size=(377,18)),
#         )
#endregion

    def encoder(self,x0):
        x0 = x0.unsqueeze(1).float()
        batchsize = x0.shape[0]
        x = self.convBlock1(x0).view(batchsize,1,-1,3)
        x = self.convBlock2(x)
        x = self.relu(x.view(batchsize,-1))
        
        return x

    def decoder(self,z):
        # z:[batchsize,numfeartures]
        batchsize = z.shape[0]
        # res = self.decodeLinear(z)
        res = self.decodeConv(z.view(batchsize,1,-1,1)).view(batchsize,self.numverts,3)
        
        return res
    
class GarmentNet_ShapeDescriptor(nn.Module):
    
    """
        Ref: 米哈游paper, 基于 "MLP_autoencoder + shape_descriptor" 的网络结构预测布料随动作形变的粗糙结果
        米哈游的网络分为三个部分:
        1)布料网格的描述符重构网络,提取形状不变描述符; 
    """
    def __init__(self,numverts=283344,dim_list=[4096,1024,512,64],device="cuda:0"):
        
        super(GarmentNet_ShapeDescriptor,self).__init__()
        self.device = device
        self.numverts = numverts
        dim_list.insert(0,numverts)
        self.dim_list = dim_list # [4096,1024,512,64]
        self.numverts = numverts
        depth = len(self.dim_list)-1

        self.enNet = nn.Sequential(
            nn.Linear(dim_list[0],dim_list[1]),
            nn.BatchNorm1d(),
            nn.PReLU(),
            nn.Linear(dim_list[1],dim_list[2]),
            nn.BatchNorm1d(),
            nn.PReLU(),
            nn.Linear(dim_list[2],dim_list[3]),
            nn.BatchNorm1d(),
            nn.PReLU(),
            nn.Linear(dim_list[3],dim_list[4]),            
            nn.BatchNorm1d(),
            nn.PReLU(),
        )

        self.deNet = nn.Sequential(
            nn.Linear(dim_list[4],dim_list[3]),
            nn.BatchNorm1d(),
            nn.PReLU(),
            nn.Linear(dim_list[3],dim_list[2]),
            nn.BatchNorm1d(),
            nn.PReLU(),
            nn.Linear(dim_list[2],dim_list[1]),
            nn.BatchNorm1d(),
            nn.PReLU(),
            nn.Linear(dim_list[0],dim_list[0]),            
            nn.PReLU(),
            
        )

    def encoder(self,x0) -> torch.FloatTensor:

        # input:[batchsize,numvert3*3]
        batchsize = x0.shape[0]
        x = x0.view(batchsize,-1)
        z = self.enNet(x)

        return z
    
    def decoder(self,z0) -> torch.FloatTensor:
        batchsize = z0.shape[0]
        z = z0.view(batchsize,-1)
        y = self.deNet(z)
        return y
        
class GarmentNet_LF(nn.Module):
    
    """
        Ref: 米哈游paper, 基于 "MLP_autoencoder + shape_descriptor" 的网络结构预测布料随动作形变的粗糙结果
        2) 提取动作描述符的网络; 
        3) 基于动作和形状不变描述符的提取特征的网络,作用是实现以动作为条件,实现形状不变描述符的重构任务
        input: motion  ; shape_descriptor
    """
    def __init__(self,numverts=283344,motion_features=128,dim_list=[64,256,256],m_dimList=[256,128,64],device="cuda:0"):

        
        super(GarmentNet_LF,self).__init__()

        m_dimList.insert(0,motion_features) # motion_features是输入的动作维度,根据具体数据确定
        self.device = device
        self.numverts = numverts

        self.dim_list = dim_list # [64,256,256]
        self.m_dimList = m_dimList

        self.m_depth = len(self.m_dimList)-1
        self.depth = len(self.dim_list)-1

        self.MNet = nn.Sequential(
            nn.Linear(m_dimList[0],m_dimList[1]),
            nn.BatchNorm1d(m_dimList[1]),
            nn.PReLU(),
            nn.Linear(m_dimList[1],m_dimList[2]),
            nn.BatchNorm1d(m_dimList[2]),
            nn.PReLU(),
            nn.Linear(m_dimList[2],m_dimList[3]),
            nn.BatchNorm1d(m_dimList[3]),
            nn.PReLU(),
        )

        self.dMNet = nn.Sequential(

            nn.Linear(m_dimList[3],m_dimList[2]),
            nn.BatchNorm1d(m_dimList[2]),
            nn.PReLU(),
            nn.Linear(m_dimList[2],m_dimList[1]),
            nn.BatchNorm1d(m_dimList[1]),
            nn.PReLU(),
            nn.Linear(m_dimList[0],m_dimList[0]), 
            nn.BatchNorm1d(m_dimList[0]),
            nn.PReLU(),
            
        )

        self.PF_depth = m_dimList[-1]
        self.Net = nn.ModuleList()
        for idx in range(self.depth):
            tmp = nn.ModuleList()
            for i in range(self.PF_depth):
                tmp.append(nn.Linear(dim_list[idx],dim_list[idx+1]))
            self.Net.append(tmp)

        self.dNet = nn.ModuleList()
        for idx in range(self.depth):
            tmp = nn.ModuleList()
            for i in range(self.PF_depth):
                tmp.append(nn.Linear(dim_list[self.depth-idx],dim_list[self.depth-idx-1]))
            self.dNet.append(tmp)

        self.bn = nn.BatchNorm1d(dim_list[1])
        self.acfunc = nn.PReLU()

    def m_encoder(self,m0) -> torch.FloatTensor:

        batchsize = m0.shape[0]
        m = m0.view(batchsize,-1)
        mz = self.MNet(m)

        return mz
    
    def m_decoder(self,mz0) -> torch.FloatTensor:

        batchsize = mz0.shape[0]
        mz = mz0.view(batchsize,-1)
        my = self.dMNet(mz)

        return my

    def encoder(self,x0,weight) -> torch.FloatTensor:

        weight = weight.unsqueeze(-1) # 不加这行后续执行乘法会报错
        batchsize = x0.shape[0]
        w_sum = torch.sum(weight)
        z = x0.view(batchsize,-1)
        for idx in range(self.depth):
            sums = 0
            for j in range(self.PF_depth):
                wx = weight[:,j] / w_sum * self.Net[idx][j](z) 
                sums = sums + wx
            z = self.bn(sums)
            z = self.acfunc(z)

        return z
    
    def decoder(self,z0,weight) -> torch.FloatTensor:
        
        weight = weight.unsqueeze(-1)
        batchsize = z0.shape[0]
        w_sum = torch.sum(weight)
        y = z0.view(batchsize,-1)
        for idx in range(self.depth):
            sums = 0
            for j in range(self.PF_depth):
                wx = weight[:,j] / w_sum * self.dNet[idx][j](y)
                sums = sums + wx
            if idx < self.depth-1:  # 输出部分不用激活和BN操作
                y = self.bn(sums)
                y = self.acfunc(y)
            else:
                y = sums

        return y

class BodyNet_HF(nn.Module):
    
    """
        基于卷积网络CNN结构重构高精度的人体网格: 输入网格经过两个卷积块提取特征,得到[128,1]的特征向量
    """

    def __init__(self,numverts=106422,num_features=128,channels=[1,3,1,1],device="cuda:0"):
        
        super(BodyNet_HF,self).__init__()
        
        self.blockNum = 2
        self.device = device
        self.channels = channels
        self.numverts = numverts
        self.num_features = num_features
        
        #  编码器设定: 两个卷积块+激活层,每个卷积块包含卷积层、BN、pool层,其中针对网格数据采用了长宽不一的矩形conv核
        h_kernel = [122,48]
        w_kernel = 3
        stride_list = [100,8]
        
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels[0],out_channels=self.channels[1],kernel_size=(h_kernel[0],w_kernel),stride=(stride_list[0],1)),
            nn.BatchNorm2d(self.channels[1]),
            # nn.MaxPool2d(kernel_size=(3,1),stride=(1,1),padding=(1,0))
        )
        
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels[2],out_channels=self.channels[3],kernel_size=(h_kernel[1],w_kernel),stride=(stride_list[1],1)),
            nn.BatchNorm2d(self.channels[3]),
            # 添加池化层导致提取的特征向量相邻维度的值重复(图像领域池化有效,但网格任务可以考虑弃用池化层)
            #nn.MaxPool2d(kernel_size=(3,1),stride=(1,1),padding=(1,0)) # 
        )
        
        self.relu = nn.PReLU()

        #解码器设定: 两种思路 -- 1) 基于多个线性层; 2) 基于反卷积层 -- 观察哪个更加有效？
#         self.decodeLinear = nn.Sequential(
#             nn.Linear(num_features,num_features*4),
#             nn.PReLU(),
#             nn.Linear(num_features*4,numverts),  # 这里内存直接爆炸，
#         )
        
        channel_list = [1,1]
        # scale = int(numverts/num_features)+1  # 扩大规模
        scale = 29  # 参数规模太大,考虑分两次上采样 -- 对比一次放大速度快了很多
        h_k = ((num_features*scale)-scale+1)*scale - numverts + 1
        self.decodeConv = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=scale**2,kernel_size=1),
            nn.BatchNorm2d(scale**2),
            nn.PixelShuffle(scale),
            nn.PReLU(),
            #nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(3377,68),stride=1,padding=(0,0)),
            nn.Conv2d(in_channels=1,out_channels=channel_list[1],kernel_size=scale),
            nn.BatchNorm2d(channel_list[1]),
            nn.PReLU(),

            nn.Conv2d(in_channels=1,out_channels=scale**2,kernel_size=1),
            nn.BatchNorm2d(scale**2),
            nn.PixelShuffle(scale),
            nn.PReLU(),

            nn.Conv2d(in_channels=channel_list[1],out_channels=channel_list[1],kernel_size=(h_k,scale-3+1)),
            nn.BatchNorm2d(channel_list[1]),
            nn.PReLU(),
        )


    def encoder(self,x0):
        x0 = x0.unsqueeze(1).float()
        batchsize = x0.shape[0]
        x = self.convBlock1(x0).view(batchsize,1,-1,3)
        x = self.convBlock2(x)
        x = self.relu(x.view(batchsize,-1))
        #print(x.shape)
        return x

    def decoder(self,z):
        # z:[batchsize,numfeartures]
        batchsize = z.shape[0]
        # res = self.decodeLinear(z)
        res = self.decodeConv(z.view(batchsize,1,-1,1)).view(batchsize,self.numverts,3)
        #print(res.shape)
        return res
    
class Penetration_D(nn.Module):

    """
        给定人体网格和任意顶点,检测顶点是否在人体网格外部(未穿模):在外则判定为1,在内判定为0
        输入: 当前人体网格,任意布料顶点
        输出: 任意布料顶点是否在人体网格外部的结果向量,长度=输入的顶点规模
        核心思路: 基于人体动作训练一个广义的超平面笼罩在人体,用于分离人体和布料
    """    
    def __init__(self,cloth_numverts=26718,cloth_features=3,body_numverts=20813,linear_dim=[32,128,1],motion_size=48,device="cuda:0"):
        
        super(Penetration_D, self).__init__()
        # 1.body_mesh 使用 transformer + conv 网络结构
        self.device = device
        self.motion_weights = GMW_Net(final_features=motion_size)
        self.motion_size = motion_size
        linear_dim.insert(0,cloth_features) # 布料网格的特征使用三维坐标
        # print(linear_dim)
        self.fusion_depth = len(linear_dim)-2
        # 2.Fusion_net for mixing the motion_weights to discriminator
        self.fusion = nn.ModuleList()
        self.fus_bn = nn.ModuleList()
        for idx in range(self.fusion_depth):
            tmp = nn.ModuleList()
            for i in range(motion_size):
                tmp.append(nn.Linear(linear_dim[idx],linear_dim[idx+1]))
            self.fusion.append(tmp)
            self.fus_bn.append(nn.BatchNorm1d(linear_dim[idx+1]))    

        self.fus_ac = nn.PReLU()

        # 3.布料网格使用线性层提取特征,因为是逐顶点判断,而不是对整个布料网格 -- 此处输入batchsize默认为1(不考虑拆分网格顶点),否则数据规模太大！
        self.cloth_numverts = cloth_numverts
        self.cloth_features = cloth_features


        # 4.最后使用线性层将动作权重与布料特征混合,并输出预测概率值
        self.out_linear = nn.Sequential(
            nn.Linear(linear_dim[-2],linear_dim[-1]),
            nn.Sigmoid()
        )

    def forward(self,vertex_pos,body_mesh):

        batchsize = vertex_pos.shape[0] 
        # 对于此判定网络:布料网格的batch_size只能是1,动作可以有多个输入,输出结果[motion_num,cloth_numverts,1]
        # 但考虑到内存限制,动作数也限制为1
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
    
    # def forward(self,cloth_mesh,body_mesh):

    #     batchsize = cloth_mesh.shape[0] 
    #     # 对于此判定网络:布料网格的batch_size只能是1,动作可以有多个输入,输出结果[motion_num,cloth_numverts,1]
    #     # 但考虑到内存限制,动作数也限制为1
    #     body_mesh = body_mesh.view(batchsize,-1,3).float()
    #     motion_weights = self.get_motion_weights(body_mesh).float()
    #     w_sum = torch.sum(motion_weights)
    #     x = cloth_mesh.view(self.cloth_numverts,3).float()

    #     for idx in range(self.fusion_depth):
    #         sums = 0
    #         for j in range(self.motion_size):
    #             wx = motion_weights[:,j] / w_sum * self.fusion[idx][j](x) 
    #             sums = sums + wx
    #         x = self.fus_bn[idx](sums)
    #         x = self.fus_ac(x)
    #         # print(x.shape)
        
    #     p = self.out_linear(x).view(batchsize,-1) # [cloth_numverts,1]>>[1,cloth_numverts]

    #     return p
    
    def get_motion_weights(self,body_mesh):
        
        motion_weights = self.motion_weights(body_mesh)
        # print(motion_weights.shape)

        return motion_weights

class MultiHeadAttention_conv(nn.Module):
    """
        网格检测中的多头注意力层: 采取类似NLP中的多头注意力层
        input: body_mesh:[numverts,3]
        out: features[out_features,1]
        ------
        params:
        
        numverts:顶点个数 --> 原始网格顶点规模太大导致训练难以开展,考虑使用卷积网络提取网格特征得到虚拟顶点作为此处的输入
        numfeatures: 卷积后提取的特征向量的第一维度,可更改,预设为1024
        input_size: 输入多头网络的特征的属性维度 
                    -- 原计划将顶点数据直接输入多头网络[numverts,3],input_size=3三维顶点坐标;
                    -- 但受顶点规模影响,只能考虑先用一个卷积网络提取原始网格特征,并将特征向量作为多头网络输入,结果[num_features,9]
                    -- [numfeatures=1024,9] 是自己指定的,可更改
        channels: 指定两个卷积层的通道值
                    
        numheads:多头的header个数
        hidden_size:多头注意力的总的隐藏层特征个数 = heads_num * head_size(多头的个数*每个头对应的隐藏特征个数)
        out_features: 输出向量的维度
        ------        
        网络的输出:
        conv的输出结果:       
                    -- [batchsize,3,numfeatures*(input_size/3),1] 对应 [batchsize,1,numverts,3]经过两个卷积网络后输出的维度
                    -- 最终输出reshape为[batchsize,numfeatures,input_size]作为多头层的输入
        multihead的输出:
                    -- 输入[batchsize,numfeatures,input_size],将input_size使用多头机制作用+全连接层
                    -- 输出[batchsize,out_features=1024]
                    
    """
    def __init__(self,numverts=20813,num_features=1024,input_size=9,channels=[1,3,1,3],num_heads=2,hidden_size=4,out_features=1024,device="cuda:0"):
        
        super(MultiHeadAttention_conv, self).__init__()

        # 使用一个卷积网络先对原始网格进行降维[numverts,3] >> [num_features,3]
        self.blockNum = 2
        self.device = device
        self.num_features = num_features
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        #self.channels = channels
        #self.numverts = numverts
        
        #  编码器设定: 两个卷积块+激活层,每个卷积块包含卷积层、BN、pool层,其中针对网格数据采用了长宽不一的矩形conv核
        # [20809~20813,3] >conv> [3072,3] >reshape> [1024,9]  
        # 发现一个有趣的事情: 卷积神经网络支持输入长度可变(在一定范围内),因为卷积运算要除以stride,这一步会取整,输入size的微小差异在大的stride下会被忽略！
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
            # 添加池化层导致提取的特征向量相邻维度的值重复(图像领域池化有效,但网格任务可以考虑弃用池化层)
            #nn.MaxPool2d(kernel_size=(3,1),stride=(1,1),padding=(1,0)) # 
        )
        self.relu = nn.PReLU()
        
        # 初始顶点规模很大 -- 这个线性层很难实现 -- 先运行下试试,不行的话只能使用卷积网络降维
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        
        self.fc_out = nn.Linear(num_features*hidden_size, out_features) #压缩成向量然后提取特征
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_size)
        # x.shape:[batchsize,num_heads,numverts,head_size]
        return x.permute(0, 2, 1, 3)
    
    def mesh_encoder(self,mesh):
        
        mesh = mesh.unsqueeze(1).float()
        batchsize = mesh.shape[0]
        x = self.convBlock1(mesh).view(batchsize,1,-1,3)
        x = self.convBlock2(x)
        x = self.relu(x.view(batchsize,self.num_features,-1))
        # print(x.shape) #[batchsize,1024,9]
        
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
        attended_values = torch.matmul(attention_weights, value)
        
        #contiguous()实现张量语义和内存顺序的一致性维护
        # [batchsize,num_features,numheads,head_size]
        attended_values = attended_values.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.head_size)
        output = self.fc_out(attended_values.view(batch_size,-1))  # [batchsize,1024]
        
        return output   

class GMW_Net(nn.Module):
    """
        GMW:get_motion_weights Net,使用一个conv+多头注意力网络提取人体网格特征,得到一组权重向量
    """
    def __init__(self,numverts=20813,num_features=1024,input_size=9,channels=[1,3,1,3],num_heads=2,hidden_size=4,head_out_features=1024,final_features=32,dropout=0.2,device="cuda:0"):
        super(GMW_Net, self).__init__()

        # params:
        self.device = device
        
        self.multihead_attention = MultiHeadAttention_conv(numverts=numverts,num_features=num_features,input_size=input_size,channels=channels,
                                                           num_heads=num_heads,hidden_size=hidden_size,out_features=head_out_features,device=device)
        self.feed_forward = nn.ModuleList()
        self.feed_forward.append(
            nn.Sequential(
            nn.Linear(head_out_features, head_out_features),
            nn.LayerNorm(head_out_features),
            nn.ELU(),)
        )
        self.feed_forward.append(nn.Linear(head_out_features,final_features))

        #self.norm1 = nn.LayerNorm(head_out_features)
        #self.dropout = nn.Dropout(dropout)
        
    def forward(self,mesh,mask=None):

        x = self.multihead_attention(mesh)  #[batchsize,out_features]
        #x = self.dropout(x)
        #x = self.norm1(x)
        feed_forward_output = self.feed_forward[0](x)
        #x = x + self.dropout(feed_forward_output)
        x = x + feed_forward_output
        x = self.feed_forward[1](x)  # [batchsize,final_features]
        
        return x


def Loss_func(pos,pred_pos,mode="L2",bool_col = 0):
    """
    # https://blog.csdn.net/WANGWUSHAN/article/details/105903765
    定义反向传播的loss函数:
    ## 只接受tensor张量输入
    1. 均方损失
    2. 绝对值损失
    3. 混合损失
    4. BCEloss: 需要将数据处理到[0,1]之间 -- sigmoid
    
    """
    if mode == "L2":
        # RMSE
        loss_fn = torch.nn.MSELoss(reduction='mean')
        return torch.sqrt(loss_fn(pos,pred_pos))

    elif mode == "L1":
        loss_fn = torch.nn.L1Loss(reduction='mean')
        return loss_fn(pos,pred_pos)
      
    elif mode == "Mix":
        loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
        return loss_fn(pos,pred_pos)
    
    elif mode == "BCE":
        sigmoid = torch.nn.Sigmoid()
        loss_fn = torch.nn.BCELoss()
        sigmoid_pos=sigmoid(pos)
        sigmoid_pos_pred=sigmoid(pred_pos)
        return loss_fn(sigmoid_pos,sigmoid_pos_pred)
    
    elif mode == "BCE_L":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        return loss_fn(pos,pred_pos)  

    
    elif mode == "CrossE":
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(pos,pred_pos)
    
    elif mode == "NL":
        # 这个loss直接算回归问题有点勉强 -- 后续看怎么处理
        log_ = torch.nn.LogSoftmax(dim=1)
        loss_fn = torch.nn.NLLLoss()
        log_pos = log_(pos)
        return loss_fn(log_pos,pred_pos)
    
    elif mode == "KL":
        logp_y = F.log_softmax(pred_pos, dim = -1)
        p_x = F.softmax(pos, dim =- 1)

        dist = F.kl_div(logp_y, p_x, reduction='batchmean')
        return dist

def MeshLoss():

    # 定义网格相似性作为预测网格和真实网格之间的误差值 --
    pass


if __name__ == "__main__":
    # ctrl + / 实现批量注释
    device = "cuda:0"
    #cloth = Mesh_obj("garment.obj")

    # 穿模判别器网络测试
    net = Penetration_D()
    a = torch.randn(3,3)
    b = torch.randn(3,20813,3)
    c = net(a,b)
    # print(c.shape) #[batchsize,1]