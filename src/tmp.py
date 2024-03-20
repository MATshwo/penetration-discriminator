
def get_full_edge_index(numverts=1,mode="empty"):
    # 根据顶点数量创建全孤立&全连通的邻接矩阵  -- 返回稀疏格式[2,edge_num]
    from torch_geometric.utils import dense_to_sparse    
    if mode == "empty":
        adj = torch.zeros((numverts,numverts))  # 使用全连接导致最终输出无法区分不同维度的差别，所以由全连通改为全不连通
    elif mode == "full":
        # 全连通邻接矩阵
        adj = torch.ones((numverts,numverts)) - torch.eye(numverts)
    return dense_to_sparse(adj)[0]

def MLP(dimensions, dropout=True, batch_norm=True, activate=True,batch_norm_momentum=1e-3):
    # 默认只执行drop & linear操作 -- 激活和归一化需要额外指定
    return nn.Sequential(*[
        nn.Sequential(
            nn.Dropout(p=0.2) if dropout else nn.Identity(),
            nn.Linear(dimensions[i - 1], dimensions[i]),
            nn.PReLU(dimensions[i]) if activate else nn.Identity(),
            nn.BatchNorm1d(dimensions[i], affine=True, momentum=batch_norm_momentum) if batch_norm else nn.Identity())
        for i in range(1, len(dimensions))])

class Motion_net(nn.Module):
    def __init__(self,motion_input_dim=63,
                 gru_dim=32,device="cuda:0",point_dim = [12273,3067,2298]):

        super(Motion_net, self).__init__()
        self.device = device
        self.motion_input_dim = motion_input_dim
        #self.motion_input_dim = 52*3+3
        self.mlayer_dim = 256
        self.point_num = point_dim
        self.gru_dim = gru_dim  # 并非越大越好 -- 还占内存 linear也是,多了反而导致模型loss疯狂上升 -- 奇怪 32目前最好；16不行

        self.motion_net = nn.ModuleList()

        self.M_layer = Motion_layer(self.gru_dim,self.mlayer_dim,device=device)

        self.motion_net.append(nn.GRUCell(self.motion_input_dim,self.gru_dim))
        self.motion_net.append(nn.Sequential(MLP([self.gru_dim,self.point_num[-1]],BN_dim=self.point_num[-1],batch_norm=True,dropout=False,activate=False))) # 乘数作用使用归一化
        self.motion_net.append(nn.Sequential(MLP([self.gru_dim,self.point_num[-1]],BN_dim=self.point_num[-1],batch_norm=False,dropout=False,activate=True))) # 常数项不使用归一化--都使用了激活层

    def motion_weight(self,motion_data,hidden=None):
        # 输入动作信息[batch=500,63],学习decode解码对应的权重信息 -- [batch,9,9] -- [batch,gcn_dim[-1],gcn_dim[-1]] 
        batchsize = motion_data.shape[0]
        motion_data = motion_data.float().to(self.device)
        if hidden is None:
            hidden = torch.zeros(batchsize, self.gru_dim).to(self.motion_net[0].bias_hh.device)
        else:
            hidden = hidden.to(self.device)
        next_hidden = self.motion_net[0](motion_data,hidden).to(self.device)  
        gru_out1 = torch.sigmoid(self.motion_net[1](next_hidden).view((batchsize,self.point_num[-1])))
        gru_out2 = self.motion_net[2](next_hidden).view((batchsize,self.point_num[-1]))
        return [gru_out1,gru_out2],next_hidden

        # 只用线性层效果并不好 -- 但也可以下降到0.03左右
        # gru_out1 = self.motion_net[1](motion_data).view((batchsize,self.point_num[-1]))
        # gru_out2 = self.motion_net[2](motion_data).view((batchsize,self.point_num[-1]))
        # return [gru_out1,gru_out2],None

    def motion_weight1(self,motion_data,hidden=None):

        # 想尝试一个新的核函数方法：结果效果很差 -- 
        batchsize = motion_data.shape[0]
        motion_data = motion_data.float().to(self.device)
        m_layer_linear = nn.Linear(self.mlayer_dim,self.point_num[-1]).to(self.device)
        hidden_linear = nn.Linear(self.gru_dim,self.gru_dim).to(self.device)
        if hidden is None:
            hidden = torch.zeros(batchsize, self.gru_dim).to(self.motion_net[0].bias_hh.device)
        else:
            hidden = hidden.to(self.device)
        next_hidden = self.motion_net[0](motion_data,hidden).to(self.device)
        next_input = self.M_layer.forward(hidden_linear(next_hidden)).to(self.device)  # [64] -- [64,512]
        #bias = torch.nn.Parameter(torch.zeros(self.point_num[-1]),requires_grad=True).to(self.device)
        next_out = m_layer_linear(torch.einsum("ij,ijk->ik",next_hidden,next_input)).view((batchsize,self.point_num[-1])) #[batc,64]*[64,512]*[512,2298] == [batch,2298]
        #temp = torch.ones_like(next_out,requires_grad=False).to(self.device)

        #return [temp,next_out],next_hidden
        return [next_out,0.0],next_hidden

class Cloth_GAN_0(nn.Module):
    
    # 1. input:[12273,3] --> input*cosine:[12273,3] --> reshape: [3,12273]
    # 2. GNN_encoder: [3,12273],全连通邻接矩阵 --> [3,9]
    # 3. GNN_decoder: [3,9],全连通邻接矩阵 --> [3,12273]

    def __init__(self,motion_input_dim=63,
                 gru_dim=128,device="cuda:0",point_dim = [12273,3067,2298],channels=[1,3],batch_norm=True,p=0):

        super(Cloth_GAN_0, self).__init__()
        self.device = device
        self.motion_input_dim = motion_input_dim
        #self.point_num = [12273,3066,2292]
        self.point_num = point_dim
        self.gru_dim = gru_dim  # 并非越大越好 -- 还占内存 linear也是，不是越多越好
        self.channels = channels

        self.encode_layers = nn.ModuleList()
        self.decode_layers = nn.ModuleList()

        self.motion_net = nn.ModuleList()
        
        #self.M_layer = Motion_layer(self.gru_dim,self.point_num[-1])

        
        net_deepth = len(self.point_num) - 1
        self.pooling = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))
        for i in range(net_deepth):
            # 可以考虑增大下卷积核的规模 
            self.encode_layers.append(nn.Conv2d(in_channels=self.channels[0],out_channels=self.channels[1],kernel_size=(3,3),stride=2)) #0526
            #self.encode_layers.append(nn.Conv2d(in_channels=self.channels[0],out_channels=self.channels[1],kernel_size=(8,3),stride=2)) #0529

        self.final_linear = nn.Sequential(
                nn.Linear(self.point_num[-1],12273*3),
                #nn.LeakyReLU(),
                #nn.Linear(1024,12273*3), # 增加了线性层网络性能下降？ -- 好奇怪，好像我越做加法，模型效果越差
                )
        self.decode_layers.append(nn.ConvTranspose2d(in_channels=self.channels[0],out_channels=self.channels[0],kernel_size=(1542,3),stride=1))
        self.decode_layers.append(nn.ConvTranspose2d(in_channels=self.channels[0],out_channels=self.channels[0],kernel_size=(7,3),stride=1))
        self.motion_net.append(nn.GRUCell(self.motion_input_dim,self.gru_dim))
        self.motion_net.append(nn.Sequential(MLP([self.gru_dim,self.point_num[1]*self.channels[1]],BN_dim=self.point_num[1]*self.channels[1],batch_norm=batch_norm,dropout=True,activate=True)))
        self.motion_net.append(nn.Sequential(MLP([self.gru_dim,self.point_num[-1]*self.channels[0]],BN_dim=self.point_num[-1]*self.channels[0],batch_norm=batch_norm,dropout=True,activate=True)))

    def gan_encoder(self,pos,motion_weights=None,is_motion=False):

    
        batch_size = pos.shape[0]
        pos = pos.unsqueeze(1).float().to(self.device)  # [batch,1,n,3]

        if is_motion:
            tmp = self.encode_layers[0](pos).view(batch_size,1,-1,3)  #[]
            tmp = self.pooling(tmp)
            tmp = torch.mul(tmp,motion_weights[0])
            tmp = self.encode_layers[1](tmp)
            #x_reduce = self.pooling(tmp).view(batch_size,-1,1) # 不使用激活函数貌似更好一点
            tmp = torch.mul(self.pooling(tmp).view(batch_size,-1,1),motion_weights[1])
            x_reduce = F.leaky_relu(tmp)
            
        else:
            tmp = self.encode_layers[0](pos).view(batch_size,1,-1,3) 
            tmp = self.pooling(tmp) #[b,3066,3]
            tmp = self.encode_layers[1](tmp)
            #x_reduce = self.pooling(tmp).view(batch_size,-1,1) # 不使用激活函数会差一点
            x_reduce = F.leaky_relu(self.pooling(tmp).view(batch_size,-1,1)) #[6,2292,1] -- [b,2292,1]

        return x_reduce
    
    def gan_decoder(self,reduce_pos,motion_weights=None,is_motion=False):
        
        batch_size = reduce_pos.shape[0]
        reduce_pos = reduce_pos.unsqueeze(1).float() #[6,1,2298,1]
        
        # recover还原阶段目前只有一个线性层
        if is_motion:
           
            tmp = torch.mul(reduce_pos.view(batch_size,-1),motion_weights[0]) + motion_weights[1] #用动作加权后的结果
            
        else:
            #tmp = self.decode_layers[0](reduce_pos)
            #tmp = self.decode_layers[1](tmp).view(batch_size,-1)
            # decode部分的设计并不复杂 -- 如何能有效降低loss -- 只用目前的结果loss最低0.017左右
            tmp = reduce_pos.view(batch_size,-1)

        x_recover = self.final_linear(tmp).view(batch_size,-1,3)
        #x_recover = tmp.view(batch_size,-1,3)  + self.final_linear(tmp).view(batch_size,-1,3) 
        return tmp.view(batch_size,-1,1),x_recover
    
    def motion_weight(self,motion_data,hidden=None):
        # 输入动作信息[batch=500,63],学习decode解码对应的权重信息 -- [batch,9,9] -- [batch,gcn_dim[-1],gcn_dim[-1]] 
        batchsize = motion_data.shape[0]
        motion_data = motion_data.float().to(self.device)
        if hidden is None:
            hidden = torch.zeros(batchsize, self.gru_dim).to(self.motion_net[0].bias_hh.device)
        else:
            hidden = hidden.to(self.device)
        next_hidden = self.motion_net[0](motion_data,hidden).to(self.device)

        gru_out1 = self.motion_net[1](next_hidden).view((batchsize,1,self.point_num[1],self.channels[1])) #作为encoder的权重输入
        gru_out2 = self.motion_net[2](next_hidden).view((batchsize,self.point_num[-1],self.channels[0])) #作为encoder的权重输入
        return [gru_out1,gru_out2],next_hidden

class Basic_vae(nn.Module):
    
    # 1. input:[12273,3] --> input*cosine:[12273,3] --> reshape: [3,12273]
    # 2. encoder: [3,12273],全连通邻接矩阵 --> [3,9]
    # 3. decoder: [3,9],全连通邻接矩阵 --> [3,12273]

    def __init__(self,motion_input_dim=63,
                 gru_dim=128,device="cuda:0",point_dim = [12273,3067,2298],channels=[1,3],batch_norm=True,p=0):

        super(Basic_vae, self).__init__()
        self.device = device
        self.motion_input_dim = motion_input_dim
        self.point_num = point_dim
        self.gru_dim = gru_dim  # 并非越大越好 -- 还占内存 linear也是，不是越多越好
        self.channels = channels

        self.encode_mean = nn.ModuleList()
        self.encode_var = nn.ModuleList()
        self.decode_layers = nn.ModuleList()

        self.motion_net = nn.ModuleList()
        
        net_deepth = len(self.point_num) - 1
        self.pooling = nn.MaxPool2d(kernel_size=(3,1),stride=(2,1))
        for i in range(net_deepth):
            self.encode_mean.append(nn.Conv2d(in_channels=self.channels[0],out_channels=self.channels[1],kernel_size=(3,3),stride=2))
            self.encode_var.append(nn.Conv2d(in_channels=self.channels[0],out_channels=self.channels[1],kernel_size=(3,3),stride=2))

        #self.decode_layers.append(nn.ConvTranspose2d(in_channels=self.channels[0],out_channels=self.channels[0],kernel_size=(256,3),stride=1))
        #self.decode_layers.append(nn.ConvTranspose2d(in_channels=self.channels[0],out_channels=self.channels[0],kernel_size=(256,3),stride=1))
        self.decode_layers.append(nn.Linear(2298,7659))
        self.decode_layers.append(nn.Linear(7659,12273*3))
        self.final_linear = nn.Sequential(
                nn.Linear(self.point_num[-1],12273*3),
                #nn.LeakyReLU(),
                #nn.Linear(1024,12273*3),
                )

        self.motion_net.append(nn.GRUCell(self.motion_input_dim,self.gru_dim))
        self.motion_net.append(nn.Sequential(MLP([self.gru_dim,self.point_num[1]*self.channels[1]],BN_dim=self.point_num[1]*self.channels[1],batch_norm=batch_norm,dropout=True,activate=True)))
        self.motion_net.append(nn.Sequential(MLP([self.gru_dim,self.point_num[-1]*self.channels[0]],BN_dim=self.point_num[-1]*self.channels[0],batch_norm=batch_norm,dropout=True,activate=True)))


    def encoder(self,pos,motion_weights=None,is_motion=False):

        batch_size = pos.shape[0]
        pos = pos.unsqueeze(1).float().to(self.device)  # [batch,1,n,3]

        if is_motion:
            x_mean = self.encode_mean[0](pos).view(batch_size,1,-1,3)  #[]
            x_mean = self.pooling(x_mean)
            x_mean = torch.mul(x_mean,motion_weights[0])
            x_mean = self.encode_mean[1](x_mean)
            #x_reduce = self.pooling(tmp).view(batch_size,-1,1) # 不使用激活函数貌似更好一点
            x_mean = torch.mul(self.pooling(x_mean).view(batch_size,-1,1),motion_weights[1])
            x_mean = F.leaky_relu(x_mean)

            x_var = self.encode_var[0](pos).view(batch_size,1,-1,3)  #[]
            x_var = self.pooling(x_var)
            x_var = torch.mul(x_var,motion_weights[0])
            x_var = self.encode_var[1](x_var)
            #x_reduce = self.pooling(tmp).view(batch_size,-1,1) # 不使用激活函数貌似更好一点
            x_var = torch.mul(self.pooling(x_var).view(batch_size,-1,1),motion_weights[1])
            x_var = F.sigmoid(x_var)
            
        else:

            x_mean = self.encode_mean[0](pos).view(batch_size,1,-1,3)  #[]
            x_mean = self.pooling(x_mean)
            x_mean = self.encode_mean[1](x_mean)
            x_mean = self.pooling(x_mean)
            x_mean = F.leaky_relu(x_mean)

            x_var = self.encode_var[0](pos).view(batch_size,1,-1,3)  #[]
            x_var = self.pooling(x_var)
            x_var = self.encode_var[1](x_var)
            x_var = self.pooling(x_var)
            x_var = torch.sqrt(F.relu(x_var)) # 正数

        return x_mean,x_var
    
    def decoder(self,reduce_pos,mean_=None,var_=None,motion_weights=None,is_motion=False):
        
        batch_size = reduce_pos.shape[0]
        reduce_pos = reduce_pos.unsqueeze(1).float()
        reduce_pos = torch.mul(reduce_pos ,var_.unsqueeze(1)) + mean_.unsqueeze(1)
        if is_motion:
             pass
            
        else:
            #x_recover = self.decode_layers[0](reduce_pos.view(batch_size,1,-1,1)).view(batch_size,-1)
            # print(x_recover.shape)
            x_recover = self.decode_layers[0](reduce_pos.view(batch_size,-1))
            x_recover = self.decode_layers[1](x_recover)
            #tmp = reduce_pos.view(batch_size,-1)

        #x_recover = self.final_linear(x_recover).view(batch_size,-1,3)
        #x_recover = tmp.view(batch_size,-1,3)  + self.final_linear(tmp).view(batch_size,-1,3) 
        return x_recover.view(batch_size,-1,3)
    
    def motion_weight(self,motion_data,hidden=None):
        # 输入动作信息[batch=500,63],学习decode解码对应的权重信息 -- [batch,9,9] -- [batch,gcn_dim[-1],gcn_dim[-1]] 
        batchsize = motion_data.shape[0]
        motion_data = motion_data.float().to(self.device)
        if hidden is None:
            hidden = torch.zeros(batchsize, self.gru_dim).to(self.motion_net[0].bias_hh.device)
        else:
            hidden = hidden.to(self.device)
        next_hidden = self.motion_net[0](motion_data,hidden).to(self.device)
        gru_out1 = self.motion_net[1](next_hidden).view((batchsize,1,self.point_num[1],self.channels[1])) #作为encoder的权重输入
        gru_out2 = self.motion_net[2](next_hidden).view((batchsize,self.point_num[-1],self.channels[0])) #作为encoder的权重输入

        return [gru_out1,gru_out2],next_hidden

class GCN_AE_PLUS(nn.Module):
    
    # 1. input:[12273,3] --> input*cosine:[12273,3] --> reshape: [3,12273]
    # 2. GNN_encoder: [3,12273],全连通邻接矩阵 --> [3,9]
    # 3. GNN_decoder: [3,9],全连通邻接矩阵 --> [3,12273]

    def __init__(self,motion_input_dim=63,
                 gru_dim=128,device="cuda:0",gcn_dim = [12273,128,9],pos_dim=[3,8,16],batch_norm=True,p=0):

        super(GCN_AE_PLUS, self).__init__()

        # 如果精度有限的话 -- 尝试增加gcn_dim的维度！
        
        self.device = device
        self.motion_input_dim = motion_input_dim
        self.gcn_dim = gcn_dim
        self.gru_dim = gru_dim
        self.pos_dim = pos_dim

        # 三维坐标固定:邻接信息也固定使用三个节点的全连通矩阵
        
        self.encode_layers = nn.ModuleList()
        self.decode_layers = nn.ModuleList()
        self.pos_conv = nn.ModuleList()
        self.pos_deconv = nn.ModuleList()
        self.motion_net = nn.ModuleList()

        self.p = p

        self.motion_net.append(nn.GRUCell(self.motion_input_dim,self.gru_dim))
        self.motion_net.append(nn.Sequential(MLP([self.gru_dim,self.gcn_dim[1]*self.pos_dim[0]],batch_norm=batch_norm,dropout=True,activate=True)))
        self.motion_net.append(nn.Sequential(MLP([self.gru_dim,self.gcn_dim[2]*self.pos_dim[1]],batch_norm=batch_norm,dropout=True,activate=True)))
        self.motion_net.append(nn.Sequential(MLP([self.gru_dim,self.gcn_dim[2]*self.pos_dim[2]],batch_norm=batch_norm,dropout=True,activate=True)))
        self.motion_net.append(nn.Sequential(MLP([self.gru_dim,self.gcn_dim[1]*self.pos_dim[1]],batch_norm=batch_norm,dropout=True,activate=True)))
            
        net_deepth = len(self.gcn_dim) - 1
        for i in range(net_deepth):
            # 编码解码 -- 同时对两个维度操作: 顶点维度下降的同时增加坐标层次的维度
            
            # 编码器与解码器对称 -- 顶点层次的降维升维 [12273,128,9]
            self.encode_layers.append(MLP([self.gcn_dim[i+1],self.gcn_dim[i+1]]))
            self.decode_layers.append(MLP([self.gcn_dim[net_deepth-i],self.gcn_dim[net_deepth-i]]))
            
            # 三维坐标的升维降维
            self.pos_conv.append(MLP([self.pos_dim[i],self.pos_dim[i+1]]))
            self.pos_deconv.append(MLP([self.pos_dim[net_deepth-i],self.pos_dim[net_deepth-i-1]]))
            
            # 动作数据对解码的影响 -- 顶点和
            
        #HIDDEN_SIZE = 128
        #self.en_fc = nn.Linear(self.pos_dim[-1]*self.gcn_dim[-1], HIDDEN_SIZE)
        #self.de_fc = nn.Linear(HIDDEN_SIZE, self.pos_dim[-1]*self.gcn_dim[-1])
        # 运行后发现使用cos/motion加权后数据变得很大,所以添加BN归一化 -- 只在使用加权的前提下启动这两个运算
        
        #self.encode_layers.append(nn.BatchNorm1d(self.pos_dim[0], affine=True, momentum=1e-3)) 
        #self.decode_layers.append(nn.BatchNorm1d(self.gcn_dim[-1], affine=True, momentum=1e-3))    

        # self.final_linear = nn.Linear(self.pos_dim[0],self.pos_dim[0])
        self.final_linear = nn.Sequential(
            # nn.Dropout(p=0),
            nn.Linear(self.pos_dim[0],self.pos_dim[0]),
            nn.PReLU(self.gcn_dim[0]),
            ) 

    def dropout_edges(self,input_edge_index,p=0.001,force_undirected=True):
        if self.training:
            edge_index, _ = dropout_edge(input_edge_index,p = self.p,force_undirected = force_undirected)
        else:
            edge_index = input_edge_index
        return edge_index
            
    def gcn_encoder(self,pos,lap_weights=None,motion_weights=None,is_motion=False):
        # encode_layers -- 注意是对顶点放缩,input要取转置
        # input: [batch,numvert,3]
        # [12273,3]-[3,12273]-[3,512]-[512,3]-[512,8]-[8,512]-[8,9]-[9,8]-[9,16]
        # pos_dim [3,8,16]   gcn_dim [12273,512,9]
        # lap_weights: [12273,128] [128,9]
        
        # 是否因为转置的操作导致结果不平滑 == 修改为reshape?
        #lap_weights[0] = lap_weights[0].float().to(self.device)
        #lap_weights[1] = lap_weights[1].float().to(self.device)
        lap_weights[0] = lap_weights[0].float()
        lap_weights[1] = lap_weights[1].float().to(self.device)
        batch_size = pos.shape[0]
        #pos = pos.view((batch_size, -1, 3)).float().to(self.device)  # [batch,numverts,3]
        pos = pos.view((batch_size, -1, 3)).float()  # [batch,numverts,3]

        smoothed_x0 = torch.zeros((batch_size, self.gcn_dim[1], self.pos_dim[1])).to(self.device) 
        smoothed_x1 = torch.zeros((batch_size, self.gcn_dim[2], self.pos_dim[2])).to(self.device)

        if is_motion:

            for i in range(batch_size):

                tmp = torch.matmul(lap_weights[0].T,pos[i]).transpose(1,0).to(self.device) # [12273,3]==[128,3]==[3,128]
                tmp = self.encode_layers[0](tmp).flatten()
                tmp = torch.mul(tmp,motion_weights[0][i]).view(self.gcn_dim[1],self.pos_dim[0]) #[3,128]=[128,3]
                smoothed_x0[i] = F.leaky_relu(self.pos_conv[0](tmp)) #[128,8]
                
            if not self.training:
                smoothed_x0 = smoothed_x0.detach()

            for i in range(batch_size):
            
                tmp = torch.matmul(lap_weights[1].T,smoothed_x0[i]).transpose(1,0) # [128,8]=[9,8]==[8,9]
                tmp = self.encode_layers[1](tmp).flatten()
                tmp = torch.mul(tmp,motion_weights[1][i]).view(self.gcn_dim[2],self.pos_dim[1])
                smoothed_x1[i] = F.leaky_relu(self.pos_conv[1](tmp)) #[9,16]
                
            if not self.training:
                smoothed_x1 = smoothed_x1.detach()
        
        else:
            for i in range(batch_size):
            
                tmp = torch.matmul(lap_weights[0].T,pos[i]).transpose(1,0).to(self.device)  # [12273,3]==[128,3]==[3,128]
                #tmp = torch.matmul(lap_weights[0].T,pos[i]).view(self.pos_dim[0],self.gcn_dim[1]) # [128,3]==[3,128]
                tmp = self.encode_layers[0](tmp).view(self.gcn_dim[1],self.pos_dim[0]) #[3,128]=[128,3]
                #tmp = self.encode_layers[0](tmp).transpose(1,0) #[3,128]=[128,3]
                smoothed_x0[i] = F.leaky_relu(self.pos_conv[0](tmp)) #[128,8]
                
            if not self.training:
                smoothed_x0 = smoothed_x0.detach()

            for i in range(batch_size):
            
                tmp = torch.matmul(lap_weights[1].T,smoothed_x0[i]).transpose(1,0) # [128,8]=[9,8]==[8,9]
                #tmp = torch.matmul(lap_weights[1].T,smoothed_x0[i]).view(self.pos_dim[1],self.gcn_dim[2])
                tmp = self.encode_layers[1](tmp).view(self.gcn_dim[2],self.pos_dim[1])
                #tmp = self.encode_layers[1](tmp).transpose(1,0) #[8,9]=[9,8]
                smoothed_x1[i] = F.leaky_relu(self.pos_conv[1](tmp)) #[9,16]
                
            if not self.training:
                smoothed_x1 = smoothed_x1.detach()
            
        x_reduce = smoothed_x1.view(batch_size,-1)
        # x_reduce = self.en_fc(smoothed_x1.view(batch_size,-1))
        return x_reduce
    
    def gcn_decoder(self,reduce_pos,lap_inv_weights=None,motion_weights=None,is_motion=False):
        
        # reduce: [batch,reduce_numvert,new_pos_dim] ; motion_weight:[batch,9*16]
        lap_inv_weights[0] = lap_inv_weights[0].float().to(self.device)
        lap_inv_weights[1] = lap_inv_weights[1].float().to(self.device)
        #lap_inv_weights[1] = lap_inv_weights[1].float().to(self.device)
        batch_size = reduce_pos.shape[0]
        
        reduce_pos = reduce_pos.view((batch_size,self.pos_dim[-1],self.gcn_dim[-1])).float() #[b,16,9] 
        
        recover_x0 = torch.zeros((batch_size, self.gcn_dim[1], self.pos_dim[1])).to(self.device)
        recover_x1 = torch.zeros((batch_size, self.gcn_dim[0], self.pos_dim[0])).to(self.device)
        
        if is_motion:
            
            for i in range(batch_size):
                
                tmp = self.decode_layers[0](reduce_pos[i]).flatten() # [16*9]
                tmp = torch.mul(motion_weights[0][i],tmp).view(self.gcn_dim[2],self.pos_dim[2]) # [9,16]
                tmp = torch.matmul(lap_inv_weights[0],tmp) # [9,16]==[128,16]
                recover_x0[i] = F.leaky_relu(self.pos_deconv[0](tmp)) #[128,8]

            if not self.training:
                recover_x0 = recover_x0.detach()

            for i in range(batch_size):
                
                tmp = self.decode_layers[1](recover_x0[i].view(self.pos_dim[1],self.gcn_dim[1])).flatten()
                tmp = torch.mul(motion_weights[1][i],tmp).view(self.gcn_dim[1],self.pos_dim[1])
                tmp = torch.matmul(lap_inv_weights[1],tmp).to(self.device) # [12273,8]
                recover_x1[i] = F.leaky_relu(self.pos_deconv[1](tmp)) #[12273,3]

            if not self.training:
                recover_x1 = recover_x1.detach()            
            
        else:
            for i in range(batch_size):
                
                #tmp = self.decode_layers[0](reduce_pos[i]).transpose(1,0)
                tmp = self.decode_layers[0](reduce_pos[i]).view(self.gcn_dim[2],self.pos_dim[2]) # [16,9]=[9,16]
                tmp = torch.matmul(lap_inv_weights[0],tmp) # [9,16]==[128,16]
                recover_x0[i] = F.leaky_relu(self.pos_deconv[0](tmp)) #[128,8]

            if not self.training:
                recover_x0 = recover_x0.detach()

            for i in range(batch_size):

                #tmp = self.decode_layers[1](recover_x0[i].transpose(1,0)).transpose(1,0) 
                tmp = self.decode_layers[1](recover_x0[i].transpose(1,0)).view(self.gcn_dim[1],self.pos_dim[1])
                tmp = torch.matmul(lap_inv_weights[1],tmp).to(self.device) # [12273,8]
                recover_x1[i] = F.leaky_relu(self.pos_deconv[1](tmp)) #[12273,3]

            if not self.training:
                recover_x1 = recover_x1.detach()

        recover_x1 = recover_x1.view(batch_size,-1,3)
        x_recover = recover_x1 + self.final_linear(recover_x1).view(batch_size,-1,3)  # 残差网络也使用了！
        return x_recover

    def motion_weight(self,motion_data,hidden=None):
        # 输入动作信息[batch=500,63],学习decode解码对应的权重信息 -- [batch,9,9] -- [batch,gcn_dim[-1],gcn_dim[-1]] 
        batchsize = motion_data.shape[0]
        motion_data = motion_data.float().to(self.device)
        if hidden is None:
            hidden = torch.zeros(batchsize, self.gru_dim).to(self.motion_net[0].bias_hh.device)
        else:
            hidden = hidden.to(self.device)
        next_hidden = self.motion_net[0](motion_data,hidden).to(self.device)
        gru_out1 = self.motion_net[1](next_hidden).view((batchsize,self.gcn_dim[1]*self.pos_dim[0])) #作为encoder的权重输入
        gru_out2 = self.motion_net[2](next_hidden).view((batchsize,self.gcn_dim[2]*self.pos_dim[1])) #作为encoder的权重输入
        gru_out3 = self.motion_net[3](next_hidden).view((batchsize,self.gcn_dim[2]*self.pos_dim[2])) #作为decoder的权重输入
        gru_out4 = self.motion_net[4](next_hidden).view((batchsize,self.gcn_dim[1]*self.pos_dim[1])) #作为decoder的权重输入


        return [gru_out1,gru_out2,gru_out3,gru_out4],next_hidden
