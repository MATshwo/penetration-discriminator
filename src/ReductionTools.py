# 工具函数定义: Loss,collision,laplace
import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from torch_geometric.utils import get_laplacian,to_dense_adj,add_self_loops
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm,gcn_norm_my
from torch_geometric.nn.conv import MessagePassing
from src.obj_parser import Mesh_obj
import fcl
from sklearn.decomposition import PCA,KernelPCA,TruncatedSVD,FactorAnalysis,DictionaryLearning,NMF
from sklearn.manifold import Isomap,MDS,LocallyLinearEmbedding,TSNE
import scipy
from scipy.spatial.transform import Rotation
import copy



class reduction_tool_plus():
    # 对原始布料模型的顶点降维&还原工具升级版 -- 顶点层次的降维:非三维坐标&时间长度
    # 支持PCA/KPCA/...等多种降维方式

    def __init__(self,init_pos,method="PCA",n_componment=3):
        self.choice = ["PCA","KPCA","FA","TSVD","DictLearn","NMF"]
        self.method = method
        self.init_pos = init_pos.T #[3,numverts] -> 初始顶点坐标数据:取转置为了保证对顶点数目降维
        self.n_componment = n_componment
    def init_process(self):
        # 首次降维操作: 对初始布料模型(不包含任何动作信息)降维,并保存fit后的模型作为预训练模型
        # 由于初始布料模型相当于只有一个样本 - 无法降维,只能使用三维坐标作为三个样本降维 -- 主成分 <= 3

        if self.method == "PCA":
            self.tool = PCA(n_components=self.n_componment) 
            self.tool.fit(self.init_pos)
            #new = self.tool.fit_transform(self.init_pos)
            #print(new)
            #new_ori = self.tool.inverse_transform(new)
            #print(new_ori[:,:3].T)
            print("PCA_compoments:{},dim = {}.".format(np.sum(self.tool.explained_variance_ratio_),self.n_componment))

        elif self.method == "KPCA":
            self.tool = KernelPCA(n_components=self.n_componment,kernel="rbf",gamma=20,degree=3,fit_inverse_transform=True) 
            self.tool.fit(self.init_pos)
            #explained_variance_ = np.var(new,axis=0)
            print("KPCA_compoments:{},dim = {}.".format(1.0,self.n_componment))
        
        elif self.method == "TSVD":
            """截断SVD分解"""
            self.tool = TruncatedSVD(n_components = 3,n_iter = 7,random_state = 3439)
            self.tool.fit(self.init_pos)
            print("TSVD_compoments:{},dim = {}.".format(1.0,self.n_componment))
        
        elif self.method == "DictLearn":
            self.tool = DictionaryLearning(n_components=self.n_componment,transform_algorithm='lasso_lars',transform_alpha=0.1,random_state=3439)
            self.tool.fit(self.init_pos)
            print("DictLearning: ...,dim = {}.".format(self.n_componment))
        
        elif self.method == "FA":
            self.tool = FactorAnalysis(n_components=self.n_componment,random_state=3439) 
            self.tool.fit(self.init_pos)
            print("FA_process: ...,dim = {}.".format(self.n_componment))

        elif self.method == "NMF":
            self.tool = NMF(n_components=self.n_componment,init="random",random_state=3439,solver="mu",max_iter=100) 
            # input_data = np.exp(self.init_pos) # 将原始数据转换为非负
            input_data = self.init_pos - np.min(self.init_pos)
            self.tool.fit(input_data)
            #new = self.tool.fit_transform(input_data)
            #print(new)
            #new_ori = self.tool.inverse_transform(new) + np.min(self.init_pos)
            #print(new_ori[:,:3].T)
            print("NMF process :...,dim = {}.".format(self.n_componment))
           
    def dim_reduction(self,x,is_fit=False):
        # 对输入数据进行降维 -- 可使用预训练的模型降维;也支持结合当前数据重新fit模型降维
        batch_size = x.shape[0]
        x = torch.transpose(x,2,1).reshape(batch_size*3,-1) # x[batch,numvert,3] -> [batch*3,-1]
        x_ = x.detach().clone() # 从计算图剥离并拷贝

        # print(x_.shape)
        if not is_fit:
            if self.method == "KPCA":
                # 如何让KPCA也能反向传播梯度是个问题？ -- 暂时保留 -- 如何求dK/dx ? -- 0415运行时发现梯度正常? -- 搞不懂
                # 训练的话只能暂时先在低维空间计算Loss,高维空间难以求导 -- 不过只在低维PCA空间计算loss感觉也可行,试试看
                
                x_ = self.tool._validate_data(x_.cpu().numpy(), accept_sparse="csr", reset=False)
                K = self.tool._centerer.transform(self.tool._get_kernel(x_, self.tool.X_fit_)) # [3,12273]
                non_zeros = np.flatnonzero(self.tool.eigenvalues_)
                scaled_alphas = np.zeros_like(self.tool.eigenvectors_)
                scaled_alphas[:, non_zeros] = self.tool.eigenvectors_[:, non_zeros] / np.sqrt(self.tool.eigenvalues_[non_zeros])
                x_reduce = torch.from_numpy(np.dot(K, scaled_alphas)).to(torch.float32).to(x.device)
                #x_reduce = torch.from_numpy(self.tool.transform(x.cpu().numpy())).to(x.device).to(torch.float32) 

            elif self.method == "PCA":
                x_reduce = torch.matmul(x - torch.from_numpy(self.tool.mean_).to(x.device), torch.from_numpy(self.tool.components_.T).to(x.device)).to(torch.float32) 

            elif self.method == "TSVD":
                x_reduce = torch.matmul(x, torch.from_numpy(self.tool.components_.T).to(x.device)).to(torch.float32)
            
            elif self.method == "DictLearn":
                x_reduce = torch.matmul(x, torch.from_numpy(self.tool.components_.T).to(x.device)).to(torch.float32) # shape:[3,3];[batchsize*3,reduce_dim]
                mask = ~(x_reduce == (torch.max(x_reduce,dim = -2)[0].repeat(x_reduce.shape[0],1)))
                #mask = not(x_reduce == (torch.max(x_reduce,dim = -2)[0].repeat(x_reduce.shape[0],1))) # 不能用not,会报错 -- not应该只能作用在单个逻辑变量..
                x_reduce[mask] = 0  # 保留第一个维度的最大值,其余=0,按照源码transfrom的结果执行此操作 -- 原理暂不清楚
            
            elif self.method == "FA":        
                Ih = np.eye(len(self.tool.components_))
                x_transformed = x - torch.from_numpy(self.tool.mean_).to(x.device)
                # FA中的Wpsi*Wpsi.T不等于单位阵,所以后续recover阶段需要使用pinv求解Wpsi.T矩阵的伪逆
                self.tool.Wpsi = self.tool.components_ / self.tool.noise_variance_
                self.tool.cov_z_inv = Ih + np.dot(self.tool.Wpsi, self.tool.components_.T)
                cov_z = torch.from_numpy(scipy.linalg.inv(self.tool.cov_z_inv)).to(x.device)
                tmp = torch.matmul(x_transformed,torch.from_numpy(self.tool.Wpsi.T).to(x.device))
                x_reduce = torch.matmul(tmp,cov_z).to(torch.float32)          
            
            elif self.method == "NMF":
                
                #input_x = np.exp(x_.cpu().numpy()) # 使用指数函数作为非负转换工具
                input_x = x_.cpu().numpy()
                self.min_temp = np.min(input_x)
                input_x = input_x - self.min_temp # 指数函数会导致数据爆炸-- 改用减去最小值

                x_reduce = torch.matmul(x-torch.min(x), torch.from_numpy(scipy.linalg.pinv(self.tool.components_)).to(torch.float32).to(x.device))
                #x_reduce = torch.from_numpy(self.tool.transform(input_x.astype(np.float64))).to(torch.float32).to(x.device)
                # 直接使用官方接口计算结果有误差 -- 暂不清楚原因
                
        else:
            if self.method == "PCA":
                self.tool = PCA(n_components=self.n_componment) 
                self.tool.fit(x_.cpu().numpy()) # 刷新当前类的pca_tool
                x_reduce = torch.matmul(x - torch.from_numpy(self.tool.mean_).to(x.device), torch.from_numpy(self.tool.components_.T).to(x.device)).to(torch.float32) 

                
            elif self.method == "KPCA":
                self.tool = KernelPCA(n_components=self.n_componment,kernel="rbf",gamma=20,degree=3,fit_inverse_transform=True) 
                self.tool.fit(x_.cpu().numpy())
                x_ = self._validate_data(x_.cpu().numpy(), accept_sparse="csr", reset=False)
                K = self.tool._centerer.transform(self.tool._get_kernel(x_, self.tool.X_fit_))
                non_zeros = np.flatnonzero(self.tool.eigenvalues_)
                scaled_alphas = np.zeros_like(self.tool.eigenvectors_)
                scaled_alphas[:, non_zeros] = self.tool.eigenvectors_[:, non_zeros] / np.sqrt(self.tool.eigenvalues_[non_zeros])
                x_reduce = torch.from_numpy(np.dot(K, scaled_alphas)).to(torch.float32).to(x.device)
            
            elif self.method == "TSVD":
                self.tool = TruncatedSVD(n_components = 3,n_iter = 7,random_state = 3439)
                self.tool.fit(x_.cpu().numpy())
                x_reduce = torch.matmul(x, torch.from_numpy(self.tool.components_.T).to(x.device)).to(torch.float32)
                
        
            elif self.method == "DictLearn":
                self.tool = DictionaryLearning(n_components=self.n_componment,transform_algorithm='lasso_lars',transform_alpha=0.1,random_state=3439)
                self.tool.fit(x_.cpu().numpy())
                x_reduce = torch.matmul(x, torch.from_numpy(self.tool.components_.T).to(x.device)).to(torch.float32)
                mask = ~(x_reduce == (torch.max(x_reduce,dim = -2)[0].repeat(x_reduce.shape[0],1)))
                x_reduce[mask] = 0  
                
            elif self.method == "FA":
                self.tool = FactorAnalysis(n_components=self.n_componment,random_state=3439) 
                self.tool.fit(x_.cpu().numpy())
                
                Ih = np.eye(len(self.tool.components_))
                x_transformed = x - torch.from_numpy(self.tool.mean_).to(x.device)
                self.tool.Wpsi = self.tool.components_ / self.tool.noise_variance_
                self.tool.cov_z_inv = Ih + np.dot(self.tool.Wpsi, self.tool.components_.T)
                cov_z = torch.from_numpy(scipy.linalg.inv(self.tool.cov_z_inv)).to(x.device)
                tmp = torch.matmul(x_transformed,torch.from_numpy(self.tool.Wpsi.T).to(x.device))
                x_reduce = torch.matmul(tmp,cov_z).to(torch.float32) 
 
            elif self.method == "NMF":
                #input_x = np.exp(x_.cpu().numpy())
                input_x = x_.cpu().numpy()
                self.min_temp = np.min(input_x)
                input_x = input_x - self.min_temp
                self.tool = NMF(n_components=self.n_componment,init="random",random_state=3439) 
                self.tool.fit(input_x)
                
                x_reduce = torch.matmul(x-torch.min(x), torch.from_numpy(scipy.linalg.pinv(self.tool.components_)).to(torch.float32).to(x.device)) 
                #x_reduce = torch.from_numpy(self.tool.transform(input_x.astype(np.float64))).to(torch.float32).to(x.device)  
                # 直接使用官方接口计算结果有误差 -- 暂不清楚原因
                
        return x_reduce.reshape(batch_size,1,-1,3)
     
    def dim_recover(self,x):
        # x:[batchsize,3*3]
   
        batch_size = x.shape[0]
        x = x.reshape(x.shape[0]*3,-1)
        x_ = x.detach().clone() # 从计算图剥离并拷贝

        if self.method == "PCA":
            x_recover = (torch.matmul(x.float(),torch.from_numpy(self.tool.components_).float().to(x.device)) + torch.from_numpy(self.tool.mean_).float().to(x.device)).T # pca.components_ :[pca_dim,12273]pca转换矩阵
            #x_recover = self.tool.inverse_transform(x).T
        
        elif self.method == "KPCA":
            # kernel貌似很难重构到原始空间 QAQ : 即低维接近的点重构到原始空间 == 测试之后重构值=真实值的二分之一,why?
            # 此处的K矩阵可以用来计算二者之间的相关性,或许可以利用下 -- 或者直接利用这个接口计算两个大型矩阵之间的相关性 -- 还支持cos,...其他多种运算
            K = self.tool._get_kernel(x_.cpu().numpy(), self.tool.X_transformed_fit_)
            
            # 乘以二倍之后才等于原始数据,暂不明白原因
            x_recover = 2.0 * torch.from_numpy(np.dot(K, self.tool.dual_coef_)).to(torch.float32).to(x.device).T

        elif self.method == "TSVD":
            x_recover = (torch.matmul(x.float(),torch.from_numpy(self.tool.components_).float().to(x.device))).T
        
        elif self.method == "DictLearn":
            x_recover = (torch.matmul(x.float(),torch.from_numpy(self.tool.components_).float().to(x.device))).T

        elif self.method == "FA":
            tmp = torch.matmul(x.float(),torch.from_numpy(self.tool.cov_z_inv).float().to(x.device))             
            x_recover = torch.matmul(tmp,torch.from_numpy(scipy.linalg.pinv(self.tool.Wpsi.T)).to(torch.float32).to(x.device))
            x_recover = (x_recover + torch.from_numpy(self.tool.mean_).float().to(x.device)).T
            
        elif self.method == "NMF":
            #x_recover = torch.log(torch.matmul(x.float(),torch.from_numpy(self.tool.components_).float().to(x.device))).T            
            x_recover = (torch.matmul(x.float(),torch.from_numpy(self.tool.components_).float().to(x.device)) + self.min_temp).T 

        return x_recover.reshape(batch_size,-1,3)

if __name__ == "__main__":

    # 测试降维重构结果与初始值结果是否一致

    init = np.random.randn(1000,3).to("cuda:0")
    vertextools = reduction_tool_plus(init,method="FA")
    vertextools.init_process()
    x_reduce = vertextools.dim_reduction(torch.from_numpy(init).reshape(1,-1,3))
    x_recover = vertextools.dim_recover(x_reduce.reshape(1,-1))
    print(x_reduce[0],"\n",x_recover[:,:3,:],"\n",init[:3,:])