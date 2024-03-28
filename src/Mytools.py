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
import matplotlib.pyplot as plt 

class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    


# 分类评估指标
# 混淆矩阵,f1_score,ROC,AUC,...
class Classification_indicator():
    def __init__(self,pred,label):
        self.pred = pred
        self.label = label

        self.TP = np.sum(((self.pred==1)&(self.label==1)))
        self.TN = np.sum(((self.pred==0)&(self.label==0)))
        self.FP = np.sum(((self.pred==1)&(self.label==0)))
        self.FN = np.sum(((self.pred==0)&(self.label==1)))
    def Precision(self):
        # 分类正确的正样本数 (TP) 占 预测为正样本的总样本数 (TP + FP) 
        return self.TP/(self.TP+self.FP)
    def Recall(self):
    #  分类正确的正样本数 (TP) 占 实际为正样本的总样本数 (TP + FP) 
        return self.TP/(self.TP+self.FN)
    def FPR(self):
        # 假阳率
        return (self.FP)/(self.TN+self.FP)
    def ErrorRate(self):
        return (self.FP+self.FN)/(self.TP+self.TN+self.FP+self.FN)
    def Accuracy(self):
        return (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)
        
    def F1_score(self):
        return 2*self.Precision()*self.Recall()/(self.Precision()+self.Recall())
    def F_score(self,a):
        return (1+a^2)*self.Precision()*self.Recall()/(a^2*self.Precision()+self.Recall())


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
        log_ = torch.nn.LogSoftmax(dim=1)
        loss_fn = torch.nn.NLLLoss()
        log_pos = log_(pos)
        return loss_fn(log_pos,pred_pos)
    
    elif mode == "KL":
        logp_y = F.log_softmax(pred_pos, dim = -1)
        p_x = F.softmax(pos, dim =- 1)

        dist = F.kl_div(logp_y, p_x, reduction='batchmean')
        return dist   


if __name__ == "__main__":

    pass
