import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from src.w_models_sim import *
from src.Mytools import *
from pe_dataloader import *
from sklearn import metrics
import time
def testing():

    # 判别器测试流程
    #1.参数设置和模型准备
    #input:[batchsize,3]
    #label:[batchsize,1]
    #motion:[batchsize,20813,3]
    cloth_numverts = 26718
    human_numverts = 20813
    data_path="datasets/train_data/Pe_0828" 

    device = "cuda:2"
    batchsize = 128 # 26718 # batch:>=256会内存溢出
    frame_len = 1

    D = Penetration_D(device=device).to(device)
    
    # 测试阶段的weights均为1,不需要根据实际label值进行加权,所以直接在外层指定loss
    loss_D = nn.BCELoss()
    
    # 加载预训练的模型
    date = "test" #用data对应预训练的某个模型 -- 个人习惯指定 -- 将每个模型用一个日期/其他字符对应表示
    D_path = "./datasets/res_gpu/{}/D_{}_gpu.pth.tar".format("1100","1100")
    if os.path.exists(D_path):
        # 如果存在,就在原始模型上继续训练
        D.load_state_dict(torch.load(D_path)) 
        print("Loading Exist model finshed!")

    #2.数据加载 -- 850帧:[1-750]训练;[750-850]测试
    test_data = Penetration_data(path=data_path,frame_len=frame_len,start=750)
    print("Data loading finished!")
    test_loader = DataLoader(dataset=test_data,batch_size=batchsize,shuffle=True,drop_last=True,num_workers=0)

    #3.启动测试
    D.eval() 
    
    loss_list = []
    Accuracy_list = []
    Recall_list,FPR_list = [],[]
    label_total = np.array([])
    pred_total = np.array([])

    for _,(v_pos,motion,label) in enumerate(test_loader):

        loss_ = 0.0
        start = time.time()*1000
        pred = D(v_pos.to(device),motion.to(device))
        end = time.time()*1000
        #print("最近邻计算时间:%f ms"%(end-start))
        label = label.float().to(device)
        #print(pred,label)
        loss_ = loss_D(pred,label)


        # 根据不同的阈值计算准确率和召回率:注意这里的metrices不是sklearn包中的metrics,命名搞混了
        predict_mine = torch.where(pred >= 8.86563212573006e-15, 1, 0)
        metrices = Classification_indicator(predict_mine,label)  # pred预测是概率值,四舍五入转换为类别
        accuracy = metrices.Accuracy()
        
        if (label==1).any():

            # label中包含正样本时:计算召回率,并记录当前batch的label和pred值用于后续绘制ROC曲线
            recall = metrices.Recall()
            Recall_list.append(recall.item())
            #fpr = metrices.FPR()
            #FPR_list.append(fpr.item())
            
            # np.hstack水平堆叠
            label_total = np.hstack((label_total,label[:,0].cpu().detach().numpy()))
            pred_total  = np.hstack((pred_total,pred[:,0].cpu().detach().numpy()))
            print("BCE_Loss:{:.3f},Recall:{:.3f},Accuracy:{:.3f}..".format(loss_.item(),recall.item(),accuracy.item()))
        
        else:
            print("BCE_Loss:{:.3f},Accuracy:{:.3f}..".format(loss_.item(),accuracy.item()))

        loss_list.append(loss_.item())
        Accuracy_list.append(accuracy.item())
            
    # 每个Epoch绘制ROC曲线
    if label_total.size != 0 :
        fpr_list, tpr_list, thresholds = metrics.roc_curve(label_total,pred_total,pos_label=1)
        best_threshold = thresholds[np.argmax(tpr_list - fpr_list)] # 寻找最佳阈值
        print(best_threshold)
        roc_auc = metrics.auc(fpr_list, tpr_list)
        plot_func(date=date,title="ROC CURVE",plot_x=fpr_list,plot_y=tpr_list,lens=len(loss_list),mode="Roc",auc=roc_auc)

    # 每个Epoch保存收敛结构
    np.save('./datasets/res/{}/BCE_gpu_{}.npy'.format(date,date),np.array(loss_list))
    np.save('./datasets/res/{}/Accuracy_gpu_{}.npy'.format(date,date),np.array(Accuracy_list))
    np.save('./datasets/res/{}/Recall_gpu_{}.npy'.format(date,date),np.array(Recall_list))
    plot_func(date=date,title="BCE_Loss",plot_x=range(len(loss_list)),plot_y=loss_list,lens=len(loss_list))
    plot_func(date=date,title="Accuracy",plot_x=range(len(Accuracy_list)),plot_y=Accuracy_list,lens=len(Accuracy_list))
    plot_func(date=date,title="Recall",plot_x=range(len(Recall_list)),plot_y=Recall_list,lens=len(Recall_list))

def plot_func(date="0",title=None,plot_x=None,plot_y=None,lens=0,id=0,mode="Loss",auc=0):
    
    train_num = lens
    f = plt.figure(figsize=(12,8))
    
    if mode == "Loss":

        plt.plot(plot_x,plot_y,label=title)
        plt.yticks(fontproperties = 'Times New Roman', size = 25)
        plt.xticks(fontproperties = 'Times New Roman', size = 25)
        plt.legend(prop={'family' : 'Times New Roman', 'size' : 30})
        plt.title("{}".format(title),fontdict={'family' : 'Times New Roman', 'size' : 30})
        f.savefig(os.path.join("./datasets/res_gpu/{}".format(date),'{}_gpu_{}.png'.format(title,id)),dpi=300)
        plt.close()

    elif mode == "Roc":
        
        plt.plot(plot_x,plot_y,linewidth=3,label=title,color='steelblue',alpha=0.8)
        plt.scatter(plot_x,plot_y,color='r',s=100,alpha=1)

        plt.plot([0, 1], [0, 1],linestyle='--', color='green', label='Random Classifier')  # 绘制随机分类器的ROC曲线
        plt.fill_between(plot_x,plot_y,0, color='lightyellow',alpha=0.6,label="Val AUC = %0.3f" % auc)

        plt.xlabel("FPR", fontdict={'family' : 'Times New Roman', 'size' : 30})
        plt.ylabel('TPR', fontdict={'family' : 'Times New Roman', 'size' : 30})
        plt.yticks(fontproperties = 'Times New Roman', size = 25)
        plt.xticks(fontproperties = 'Times New Roman', size = 25)
        plt.legend(prop={'family' : 'Times New Roman', 'size' : 30})

        plt.title("{}".format(title),fontdict={'family' : 'Times New Roman', 'size' : 30})
        f.savefig(os.path.join("./datasets/res_gpu/{}".format(date),'{}_gpu_{}.png'.format(title,id)),dpi=300)
        plt.close()

if __name__ == "__main__":

    testing()
    