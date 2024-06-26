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
import seaborn as sns
from sklearn.metrics import confusion_matrix

def training():

    #1.参数设置和模型准备

    cloth_numverts = 26718
    human_numverts = 20813
    data_path="datasets/train_data/Pe_0827" 

    device = "cuda:2"
    batchsize = 128 
    Epochs = 25 
    frame_len = 30
    D = Penetration_D(device=device).to(device)

    date0 = "0319"
    model0 = "0319"
    root_path = "./datasets/res_gpu/{}".format(date0)
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    optimizer = torch.optim.Adam(D.parameters(),lr=0.01)

    # #2.数据加载 -- 放在epoch外

    #3.启动训练
    D.train()
    for t in range(5):
        date = "0319{}".format(t)
        model = "0319{}".format(t)
        save_path = "./datasets/res_gpu/{}".format(date)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        loss_list = []
        Accuracy_list = []
        Recall_list,FPR_list = [],[]
        threshold_list = []

        for epoch in range(Epochs):
    
            train_data = Penetration_data(path=data_path,frame_len=frame_len,start=epoch*frame_len)
            train_loader = DataLoader(dataset=train_data,batch_size=batchsize,shuffle=True,drop_last=True,num_workers=0)

            loss_total = 0.0
            temp = []
            label_total = np.array([])
            pred_total = np.array([])

            iters = 0 

            for _,(v_pos,motion,label) in enumerate(train_loader):
                
                iters += 1
                loss_ = 0.0
                pred = D(v_pos.to(device),motion.to(device))
                
                weights = torch.ones_like(label).to(device)  
                label = label.long()
                weights[label==1] = 10
                weights[label==0] = 1
                label = label.float().to(device)

                loss_D = nn.BCELoss(weight=weights)
                loss_ = loss_D(pred,label)
                loss_total += loss_.item()
                optimizer.zero_grad()
                loss_.backward()
                optimizer.step()
                temp.append(loss_.item())

                label_total = np.hstack((label_total,label[:,0].cpu().detach().numpy()))
                pred_total  = np.hstack((pred_total,pred[:,0].cpu().detach().numpy()))
            
                print("Epoch:{}/{},BCE_Loss:{:.3f}..".format(epoch+1,Epochs,loss_.item()))

            if label_total.size != 0 :
                fpr_list, tpr_list, thresholds = metrics.roc_curve(label_total,pred_total,pos_label=1)
                best_threshold = thresholds[np.argmax(tpr_list - fpr_list)] # 寻找最佳阈值
                threshold_list.append(best_threshold.item())

                roc_auc = metrics.auc(fpr_list, tpr_list)
                plot_func(date=date,title="ROC CURVE",plot_x=fpr_list,plot_y=tpr_list,lens=iters,id=epoch,mode="Roc",auc=roc_auc)
            
            # 根据最佳阈值计算准确率和召回率
            predict_mine = np.where(pred_total >= best_threshold, 1, 0)
            metrices = Classification_indicator(predict_mine,label_total)  
            accuracy = metrices.Accuracy()
            Accuracy_list.append(accuracy.item())
            loss_list.append(loss_total/iters)

            if (label_total==1).any() :

                recall = metrices.Recall()
                Recall_list.append(recall.item())
                print("Epoch:{}/{},BCE_Loss:{:.3f},Recall:{:.3f},Accuracy:{:.3f}..".format(epoch+1,Epochs,loss_total/iters,recall.item(),accuracy.item()))
            else:
                print("Epoch:{}/{},BCE_Loss:{:.3f},Accuracy:{:.3f}..".format(epoch+1,Epochs,loss_total/iters,accuracy.item()))

            plot_confusion(np.array(predict_mine),np.array(label_total),date=date,epoch=epoch)

            torch.save(D.state_dict(),"./datasets/res_gpu/{}/D_{}_gpu.pth.tar".format(date,date))
            
            # 绘制最佳阈值曲线
            np.save('./datasets/res_gpu/{}/yuzhi_gpu_{}.npy'.format(date,epoch),np.array(threshold_list))
            plot_func(date=date,title="Best threshold",plot_x=range(len(threshold_list)),plot_y=threshold_list,lens=len(threshold_list),id=epoch)

            np.save('./datasets/res_gpu/{}/BCE_gpu_{}.npy'.format(date,epoch),np.array(temp))
            plot_func(date=date,title="BCE_Loss",plot_x=range(len(loss_list)),plot_y=loss_list,lens=len(loss_list),id=epoch)

def plot_func(date="0",title=None,plot_x=None,plot_y=None,lens=0,id=0,mode="Loss",auc=0):
    
    train_num = lens
    plt.rcParams.update(plt.rcParamsDefault)
    f = plt.figure(figsize=(12,8),facecolor="white")
    
    if mode == "Loss":

        plt.plot(plot_x,plot_y,label=title)
        plt.grid(False)
        plt.yticks(fontproperties = 'Times New Roman', size = 25)
        plt.xticks(fontproperties = 'Times New Roman', size = 25)
        plt.legend(prop={'family' : 'Times New Roman', 'size' : 30})
        plt.title("{}".format(title),fontdict={'family' : 'Times New Roman', 'size' : 30})
        f.savefig(os.path.join("./datasets/res_gpu/{}".format(date),'{}_gpu_{}.png'.format(title,id)),dpi=300,bbox_inches='tight')
        plt.close()

    elif mode == "Roc":
        
        plt.plot(plot_x,plot_y,linewidth=3,label=title,color='steelblue',alpha=0.8)
        plt.scatter(plot_x,plot_y,color='red',s=100,alpha=1)

        plt.plot([0, 1], [0, 1],linestyle='--', color='green', label='Random Classifier')  # 绘制随机分类器的ROC曲线
        plt.fill_between(plot_x,plot_y,0, color='lightyellow',alpha=0.6,label="Val AUC = %0.3f" % auc)
        plt.grid(False)

        plt.xlabel("FPR", fontdict={'family' : 'Times New Roman', 'size' : 30})
        plt.ylabel('TPR', fontdict={'family' : 'Times New Roman', 'size' : 30})
        plt.yticks(fontproperties = 'Times New Roman', size = 25)
        plt.xticks(fontproperties = 'Times New Roman', size = 25)
        plt.legend(prop={'family' : 'Times New Roman', 'size' : 30})
        plt.title("{}".format(title),fontdict={'family' : 'Times New Roman', 'size' : 30})
        f.savefig(os.path.join("./datasets/res_gpu/{}".format(date),'{}_gpu_{}.png'.format(title,id)),dpi=300,bbox_inches='tight')
        plt.close()

def plot_confusion(y_pred, y_true,date=0,epoch=0):

        cm = confusion_matrix(y_true, y_pred)
        f = plt.figure(figsize=(12,8))
        sns.set_theme(style="white",font='Times New Roman',font_scale=2.5)
        sns.heatmap(cm, annot=True, fmt='d',cmap="YlGnBu_r",linewidths=1,linecolor='black')

        plt.title('Confusion Matrix',fontdict={'family' : 'Times New Roman', 'size' : 30})
        plt.yticks(fontproperties = 'Times New Roman', size = 30)
        plt.xticks(fontproperties = 'Times New Roman', size = 30)
        plt.xlabel('Predicted Label',fontdict={'family' : 'Times New Roman', 'size' : 30})
        plt.ylabel('True Label',fontdict={'family' : 'Times New Roman', 'size' : 30})   
        plt.grid(False)
        f.savefig(os.path.join("./datasets/res_gpu/{}".format(date),'confusion_{}.png'.format(epoch)),dpi=300,bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    training()
