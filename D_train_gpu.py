import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
# from src.w_models_copy import *
from src.w_models_sim import *
from src.Mytools import *
from pe_dataloader import *
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import confusion_matrix

def training_batch1():
    # 判别器训练流程
    #1.参数设置和模型准备
    #input:[1,26718,3]
    #label:[1,26718,1]
    #motion:[1,20813,3]
    cloth_numverts = 26718
    human_numverts = 20813
    data_path="datasets/train_data/Pe_0827"
    device = "cuda:2"
    batchsize = 1
    Epochs = 5 # 一个Epoch可以跑850帧,或者取其中的750帧
    D = Penetration_D(device=device).to(device)
    optimizer = torch.optim.Adam(D.parameters(),lr=0.003)

    #2.数据加载
    train_data = Penetration_data(path=data_path)
    print("Data loading finished!")
    train_loader = DataLoader(dataset=train_data,batch_size=batchsize,shuffle=True,drop_last=True,num_workers=0)

    #3.启动训练
    D.train()
    loss_list = []
    for epoch in range(Epochs):
        for _,(cloth,motion,label) in enumerate(train_loader):
            loss_ = 0.0
            pred = D(cloth.to(device),motion.to(device))
            
            #指定loss函数
            weights = torch.ones_like(label).to(device)  #[batchsize,cloth_numverts]
            weights[:,label] = 10
            weights[:,1-label] = 1 
            label = label.float().to(device)

            loss_D = nn.BCELoss(weight=weights)
            loss_ = loss_D(pred,label)

            precision = torch.sum(((pred.round())==label).int())/label.shape[-1]

            print("Epoch:{}/{},BCE_Loss:{},Precision:{:.3f}..".format(epoch+1,Epochs,loss_.item(),precision))

            loss_list.append(loss_.item())
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()

            torch.save(D.state_dict(),"D_0830_lr001.pth.tar")
            np.save('D_loss_20230830.npy',np.array(loss_list))

        plot_loss(loss_list=loss_list,lens=len(loss_list))

def training():
    # 内存占用 8441 Mb
    # 判别器训练流程
    #1.参数设置和模型准备
    #input:[batchsize,3]
    #label:[batchsize,1]
    #motion:[batchsize,20813,3]
    cloth_numverts = 26718
    human_numverts = 20813
    data_path="datasets/train_data/Pe_0827"  # 0827数据集中有更多的穿模点 -- 避免正负样本失衡

    device = "cuda:2"
    batchsize = 128 # batch:>=256会内存溢出
    Epochs = 25 # 一个Epoch取了frame_len帧: frame_len*26718/(batchsize) = 次
    frame_len = 30
    D = Penetration_D(device=device).to(device)

    # 加载预训练的模型
    date0 = "0319"
    model0 = "0319"
    root_path = "./datasets/res_gpu/{}".format(date0)
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    # D_path = "./datasets/res_gpu/{}/D_{}_gpu.pth.tar".format(model0,model0)
    # if os.path.exists(D_path):
    #     # 如果存在,就在原始模型上继续训练
    #     D.load_state_dict(torch.load(D_path)) 
    #     print("Loading Exist model finshed!")

    optimizer = torch.optim.Adam(D.parameters(),lr=0.01)

    # #2.数据加载 -- 放在epoch外
    # train_data = Penetration_data(path=data_path,frame_len=frame_len,start0)
    # train_loader = DataLoader(dataset=train_data,batch_size=batchsize,shuffle=True,drop_last=True,num_workers=0)

    #3.启动训练
    D.train()
    for t in range(5):
        date = "0319{}".format(t)
        model = "0319{}".format(t)
        save_path = "./datasets/res_gpu/{}".format(date)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 训练全程的损失函数曲线
        loss_list = []
        Accuracy_list = []
        Recall_list,FPR_list = [],[]
        threshold_list = []

        for epoch in range(Epochs):
            # 放在epoch里
            train_data = Penetration_data(path=data_path,frame_len=frame_len,start=epoch*frame_len)
            train_loader = DataLoader(dataset=train_data,batch_size=batchsize,shuffle=True,drop_last=True,num_workers=0)
           
            # Accuracy_list = []
            # Recall_list,FPR_list = [],[]
            loss_total = 0.0
            temp = []
            label_total = np.array([])
            pred_total = np.array([])

            iters = 0 # 记录迭代次数

            for _,(v_pos,motion,label) in enumerate(train_loader):
                
                iters += 1
                loss_ = 0.0
                pred = D(v_pos.to(device),motion.to(device))
                
                #指定loss函数
                weights = torch.ones_like(label).to(device)  #[batchsize,1]
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
                # 每个迭代内只看BCE_loss，然后记录每次迭代的预测值和标签值,一个epoch更新一次
                label_total = np.hstack((label_total,label[:,0].cpu().detach().numpy()))
                pred_total  = np.hstack((pred_total,pred[:,0].cpu().detach().numpy()))
            
                print("Epoch:{}/{},BCE_Loss:{:.3f}..".format(epoch+1,Epochs,loss_.item()))

            
            # 每个Epoch绘制ROC曲线并寻找最佳阈值，根据最佳阈值计算准确率和召回率
            if label_total.size != 0 :
                fpr_list, tpr_list, thresholds = metrics.roc_curve(label_total,pred_total,pos_label=1)
                best_threshold = thresholds[np.argmax(tpr_list - fpr_list)] # 寻找最佳阈值
                threshold_list.append(best_threshold.item())
                #print(best_threshold)
                # 计算auc面积值
                roc_auc = metrics.auc(fpr_list, tpr_list)
                plot_func(date=date,title="ROC CURVE",plot_x=fpr_list,plot_y=tpr_list,lens=iters,id=epoch,mode="Roc",auc=roc_auc)
            
            
            # 根据最佳阈值计算准确率和召回率
            predict_mine = np.where(pred_total >= best_threshold, 1, 0)
            metrices = Classification_indicator(predict_mine,label_total)  # pred预测是概率值,四舍五入转换为类别
            accuracy = metrices.Accuracy()
            Accuracy_list.append(accuracy.item())
            loss_list.append(loss_total/iters)

            if (label_total==1).any() :

                # label中包含正样本时:计算召回率
                recall = metrices.Recall()
                Recall_list.append(recall.item())
                # fpr = metrices.FPR()
                # FPR_list.append(fpr.item())

                # 显示单个Epoch的平均loss和准确率、召回率
                print("Epoch:{}/{},BCE_Loss:{:.3f},Recall:{:.3f},Accuracy:{:.3f}..".format(epoch+1,Epochs,loss_total/iters,recall.item(),accuracy.item()))
            else:
                print("Epoch:{}/{},BCE_Loss:{:.3f},Accuracy:{:.3f}..".format(epoch+1,Epochs,loss_total/iters,accuracy.item()))

            plot_confusion(np.array(predict_mine),np.array(label_total),date=date,epoch=epoch)
            
            # 每个Epoch保存预训练的模型和到目前为止的指标情况，并绘制收敛曲线
            torch.save(D.state_dict(),"./datasets/res_gpu/{}/D_{}_gpu.pth.tar".format(date,date))
            
            # 绘制最佳阈值曲线
            np.save('./datasets/res_gpu/{}/yuzhi_gpu_{}.npy'.format(date,epoch),np.array(threshold_list))
            plot_func(date=date,title="Best threshold",plot_x=range(len(threshold_list)),plot_y=threshold_list,lens=len(threshold_list),id=epoch)

            np.save('./datasets/res_gpu/{}/BCE_gpu_{}.npy'.format(date,epoch),np.array(temp))
            # np.save('./datasets/res_gpu/{}/Accuracy_gpu_{}.npy'.format(date,epoch),np.array(Accuracy_list))
            # np.save('./datasets/res_gpu/{}/Recall_gpu_{}.npy'.format(date,epoch),np.array(Recall_list))
            plot_func(date=date,title="BCE_Loss",plot_x=range(len(loss_list)),plot_y=loss_list,lens=len(loss_list),id=epoch)
            # plot_func(date=date,title="Accuracy",plot_x=range(len(Accuracy_list)),plot_y=Accuracy_list,lens=len(Accuracy_list),id=epoch)
            # plot_func(date=date,title="Recall",plot_x=range(len(Recall_list)),plot_y=Recall_list,lens=len(Accuracy_list),id=epoch)
            

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
        #plt.rcParams['axes.facecolor'] = 'white'
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

        # plt.title("{} after {} training...".format(title,train_num*(1+id)),fontdict={'family' : 'Times New Roman', 'size' : 30})
        plt.title("{}".format(title),fontdict={'family' : 'Times New Roman', 'size' : 30})

        f.savefig(os.path.join("./datasets/res_gpu/{}".format(date),'{}_gpu_{}.png'.format(title,id)),dpi=300,bbox_inches='tight')
        plt.close()

def plot_confusion(y_pred, y_true,date=0,epoch=0):

        cm = confusion_matrix(y_true, y_pred)
        f = plt.figure(figsize=(12,8))
        sns.set_theme(style="white",font='Times New Roman',font_scale=2.5)
        sns.heatmap(cm, annot=True, fmt='d',cmap="YlGnBu_r",linewidths=1,linecolor='black')
        #sns.heatmap(cm, annot=True, fmt='d')
        # 设置图形标题和坐标轴标签
        plt.title('Confusion Matrix',fontdict={'family' : 'Times New Roman', 'size' : 30})
        plt.yticks(fontproperties = 'Times New Roman', size = 30)
        plt.xticks(fontproperties = 'Times New Roman', size = 30)
        plt.xlabel('Predicted Label',fontdict={'family' : 'Times New Roman', 'size' : 30})
        plt.ylabel('True Label',fontdict={'family' : 'Times New Roman', 'size' : 30})   
        # plt.legend(prop={'family' : 'Times New Roman', 'size' : 30})
        plt.grid(False)
        f.savefig(os.path.join("./datasets/res_gpu/{}".format(date),'confusion_{}.png'.format(epoch)),dpi=300,bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    training()