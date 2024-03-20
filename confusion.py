import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import palettable
import os
# 模拟分类结果
y_true = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
temp = np.array([["TP",'FN'],['FP','TN']])
# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)
f = plt.figure(figsize=(12,8))
# 绘制混淆矩阵热图
sns.heatmap(cm, annot=False, fmt='d', cmap="YlGnBu",linewidths=1,linecolor='black')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j + 0.5, i + 0.5, temp[i][j], ha='center', va='center', color='black',fontdict={'family' : 'Times New Roman', 'size' : 30})

plt.title('Confusion Matrix',fontdict={'family' : 'Times New Roman', 'size' : 30})
plt.yticks(fontproperties = 'Times New Roman', size = 30)
plt.xticks(fontproperties = 'Times New Roman', size = 30)
plt.xlabel('Predicted Label',fontdict={'family' : 'Times New Roman', 'size' : 30})
plt.ylabel('True Label',fontdict={'family' : 'Times New Roman', 'size' : 30})   

f.savefig(os.path.join("./confusion.png"),dpi=300)
# 显示图形
#plt.show()

