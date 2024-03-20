# 检测布料网格和人体网格是否发生穿模,作为判别器的标签
import trimesh
import numpy as np
from src.obj_parser import Mesh_obj
from tqdm import tqdm
import sys

# Pe0828_total.npz数据集: 
# -- 人体1-850帧与布料851-2分别对应进行穿模检测后生成布料每个顶点的穿模情况 [i+1,851-i] i=0,..,849
# -- 之所有这么生成主要是希望能够增加正样本(穿模样本的比例,避免正负样本失衡的问题),但发现好像帧数间隔过大导致布料位移过大,从而都是负样本
# -- 还是选择使用0827的数据集

def generate_batch_label():
    # region 网格数据加载
    # 数据说明：
    #    == body_mesh:人体动作序列,[850帧,body_numverts,3]
    #    == cloth_mesh:布料序列,[851帧,cloth_numverts,3] -- 851帧第一帧是服装初始位置,后续2-851与动作帧对应
    total_frames = 850
    for idx in range(17)[2:]:
        #print(idx)
        contain_res = []
        mesh_index = []
        # 间隔50分开生成label,共计17次可以完成,9h差不多
        #for i in tqdm(range(total_frames)[50*(idx):50*(idx+1)]):
        for i in range(total_frames)[50*(idx):50*(idx+1)]:

            human_path = f"./datasets/body_mesh/0000_000{(i+1):03d}.obj" # 格式化输出:{var:format} var表示变量,format是指定变量输出的格式
            cloth_path = f"../DynamicNeuralGarments/datasets/train/tango/10_L/PD10_0000{(851-i):03d}.obj" #851-i

            human_mesh = trimesh.load_mesh(human_path)
            cloth_mesh = Mesh_obj(cloth_path)
            # print(cloth_mesh.v.shape)
            #cloth_mesh = trimesh.load_mesh("./datasets/cloth_mesh/PD10_0000001.obj",force='mesh') # 服装obj有多层,trimesh读取成场景格式,比较麻烦,弃用

            # region 穿模检测
            #    == trimesh.base.Trimesh.contain 方法
            # 检测布料网格顶点是否在人体网格顶点内部,支持批量顶点输入
            # 1. contains方法使用射线检测方法来确定布料网格每个顶点是否在人体内部,返回结果:长度=布料顶点个数的numpy数组,元素是false/true,对应不包含/包含状态
            # 容易出现内存溢出导致程序终止,使用try-except保证程序运行
            try:
                # 可能出错的语句 
                contains = human_mesh.contains(cloth_mesh.v) #[cloth_numverts,]
                # print(contains.any())
            except:
                # 出错后执行
                pass
            else:
                # 未出错执行
                # 2. 记录结果和对应动作、布料索引
                contain_res.append(contains.astype(int))
                mesh_index.append([i+1,851-i])  # 这里没改过来QAQ:应该是[i+1,i+1] -- 记得要修改数据集！#851-i
                #print(i)
            
            # endregion
        # print(contain_res,mesh_index)
        save_path = "./datasets/train_data/Pe_0828_{}.npz".format(idx)
        np.savez(save_path,label=np.array(contain_res),index=np.array(mesh_index))
    # endregion

    # 损失函数可以使用加权的交叉熵 -- 更加关注穿模的顶点是否被检测到！

    # trimesh 计算任意点与网格之间的最短距离
    # dis_res = trimesh.proximity.closest_point(human_mesh,cloth_mesh.vertices) 

def generate_label():

    # 将分批次生成的标签npz文件合并为一个大文件
    total_label = np.load("./datasets/train_data/Pe_0828_0.npz",allow_pickle=True)["label"]
    total_index = np.load("./datasets/train_data/Pe_0828_0.npz",allow_pickle=True)["index"]
    for i in range(16)[:]:
        datas = np.load("./datasets/train_data/Pe_0828_{}.npz".format(i+1),allow_pickle=True)
        # datas["index"][:,1] = datas["index"][:,0] # 这里是我生成0827数据标签的时候索引写错了，这里加一步矫正,正常情况下不需要此行代码
        total_label = np.vstack([total_label,datas["label"]])    
        total_index = np.vstack([total_index,datas["index"]])
    # print(total_label.shape)
    # print(total_index.shape)
    save_path = "./datasets/train_data/Pe_0828_total.npz"
    np.savez(save_path,label=np.array(total_label),index=np.array(total_index))
    
if __name__ == "__main__":
    generate_label()

# conda 配置属性 -- 没啥用
# # >>> conda initialize >>>
# # !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/project1/wangwenbin/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
# eval "$__conda_setup"
# else
# if [ -f "/project1/wangwenbin/miniconda3/etc/profile.d/conda.sh" ]; then
# . "/project1/wangwenbin/miniconda3/etc/profile.d/conda.sh"
# else
# export PATH="/project1/wangwenbin/miniconda3/bin:$PATH"
# fi
# fi
# unset __conda_setup
#  #<<< conda initialize <<</project1/wangwenbin/miniconda3/bin
