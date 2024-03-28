# 检测布料网格和人体网格是否发生穿模,作为判别器的标签
import trimesh
import numpy as np
from src.obj_parser import Mesh_obj
from tqdm import tqdm
import sys

def generate_batch_label():
    total_frames = 850
    for idx in range(17)[2:]:
        contain_res = []
        mesh_index = []

        for i in range(total_frames)[50*(idx):50*(idx+1)]:
            human_path = f"./datasets/body_mesh/0000_000{(i+1):03d}.obj" 
            cloth_path = f"../DynamicNeuralGarments/datasets/train/tango/10_L/PD10_0000{(851-i):03d}.obj" 

            human_mesh = trimesh.load_mesh(human_path)
            cloth_mesh = Mesh_obj(cloth_path)

            try:
                contains = human_mesh.contains(cloth_mesh.v) #[cloth_numverts,]
            except:
                pass
            else:
                contain_res.append(contains.astype(int))
                mesh_index.append([i+1,851-i])  

        save_path = "./datasets/train_data/Pe_0828_{}.npz".format(idx)
        np.savez(save_path,label=np.array(contain_res),index=np.array(mesh_index))

def generate_label():
    # 将分批次生成的标签npz文件合并为一个大文件
    total_label = np.load("./datasets/train_data/Pe_0828_0.npz",allow_pickle=True)["label"]
    total_index = np.load("./datasets/train_data/Pe_0828_0.npz",allow_pickle=True)["index"]
    for i in range(16)[:]:
        datas = np.load("./datasets/train_data/Pe_0828_{}.npz".format(i+1),allow_pickle=True)
        total_label = np.vstack([total_label,datas["label"]])    
        total_index = np.vstack([total_index,datas["index"]])

    save_path = "./datasets/train_data/Pe_0828_total.npz"
    np.savez(save_path,label=np.array(total_label),index=np.array(total_index))
    
if __name__ == "__main__":
    generate_label()

