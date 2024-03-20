import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from src.obj_parser import Mesh_obj
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import trimesh

class Penetration_data_batch1(Dataset):
    
    def __init__(self,path = "datasets/train_data/Pe_0827"):

        self.path = path

        self.label = []
        self.body_mesh = []
        self.cloth_mesh = []
        Total_frame = 850
        for idx in range(12)[:]:
            datas = np.load("{}_{}.npz".format(path,idx),allow_pickle=True)
            for i in range(50):
                self.label.append(torch.from_numpy(datas["label"][i]))
                body_path = f"./datasets/body_mesh/0000_000{(datas['index'][i][0]):03d}.obj"
                #self.body_mesh.append(torch.from_numpy(Mesh_obj(body_path).v))
                
                # 为了统一动作网格规模
                body_mesh = np.zeros(shape=(20813,3))
                real_mesh = trimesh.load_mesh(body_path).vertices
                real_numverts = real_mesh.shape[0]
                body_mesh[:real_numverts,:] = real_mesh
                if real_numverts < 20813:
                    body_mesh[real_numverts:,:] = real_mesh[0]
                self.body_mesh.append(torch.from_numpy(body_mesh))

                cloth_path = f"../DynamicNeuralGarments/datasets/train/tango/10_L/PD10_0000{(datas['index'][i][0]):03d}.obj"
                self.cloth_mesh.append(torch.from_numpy(Mesh_obj(cloth_path).v))

        self.len = len(self.label)


    def __getitem__(self,idx):
        return self.cloth_mesh[idx],self.body_mesh[idx],self.label[idx]

    def __len__(self):
        return self.len
    
class Penetration_data(Dataset):
    
    def __init__(self,path = "datasets/train_data/Pe_0827",frame_len=50,start=0):

        self.path = path
        cloth_numverts = 26718
        Total_frame = 850 # 1-700训练 700-850测试
        #frame_len = 50
        datas = np.load("{}_total.npz".format(path),allow_pickle=True) # [850,*]

        self.label = torch.zeros([frame_len*cloth_numverts,1])
        self.body_mesh = torch.zeros([frame_len*cloth_numverts,20813,3])
        self.vertex_pos = torch.zeros([frame_len*cloth_numverts,3])


        for idx in range(frame_len)[:]:
            # 0-80帧的数据先导入:80*26718
            body_path = f"./datasets/body_mesh/0000_000{(datas['index'][start+idx][0]):03d}.obj" # [start,start+frame_len]帧
            # 为了统一动作网格规模
            body_mesh = np.zeros(shape=(20813,3))
            real_mesh = trimesh.load_mesh(body_path).vertices
            real_numverts = real_mesh.shape[0]
            body_mesh[:real_numverts,:] = real_mesh
            if real_numverts < 20813:
                body_mesh[real_numverts:,:] = real_mesh[0]
            
            cloth_path = f"../DynamicNeuralGarments/datasets/train/tango/10_L/PD10_0000{(datas['index'][start+idx][1]):03d}.obj"
            cloth_pos = Mesh_obj(cloth_path).v

            self.label[cloth_numverts*idx:cloth_numverts*(idx+1),:] = torch.from_numpy(datas["label"][start+idx]).unsqueeze(-1)
            self.body_mesh[cloth_numverts*idx:cloth_numverts*(idx+1),:,:] = torch.from_numpy(body_mesh)
            self.vertex_pos[cloth_numverts*idx:cloth_numverts*(idx+1),:] = torch.from_numpy(cloth_pos)


            # for i in range(cloth_numverts):
            #     self.label.append(datas["label"][idx][i])
            #     #self.body_mesh.append(torch.from_numpy(Mesh_obj(body_path).v))
            #     self.body_mesh.append(torch.from_numpy(body_mesh))
            #     self.vertex_pos.append(torch.from_numpy(cloth_pos[i]))

        self.len = len(self.label)


    def __getitem__(self,idx):
        return self.vertex_pos[idx],self.body_mesh[idx],self.label[idx]

    def __len__(self):
        return self.len
    
if __name__ == "__main__":
    
    datas = Penetration_data()
    print(datas.len)
    