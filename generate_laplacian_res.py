from torch_geometric.utils import get_laplacian,to_dense_adj
import numpy as np
from src.obj_parser import Mesh_obj
import torch
from torch.utils.data import Dataset,DataLoader
from torch_geometric.transforms import FaceToEdge
from torch_geometric.data import Data
import os
from tqdm import tqdm
import torch_geometric
from Mytools import *
import copy

# 将原始网格使用laplacian平滑
HF_path = "VirtualBoneDataset/dress02/HF_res"
LF_path = "VirtualBoneDataset/dress02/LF_res"
if not os.path.exists(LF_path):
    os.makedirs(LF_path)

cloth = Mesh_obj("assets/dress02/garment.obj")
data = FaceToEdge()(Data(num_nodes=cloth.v.shape[0],face=torch.from_numpy(cloth.f.astype(int).transpose() - 1).long()))
smoother = Laplacian_Smooth()

def laplacian_compeat(pos,edge_index):

    pos_list = smoother.laplacian_smooth(torch.from_numpy(pos),edge_index)
    return pos_list[9],pos_list[-1]

for file in os.listdir(HF_path)[:]:

    if "npz" in file :
        init_pos = np.load(os.path.join(HF_path,file),allow_pickle=True)["final_ground"]
        lap_pos = np.zeros_like(init_pos)
        dpos = np.zeros_like(init_pos)
        for idx in range(init_pos.shape[0])[:]:
            lap_pos[idx],dpos[idx] = laplacian_compeat(copy.deepcopy(init_pos[idx]),data.edge_index)
        np.savez(os.path.join(LF_path,file),lap_pos = lap_pos,dpos = dpos)
        





