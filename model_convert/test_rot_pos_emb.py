import torch
from utils import rot_pos_emb, rot_pos_id

grid_thw = torch.tensor([[1,24,24]])

emb1 = rot_pos_emb(grid_thw)


grid_thw = torch.tensor([[2,24,24]])

emb2 = rot_pos_emb(grid_thw)


grid_thw = torch.tensor([[3,24,24]])

emb3 = rot_pos_emb(grid_thw)


grid_thw = torch.tensor([[4,24,24]])

emb4 = rot_pos_emb(grid_thw)



pos_id = rot_pos_id(grid_thw)

print("end")