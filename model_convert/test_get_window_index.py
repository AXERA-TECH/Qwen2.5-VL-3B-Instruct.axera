import torch
from utils import get_window_index

grid_thw = torch.tensor([[1,24,24]])

window_index1, cu_window_seqlens1 = get_window_index(grid_thw)

# print("window_index",window_index)
# print("cu_window_seqlens",cu_window_seqlens)


grid_thw = torch.tensor([[2,24,24]])

window_index2, cu_window_seqlens2 = get_window_index(grid_thw)

# print("window_index",window_index)
# print("cu_window_seqlens",cu_window_seqlens)


grid_thw = torch.tensor([[3,24,24]])

window_index3, cu_window_seqlens3 = get_window_index(grid_thw)

# print("window_index",window_index)
# print("cu_window_seqlens",cu_window_seqlens)



grid_thw = torch.tensor([[4,24,24]])

window_index4, cu_window_seqlens4 = get_window_index(grid_thw)

# print("window_index",window_index)
# print("cu_window_seqlens",cu_window_seqlens)
print("end")