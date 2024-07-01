import torch

# 假设 tensor 是你的张量
tensor = torch.tensor([1, 2, 3, 1, 2, 4, 5, 6, 3, 7, 8, 1])

# 使用 torch.unique 获取张量中的唯一值和它们的出现次数
unique_values, counts = torch.unique(tensor, return_counts=True)

# 输出唯一值和它们的出现次数
print("Unique values:", unique_values)
print("Counts:", counts)
