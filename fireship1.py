import torch

data = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]

x_data = torch.tensor(data)

x_rand = torch.rand_like(x_data, dtype=torch.float)

tensor1 = torch.randn(3, 4)
tensor2 = torch.randn(4)

result = torch.matmul(tensor1, tensor2)

print(x_data)
print(x_rand)
print(result)