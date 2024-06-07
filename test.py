import torch
import intel_extension_for_pytorch as ipex

tensor = torch.tensor([[1, 2], [3, 4]])
print(tensor, tensor.device)

tensor_ipex = tensor.to('xpu')
print(tensor, tensor_ipex.device)
