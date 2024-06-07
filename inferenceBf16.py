import torch
import torchvision.models as models
from custom_modules import timing

timing.calculateTime()

############# code changes ###############
import intel_extension_for_pytorch as ipex

############# code changes ###############

model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()
data = torch.rand(1, 3, 224, 224)

#################### code changes #################
model = model.to("xpu")
data = data.to("xpu")
model = ipex.optimize(model, dtype=torch.bfloat16)
#################### code changes #################

with torch.no_grad():
    ############################# code changes #####################
    with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    ############################ code changes ######################
        model(data)

print("Execution finished")