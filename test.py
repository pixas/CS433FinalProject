import torchvision.models as models 
import pprint 
resnet18 = models.resnet18(pretrained=True)
import torch 
torch.save(resnet18, "resnet18.t7")