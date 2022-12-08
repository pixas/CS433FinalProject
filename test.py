import torchvision.models as models 
import pprint 
resnet18 = models.resnet18(pretrained=True)
import torch 
torch.save(resnet18.state_dict(), "resnet18.t7")