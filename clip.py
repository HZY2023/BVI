import torch
import timm
model_resnet = timm.list_models('*clip*')
print(model_resnet)