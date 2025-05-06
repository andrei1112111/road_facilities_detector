import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from segmentation_models_pytorch import DeepLabV3Plus

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_classify_model(weight_path):
	model = models.mobilenet_v3_large()  
	num_ftrs = model.classifier[-1].in_features
	model.classifier[-1] = nn.Linear(num_ftrs, 1)  
	model.classifier.add_module('sigmoid', nn.Sigmoid())
	model.to(device=device)
	model.load_state_dict(torch.load(weight_path, map_location=device))
	model.eval()

	return model

def create_segment_model(weight_path):
	model = DeepLabV3Plus(encoder_name='resnet101',classes=25) 
	model.to(device)
	model.load_state_dict(torch.load(weight_path,map_location=device))
	model.eval()

	return model

	
