import torch
from torchvision import models
import math
from resnet1D import ResNet1D

def get_model(name="vgg16", pretrained=True, **kwargs):
	if name == "resnet18":      #残差卷积网络
		model = models.resnet18(pretrained=pretrained)
		model.conv1 = torch.nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
		num_ftrs = model.fc.in_features
		model.fc = torch.nn.Linear(num_ftrs, 2)
	# print(model)
	elif name == "resnet50":
		model = models.resnet50(pretrained=pretrained)	
	elif name == "densenet121":  #密集连接卷积网络
		model = models.densenet121(pretrained=pretrained)
	elif name == "googlenet":		
		model = models.googlenet(pretrained=pretrained)
	elif name == "resnet1D":
		in_channel = kwargs.get("in_channel", 55)
		num_class = kwargs.get("num_class", 5)
		model = ResNet1D(in_channel, num_class)
		
	if torch.cuda.is_available():
		return model.cuda()
	else:
		return model 
		
def model_norm(model_1, model_2):
	squared_sum = 0
	for name, layer in model_1.named_parameters():   #计算两个模型对应参数的欧氏距离
	#	print(torch.mean(layer.data), torch.mean(model_2.state_dict()[name].data))
		squared_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
	return math.sqrt(squared_sum)