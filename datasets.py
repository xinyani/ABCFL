import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import json
import os

class MyDataset(Dataset):
	def __init__(self, datafile, mode='train'):
		with open(datafile, 'r') as f:
			self.data = json.load(f)[mode]
		
	def __getitem__(self, index):
		return (torch.tensor(self.data[index]['data'], dtype=torch.float).unsqueeze(1), torch.tensor(self.data[index]['label'], dtype=torch.long))
	
	def __len__(self):
		return len(self.data)


def get_dataset(dir, name, filename):

	if name=='mnist':
		train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
		eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())
		
	elif name=='cifar':
		transform_train = transforms.Compose([  #串联多个transform
			transforms.RandomCrop(32, padding=4),  #随即裁剪
			transforms.RandomHorizontalFlip(),  #以给定的概率随机水平翻转给定的PIL图像，默认值为0.5
			transforms.ToTensor(),  #将图片转换成Tensor
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),#将图片转换成Tensor
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		
		train_dataset = datasets.CIFAR10(dir, train=True, download=False, transform=transform_train)
		eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
	elif name=="custom_1D":
		train_dataset = MyDataset(os.path.join(dir, filename), mode='train_data')
		eval_dataset = MyDataset(os.path.join(dir, filename), mode='val_data')
	return train_dataset, eval_dataset