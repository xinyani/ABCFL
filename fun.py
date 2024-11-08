import argparse, json, time   #argparse 模块可以让人轻松编写用户友好的命令行接口
import datetime
import os
import logging
import torch, random
from server import *
from client import *
import models, datasets
import ssl
import matplotlib.pyplot as plt
ssl._create_default_https_context = ssl._create_unverified_context
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def fu(lr, batch_size, local_epochs,momentum):
	time_start=time.time()
	global_epochs=50

	print(lr, batch_size, local_epochs,momentum)
	no_models = 10
	k=2
	CC=1
	sigma=0.01
	lambdaa = 1/k
	name1 = "resnet1D"

	train_datasets, eval_datasets = datasets.get_dataset( "data","custom_1D",'x.json')
	server = Server(batch_size, eval_datasets, lambdaa, name1, sigma)
	clients = []
	#创建多个客户端
	for c in range(no_models):
		clients.append(Client(CC,lr, momentum, local_epochs, no_models, batch_size, name1, train_datasets, c))
	AA = []
	bb = []
	for e in range(global_epochs):
		candidates = random.sample(clients, k)
		weight_accumulator = {}
		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)
		for c in candidates:
			diff = c.local_train(server.global_model)
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name])

		server.model_aggregate(weight_accumulator)
		acc, loss = server.model_eval()
		AA.append(acc)
		bb.append(loss)
	time_end = time.time()
	print(AA,bb)
	print('totally cost', time_end - time_start)
	file = open('abcc.txt', mode='a')
	file.write(str(lr))
	file.write('\n')
	file.write(str(batch_size))
	file.write('\n')
	file.write(str(local_epochs))
	file.write('\n')
	file.write(str(momentum))
	file.write('\n')
	file.write(str(AA))
	file.write('\n')
	file.write(str(bb))
	file.write('\n')
	return max(AA)
	