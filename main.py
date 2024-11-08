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
# 全局取消证书验证
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
	time_start = time.time()
	parser = argparse.ArgumentParser(description='Federated Learning')  #ArgumentParser 建立解析对象
	parser.add_argument('-c', '--conf', dest='conf', default='.conf.json')          #增加属性
	args = parser.parse_args()
	with open(r'conf.json') as f:
		conf = json.load(f)
	train_datasets, eval_datasets = datasets.get_dataset("../data/", conf["type"], conf["filename"])
	server = Server(conf, eval_datasets)
	clients = []
	for c in range(conf["no_models"]):
		clients.append(Client(conf, server.global_model, train_datasets, c))
	print("\n\n")
	AA=[]
	bb=[]
	for e in range(conf["global_epochs"]):
		candidates = random.sample(clients, conf["k"])
		weight_accumulator = {}
		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)
		for c in candidates:
			diff = c.local_train(server.global_model)
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name])

		server.model_aggregate(weight_accumulator)
		acc, loss = server.model_eval()
		print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
		AA.append(acc)
		bb.append(loss)
	time_end = time.time()
	print(AA,bb)
	print('totally cost', time_end - time_start)
	