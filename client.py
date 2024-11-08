import models, torch, copy


class Client(object):

    def __init__(self, CC, lr, momentum, local_epochs, no_models, batch_size, model, train_dataset, id=-1):
        self.lr = lr
        self.CC = CC
        self.momentum = momentum
        self.local_epochs = local_epochs
        self.local_model = models.get_model(model)  # 客户端本地模型
        self.client_id = id  # 客户端id
        self.train_dataset = train_dataset  # 客户端本地数据集
        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / no_models)
        train_indices = all_range[id * data_len: (id + 1) * data_len]
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size,
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            train_indices), drop_last=True)

    def local_train(self, model):
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.lr, momentum=self.momentum)
        self.local_model.train()
        for e in range(self.local_epochs):
            for batch_id, batch in enumerate(self.train_loader):  # 枚举，对于一个可迭代的对象，enumerate将它变成一个索引序列
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0
                output = self.local_model(data)  # 计算模型输出
                loss = torch.nn.functional.cross_entropy(output, target)  # 计算损失函数
                loss.backward()  # 通过反向传播过程来实现可训练参数的更新,反向传播求参数梯度
                optimizer.step()  # step()函数的作用是执行一次优化步骤，通过梯度下降法来更新模型参数的值。
                # model_norm = models.model_norm(model, self.local_model)
                # norm_scale = min(1, self.CC / (model_norm))
                # for name, layer in self.local_model.named_parameters():
                #     clipped_difference = norm_scale * (layer.data - model.state_dict()[name])
                #     layer.data.copy_(model.state_dict()[name] + clipped_difference)
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])
        return diff
