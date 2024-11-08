import models, torch


class Server(object):
    def __init__(self, batch_size, eval_dataset, lambdaa, model, sigma):
        self.global_model = models.get_model(model)
        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
        self.lambdaa = lambdaa
        self.sigma = sigma

    def model_aggregate(self, weight_accumulator):  # 全局模型参数聚合时添加噪声
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * self.lambdaa
            # if torch.cuda.is_available():
            #     noise = torch.cuda.FloatTensor(update_per_layer.shape).normal_(0, self.sigma)  # normal正态分布
            # else:
            #     noise = torch.FloatTensor(update_per_layer.shape).normal_(0, self.sigma)  # 生成高斯分布噪声
            # update_per_layer.add_(noise)
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    def model_eval(self):
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = self.global_model(data)  # print(output)
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l