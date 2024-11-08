import numpy as np
import random, math, copy,time
import matplotlib.pyplot as plt
from fun import fu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def GrieFunc(data):  # 目标函数
    y = fu(data[0],int(data[1]),int(data[2]),data[3])
    return y

class ABSIndividual:
    def __init__(self, bound):
        self.score = 0.
        self.invalidCount = 0  # 无效次数（成绩没有更新的累积次数）
        self.bound=bound
        self.chrom = [random.uniform(self.bound[0, 0], self.bound[1, 0]), random.randint(self.bound[0, 1], self.bound[1, 1]), random.randint(self.bound[0, 2], self.bound[1, 2]), random.uniform(self.bound[0, 3], self.bound[1, 3])]  # 随机初始化
        self.calculateFitness()

    def calculateFitness(self):
        self.score = GrieFunc(self.chrom)  # 计算当前适应度

class ArtificialBeeSwarm:
    def __init__(self, foodCount, onlookerCount, bound, maxIterCount, maxInvalidCount):
        self.foodCount = foodCount  # 蜜源个数，等同于雇佣蜂数目
        self.onlookerCount = onlookerCount  # 观察蜂个数
        self.bound = bound  # 各参数上下界
        self.maxIterCount = maxIterCount  # 迭代次数
        self.maxInvalidCount = maxInvalidCount  # 最大无效次数
        self.foodList = [ABSIndividual(self.bound) for k in range(self.foodCount)]  # 初始化各蜜源
        self.foodScore = [d.score for d in self.foodList]  # 各蜜源最佳成绩
        self.bestFood = self.foodList[np.argmax(self.foodScore)]  # 全局最佳蜜源

    def updateFood(self, i):  # 更新第i个蜜源
        j = random.choice([d for d in range(self.foodCount) if d != i])
        vi = copy.deepcopy(self.foodList[i])

        vi.chrom[0] = self.foodList[i].chrom[0] + random.uniform(-1.0, 1.0) * (self.foodList[i].chrom[0] - self.foodList[j].chrom[0])
        vi.chrom[1] = int(self.foodList[i].chrom[1] + random.uniform(-1.0, 1.0) * (self.foodList[i].chrom[1] - self.foodList[j].chrom[1]))
        vi.chrom[2] = int(self.foodList[i].chrom[2] + random.uniform(-1.0, 1.0) * (self.foodList[i].chrom[2] - self.foodList[j].chrom[2]))
        vi.chrom[3] = self.foodList[i].chrom[3] + random.uniform(-1.0, 1.0) * (self.foodList[i].chrom[3] - self.foodList[j].chrom[3])

        vi.chrom[0] = np.clip(vi.chrom[0], self.bound[0, 0], self.bound[1, 0])  # 参数不能越界
        vi.chrom[1] = int(np.clip(vi.chrom[1], self.bound[0, 1], self.bound[1, 1]))  # 参数不能越界
        vi.chrom[2] = int(np.clip(vi.chrom[2], self.bound[0, 2], self.bound[1, 2]))  # 参数不能越界
        vi.chrom[3] = np.clip(vi.chrom[3], self.bound[0,3], self.bound[1, 3])  # 参数不能越界

        vi.calculateFitness()
        if vi.score > self.foodList[i].score:  # 如果成绩比当前蜜源好
            self.foodList[i] = vi
            if vi.score > self.foodScore[i]:  # 如果成绩比历史成绩好（如重新初始化，当前成绩可能低于历史成绩）
                self.foodScore[i] = vi.score
                if vi.score > self.bestFood.score:  # 如果成绩全局最优
                    self.bestFood = vi
            self.foodList[i].invalidCount = 0
        else:
            self.foodList[i].invalidCount += 1

    def employedBeePhase(self):
        for i in range(0, self.foodCount):  # 各蜜源依次更新
            self.updateFood(i)

    def onlookerBeePhase(self):
        foodScore = [d.score for d in self.foodList]
        maxScore = np.max(foodScore)
        accuFitness = [(0.9 * d / maxScore + 0.1, k) for k, d in enumerate(foodScore)]  # 得到各蜜源的 相对分数和索引号
        for k in range(0, self.onlookerCount):
            i = random.choice([d[1] for d in accuFitness if d[0] >= random.random()])  # 随机从相对分数大于随机门限的蜜源中选择跟随
            self.updateFood(i)

    def scoutBeePhase(self):
        for i in range(0, self.foodCount):
            if self.foodList[i].invalidCount > self.maxInvalidCount:  # 如果该蜜源没有更新的次数超过指定门限，则重新初始化
                self.foodList[i] = ABSIndividual(self.bound)
                self.foodScore[i] = max(self.foodScore[i], self.foodList[i].score)
        # print(self.foodScore)

    def solve(self):
        trace = []
        trace.append((self.bestFood.score, np.mean(self.foodScore)))
        start = time.time()
        for k in range(self.maxIterCount):
            self.employedBeePhase()
            self.onlookerBeePhase()
            self.scoutBeePhase()
            trace.append((self.bestFood.score, np.mean(self.foodScore)))
        end = time.time()
        print(str(end - start))
        print(self.bestFood.score,self.bestFood.chrom)
        self.printResult(np.array(trace))

    def printResult(self, trace):
        x = np.arange(0, trace.shape[0])
        print(trace[:, 0])
        plt.plot(x, [d for d in trace[:, 0]], 'b', label='optimal value')
        plt.plot(x, [d for d in trace[:, 1]], 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value(log)")
        plt.yscale('log')
        plt.title("Artificial Bee Swarm algorithm for function optimization")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    random.seed()
    max_lr = 0.01
    min_lr = 0.00001
    max_batch_size = 256
    min_batch_size = 4
    max_CLIENT_EPOCHS = 10
    min_CLIENT_EPOCHS = 1
    max_momentum= 1
    min_momentum=0.00001
    foodCount = 5
    onlookerCount = foodCount
    maxIterCount = 5
    maxInvalidCount = 3
    bound = np.array([[min_lr,min_batch_size,min_CLIENT_EPOCHS,min_momentum], [max_lr,max_batch_size,max_CLIENT_EPOCHS,max_momentum]])  #默认np.tile(参数1是要扩充的实体, 参数2是沿着某个方向进行扩充)
    abc = ArtificialBeeSwarm(foodCount, onlookerCount, bound, maxIterCount, maxInvalidCount)
    abc.solve()

