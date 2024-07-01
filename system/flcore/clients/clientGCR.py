import torch
import torch.nn as nn
import numpy as np
import time
import copy
from collections import defaultdict
from flcore.clients.clientbase import Client
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn import metrics

from torch.autograd import Variable
from utils.data_utils import read_client_data


class clientGCR(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.lamda = args.lamda
        self.protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()

        trainloader = self.load_train_data()
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                rep = self.model.base(x).detach()
            break
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=self.device)
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        X_train, y_train = zip(*train_data)
        self.dual = True if len(X_train) >250 else False 
        print('-----'+str(self.id)+'---------'+str(len(X_train)) +'---------'+str(self.dual) +'---------')
        # y_train = torch.stack(y_train).type(torch.int64)
        # self.extra_param = design_parameter(y_train,args.num_classes)
        self.extra_param = 0
        self.client_mean = nn.Linear(rep.shape[1], args.num_classes, bias=True).to(self.device)
        self.opt_client_mean = torch.optim.SGD(list(self.client_mean.parameters()), lr=self.learning_rate)

    def train(self):
        if self.dual:
            trainloader = self.load_train_data()
            # self.model.to(self.device)
            self.model.train()

            start_time = time.time()

            max_local_steps = self.local_epochs
            if self.train_slow:
                max_local_steps = np.random.randint(1, max_local_steps // 2)

            self.reset_running_stats()
            protos = defaultdict(list)
            for step in range(max_local_steps):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))

                    # ====== begin
                    rep = self.model.base(x)

                    if self.num_batches_tracked is not None:
                        self.num_batches_tracked.add_(1)

                    if self.global_protos is not None:
                        output_client = self.client_mean(rep)
                        output = (1 - self.extra_param) * self.model.head(rep) + self.extra_param * output_client
                        loss = self.loss(output, y)
                        proto_new = copy.deepcopy(rep.detach())
                        for i, yy in enumerate(y):
                            y_c = yy.item()
                            if type(self.global_protos[y_c]) != type([]):
                                proto_new[i, :] = self.global_protos[y_c].data
                        loss += self.loss_mse(proto_new, rep) * self.lamda
                    else:
                        output = self.model.head(rep)
                        loss = self.loss(output, y)

                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        protos[y_c].append(rep[i, :].detach().data)
                    # ====== end

                    self.opt_client_mean.zero_grad()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.opt_client_mean.step()
                    # self.detach_running()

            # self.model.cpu()
            # print("extra_param的值: {}".format(self.extra_param))
            self.protos = agg_func(protos)
            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()
        else:
            trainloader = self.load_train_data()
            # self.model.to(self.device)
            self.model.train()

            start_time = time.time()

            max_local_steps = self.local_epochs
            if self.train_slow:
                max_local_steps = np.random.randint(1, max_local_steps // 2)

            self.reset_running_stats()
            protos = defaultdict(list)
            for step in range(max_local_steps):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))

                    # ====== begin
                    rep = self.model.base(x)

                    if self.num_batches_tracked is not None:
                        self.num_batches_tracked.add_(1)

                    if self.global_protos is not None:
                        output = self.model.head(rep)
                        loss = self.loss(output, y)
                        proto_new = copy.deepcopy(rep.detach())
                        for i, yy in enumerate(y):
                            y_c = yy.item()
                            if type(self.global_protos[y_c]) != type([]):
                                proto_new[i, :] = self.global_protos[y_c].data
                        loss += self.loss_mse(proto_new, rep) * self.lamda
                    else:
                        output = self.model.head(rep)
                        loss = self.loss(output, y)

                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        protos[y_c].append(rep[i, :].detach().data)
                    # ====== end

                    loss.backward()
                    self.optimizer.step()
                    self.opt_client_mean.step()
                    # self.detach_running()

            # self.model.cpu()
            # print("extra_param的值: {}".format(self.extra_param))
            self.protos = agg_func(protos)
            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def reset_running_stats(self):
        # self.running_mean.zero_()
        self.num_batches_tracked.zero_()

    def set_protos(self, global_protos):
        self.global_protos = global_protos

    def detach_running(self):
        self.running_mean.detach_()

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                if self.dual:
                    output_client = self.client_mean(rep)
                    output = (1 - self.extra_param) * self.model.head(rep) + self.extra_param * output_client
                else:
                    output = self.model.head(rep)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        reps = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                if self.dual:
                    output_client = self.client_mean(rep)
                    output = (1 - self.extra_param) * self.model.head(rep) + self.extra_param * output_client
                else:
                    output = self.model.head(rep)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)
                reps.extend(rep.detach())

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos


def calculate_data_complexity(client_data):
    # 统计每个类别的样本数量
    class_counts = torch.bincount(client_data)

    # 获取存在的类别索引
    existing_classes = torch.nonzero(class_counts).squeeze()

    existing_class_counts = class_counts[existing_classes].float()

    # 计算概率分布
    probabilities = existing_class_counts.float() / torch.sum(existing_class_counts)

    # 计算熵
    entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-10))

    return entropy,existing_class_counts

def design_parameter(data_distribution, max_classes):
    # 计算数据分布的熵
    entropy,class_counts = calculate_data_complexity(data_distribution)

    # 计算最大熵（使用每个类别出现的最大次数）
    max_entropy = -torch.log2(torch.tensor(1.0 / class_counts.shape[0]))

    # 设计个性化程度参数 alpha
    # alpha = (1 - entropy / max_entropy) * (1 - class_counts.shape[0] / max_classes)
    alpha = 0.5 * (1 - entropy / max_entropy)

    # 将 alpha 限制在 [0, 0.5] 范围内
    alpha = 0.3 - torch.clamp(alpha, 0.1, 0.5)

    return alpha.item()
