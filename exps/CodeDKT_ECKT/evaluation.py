import tqdm
import torch
import logging
import seaborn as sns
import numpy as np
import matplotlib.pylab as plt
import torch.nn as nn
from sklearn import metrics

def performance_granular(ground_truth, prediction):
    f1 = metrics.f1_score(ground_truth.detach().numpy(),
                          torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    recall = metrics.recall_score(ground_truth.detach().numpy(),
                                  torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    precision = metrics.precision_score(
        ground_truth.detach().numpy(),
        torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    acc = metrics.accuracy_score(
        ground_truth.detach().numpy(),
        torch.where(prediction>0.5,torch.tensor(1), torch.tensor(0)).detach().numpy())
    auc = metrics.roc_auc_score(
        ground_truth.detach().numpy(),
        prediction.detach().numpy())
    return auc,f1,recall,precision,acc

class lossFunc(nn.Module):
    def __init__(self, num_of_questions, max_step, device):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.num_of_questions = num_of_questions
        self.max_step = max_step
        self.device = device

    def forward(self, pred, batch):
        loss = 0
        prediction = torch.tensor([])
        ground_truth = torch.tensor([])
        pred = pred.to('cpu')

        for student in range(pred.shape[0]):
            delta = (batch[student][:, 0:self.num_of_questions] +
                     batch[student][:, self.num_of_questions:self.num_of_questions*2])  # shape: [length, questions]
            temp = pred[student][:self.max_step-1].mm(delta[1:].t())
            index = torch.tensor([[i for i in range(self.max_step-1)]],
                                 dtype=torch.long)
            p = temp.gather(0, index)[0]
            a = (((batch[student][:, 0:self.num_of_questions] -
                   batch[student][:, self.num_of_questions:self.num_of_questions*2]).sum(1) + 1) // 2)[1:]
            
            for i in range(len(p)):
                if p[i] > 0:
                    p = p[i:]
                    a = a[i:]
                    break

            loss += self.crossEntropy(p, a)
            prediction = torch.cat([prediction, p])
            ground_truth = torch.cat([ground_truth, a])

        return loss, prediction, ground_truth

def train_epoch(model, trainLoader, optimizer, loss_func, n_problems, device):
    model.to(device)
    model.train()

    total_loss = 0
    for batch in trainLoader:
        batch_new = batch[:,:-1,:].to(device)
        pred = model(batch_new)
        loss, prediction, ground_truth = loss_func(pred, batch[:,:,:n_problems*2])

        total_loss += float(loss.cpu().detach().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    mean_loss = total_loss/ len(trainLoader)
    return mean_loss

def evaluate(model, testLoader, loss_func, device):
    model.to(device)
    ground_truth = torch.tensor([])
    prediction = torch.tensor([])
    full_data = torch.tensor([])
    preds = torch.tensor([])
    batch_n = 0

    model.eval()
    with torch.no_grad():
        for batch in testLoader:
            batch_new = batch[:,:-1,:].to(device)
            pred = model(batch_new)
            loss, p, a = loss_func(pred, batch)
            
            prediction = torch.cat([prediction, p])
            ground_truth = torch.cat([ground_truth, a])
            full_data = torch.cat([full_data, batch])
            preds = torch.cat([preds, pred.cpu()])
            # plot_heatmap(batch, pred, fold, batch_n, n_problems)
            batch_n += 1

    return performance_granular(ground_truth, prediction)
