import numpy as np
import scipy.sparse as sp
import torch
import dgl
import os
import random
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

random.seed(65)
np.random.seed(65)
torch.manual_seed(65)

def load_features(ID,data_path):
    pssm_feature = np.load(data_path + 'esm_aa_float32/' + ID + '.npy').astype(np.float32) # 23447
    return pssm_feature    
def load_graph(ID, data_path):
    matrix = np.load(data_path + 'edges_unordered/' + ID + '.npy').astype(np.int32)
    return matrix
import os
file_dir_1 = '/data/gcn/edges_unordered'
file_names_1 = os.listdir(file_dir_1)
pro_graph = [i[:-4] for i in file_names_1]
file_dir_2 = '/data/esm/esm_aa_float32'
file_names_2 = os.listdir(file_dir_2)
pro_esm = [i[:-4] for i in file_names_2]
pro_common = set(pro_esm)&set(pro_graph)

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

import scipy.sparse as sp
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_gcndata(protein1,protein2):
    idx_features1 = load_features(protein1,'/data/esm/')
    idx_features2 = load_features(protein2,'/data/esm/')
    
    feature1 = sp.csr_matrix(idx_features1[1:-1], dtype=np.float32) 
    feature2 = sp.csr_matrix(idx_features2[1:-1], dtype=np.float32) 
    
    edges1 = load_graph(protein1,'/data/gcn/')
    edges2 = load_graph(protein2,'/data/gcn/')
    
    adj1 = sp.coo_matrix((np.ones(edges1.shape[0]), (edges1[:, 0], edges1[:, 1])),
                         shape=(idx_features1.shape[0]-2, idx_features1.shape[0]-2), dtype=np.float32)
    adj2 = sp.coo_matrix((np.ones(edges2.shape[0]), (edges2[:, 0], edges2[:, 1])),
                         shape=(idx_features2.shape[0]-2, idx_features2.shape[0]-2), dtype=np.float32)
    
    adj1 = adj1 + adj1.T.multiply(adj1.T > adj1) - adj1.multiply(adj1.T > adj1)
    adj2 = adj2 + adj2.T.multiply(adj2.T > adj2) - adj2.multiply(adj2.T > adj2)
    
    g1 = dgl.DGLGraph(adj1)
    g1 = dgl.add_self_loop(g1)
    g2 = dgl.DGLGraph(adj2)
    g2 = dgl.add_self_loop(g2)
    feature1 = torch.FloatTensor(np.array(feature1.todense()))
    feature2 = torch.FloatTensor(np.array(feature2.todense()))
    g1.ndata['fea'] = feature1
    g2.ndata['fea'] = feature2
    return g1, g2

train_ppi = []
train_label = []
with open("/Dataset/ara_cd_hit_10_train_sample.txt") as infile:
    for line in infile:
        line = line.strip().split('\t', 2)
        if line[0] in pro_common and line[1] in pro_common:
            train_ppi.append((line[0],line[1]))
            line[2] = int(line[2])
            train_label.append(line[2])

from tqdm import tqdm
train_samples = []
for i in tqdm(range(len(train_ppi))):
    try:
        g1,g2 = get_gcndata(train_ppi[i][0],train_ppi[i][1])
        train_samples.append((g1,g2,train_label[i]))
    except:
        pass

from sklearn.model_selection import KFold, ShuffleSplit
kf = KFold(n_splits=5,random_state=13, shuffle=True)

import torch.nn as nn
import torch
import math
import random

import torch.nn.functional as F

from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
from torch.utils.data import DataLoader
import dgl.nn.pytorch as dglnn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
class MyGCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, num_heads):
        super(MyGCN, self).__init__()
        self.out1 = dglnn.GraphConv(nfeat, nhid)
        self.transformer_layer = self.get_transformer_layer(nhid, num_heads)
        self.out2 = dglnn.GraphConv(nhid, nhid)
        self.l1 = nn.Linear(nhid, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 2)
    def get_transformer_layer(self, nhid, num_heads):
        return nn.TransformerEncoderLayer(d_model=nhid, nhead=num_heads)
    def forward(self, x1, x2, fea1, fea2):
        fea1 = F.relu(self.out1(x1, fea1))   
        fea2 = F.relu(self.out1(x2, fea2))   

        fea1 = self.transformer_layer(fea1)
        fea2 = self.transformer_layer(fea2)
        
        fea1 = F.relu(self.out2(x1, fea1))   
        fea2 = F.relu(self.out2(x2, fea2))   
        x1.ndata['fea'] = fea1
        x2.ndata['fea'] = fea2
        hg1 = dgl.mean_nodes(x1, 'fea')
        hg2 = dgl.mean_nodes(x2, 'fea')
        hg = torch.mul(hg1, hg2)
        l1 = self.l1(hg)
        l2 = self.l2(l1)
        l3 = F.softmax(self.l3(l2))
        return l3
def collate_GCN(samples):
    g1s,g2s,labels= map(list,zip(*samples))
    return dgl.batch(g1s),dgl.batch(g2s),torch.tensor(labels, dtype=torch.long)

test_ppi = []
test_label = []
with open("/data/ara_cd_hit_10_test_sample.txt") as infile:
    for line in infile:
        line = line.strip().split('\t', 2)
        if line[0] in pro_common and line[1] in pro_common:
            task2_test_ppi.append((line[0],line[1]))
            line[2] = int(line[2])
            task2_test_label.append(line[2])
test_samples = []
for i in tqdm(range(len(test_ppi))):
    try:
        g1,g2 = get_gcndata(test_ppi[i][0],test_ppi[i][1])
        task2_test_samples.append((g1,g2,test_label[i]))
    except:
        pass

test_data_loader = DataLoader(test_samples,collate_fn=collate_GCN)

import torch
import random

max_aupr = 0


epochs = 35   
lr = 0.0001   
weight_decay = 5e-4  
hidden = 512  
dropout = 0.2  
nfeat = 1280 
num_heads = 8

def get_num_correct(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return (correct,len(labels))


def train(epoch,train_loader,val_loader):
    t = time.time()
    model.train()
    
    train_loss = 0
    train_correct = 0
    train_len = 0

    num_losses = 0  
    accumulated_loss = 0
    
    for iter, (batch_g1, batch_g2, batch_label) in enumerate(train_loader):
        fea1 = batch_g1.ndata['fea']
        fea2 = batch_g2.ndata['fea']
        batch_g1, batch_g2, fea1, fea2 = batch_g1.to('cuda:0'), batch_g2.to('cuda:0'), fea1.cuda(), fea2.cuda() 
        batch_label = batch_label.cuda() 
        loss_label = torch.stack((torch.abs(batch_label - 1), batch_label), dim=1).float()
        
        prediction = model(batch_g1, batch_g2, fea1, fea2) 
        loss = loss_func(prediction, loss_label) 
        
        correct,sample_len = get_num_correct(prediction, batch_label)
        train_loss += loss.item()
        train_correct += correct
        train_len += sample_len

        accumulated_loss += loss
        num_losses += 1


        if num_losses == 64:
            optimizer.zero_grad()
            accumulated_loss.backward()
            optimizer.step()

            accumulated_loss = 0
            num_losses = 0
    if num_losses > 0:  
        optimizer.zero_grad()
        accumulated_loss.backward()
        optimizer.step()
    scheduler.step() 
    
    model.eval() 
    with torch.no_grad():
        val_loss_sum = 0
        val_correct_sum = 0
        for g1, g2, label in val_loader:
            fea1 = g1.ndata['fea']
            fea2 = g2.ndata['fea']
            g1, g2, fea1, fea2 = g1.to('cuda:0'), g2.to('cuda:0'), fea1.cuda(), fea2.cuda()
        
            prediction = model(g1, g2, fea1, fea2)
        
            label = label.cuda()
            val_loss_label = torch.stack((torch.abs(label - 1), label), dim=1).float()
            val_loss = loss_func(prediction, val_loss_label) # 计算loss
        
            val_loss_sum += val_loss
            val_correct, val_sample_len = get_num_correct(prediction, label)
            val_correct_sum += val_correct
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(train_loss/len(train_loader)),
          'acc_train: {:.4f}'.format(train_correct/train_len),
          'time: {:.4f}s'.format(time.time() - t),
          'loss_val: {:.4f}'.format(val_loss_sum/len(val_loader)),
          'acc_val: {:.4f}'.format(val_correct_sum/len(val_loader)))
    return (float(train_loss/len(train_loader)),float(train_correct/train_len),float(val_loss_sum/len(val_loader)),float(val_correct_sum/len(val_loader)))    

#import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import csv
import sys
import numpy as np
def get_test_auprc(test_data_loader):
    test_pred = []
    labels = []
    for g1, g2, label in tqdm(test_data_loader):
        fea1 = g1.ndata['fea']
        fea2 = g2.ndata['fea']
        g1, g2, fea1, fea2 = g1.to('cuda:0'), g2.to('cuda:0'), fea1.cuda(), fea2.cuda()
        prediction = model(g1, g2, fea1, fea2)
        prediction = prediction.cpu()
        prediction = prediction.detach().numpy().flatten()
        test_pred.append(prediction)
        label = label.cpu()
        label = label.detach().numpy().flatten()
        labels.append(label)
    test_label = [] 
    for i in labels:
        test_label.append((i[0]))
    test_score = []
    for i in test_pred:
        test_score.append((i[1]))
    test_label = np.array(test_label)
    test_score = np.array(test_score) 
    test_precision, test_recall, _ = precision_recall_curve(test_label, test_score)
    #plt.plot(test_recall, test_precision)
    test_auprc = auc(test_recall, test_precision)
    return (test_auprc,test_label,test_score)


import time
from pygcn.utils import accuracy
t_total = time.time()
all_task2_score = []
all_task2_label = []

train_loss_list_5_cv = []
train_acc_list_5_cv = []
val_loss_list_5_cv = []
val_acc_list_5_cv = []
k = 1
for train_index, val_index in kf.split(train_samples):
    model = MyGCN(nfeat, hidden, dropout, num_heads)
    model = model.cuda() 
    loss_func = nn.BCELoss()
    loss_func = loss_func.cuda() 
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay = 1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

    train_fold = torch.utils.data.dataset.Subset(train_samples, train_index)
    val_fold = torch.utils.data.dataset.Subset(train_samples, val_index) 
    
    train_load = DataLoader(train_fold, batch_size=1, shuffle=False,
                         collate_fn=collate_GCN)
    val_load = DataLoader(val_fold, collate_fn=collate_GCN)
    
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(epochs):
        Loss_train,Acc_train,Loss_val,Acc_val = train(epoch,train_load,val_load)
        train_loss_list.append(Loss_train)
        train_acc_list.append(Acc_train)
        val_loss_list.append(Loss_val)
        val_acc_list.append(Acc_val)
        
    train_loss_list_5_cv.append(train_loss_list)
    train_acc_list_5_cv.append(train_acc_list)
    val_loss_list_5_cv.append(val_loss_list)
    val_acc_list_5_cv.append(val_acc_list)
   
    val_auprc,val_label,val_score = get_test_auprc(val_load)
    test_auprc,test_label,test_score = get_test_auprc(task2_data_loader)
    all_task2_score.append(test_score)
    all_task2_label.append(test_label)
    print(test_auprc)
    k+=1
        
all_task2_score = np.array(all_task2_score)
task2_score_mean = np.mean(all_task2_score, axis=0)
task2_label = all_task2_label[0]
test_precision, test_recall, _ = precision_recall_curve(test_label, task2_score_mean)
test_auprc = auc(test_recall, test_precision)  
print(test_auprc)

