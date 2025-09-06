# -*- coding: utf-8 -*-

import warnings
import random
import numpy as np
import scipy.sparse as sp
import networkx as nx
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import math
from torch import optim
import torch.nn.functional as F
from torch import nn
import torch
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
import os



def get_graph(adj):
    G = nx.DiGraph()
    G.add_nodes_from(list(range( np.size(adj,0) )))
    tmp = sp.coo_matrix(adj)
    row,col,data = tmp.row,tmp.col,tmp.data
    
    for u,v,s in zip(row,col,data):
        G.add_edge(u, v, weight = s)
        
    return G


def get_fea(edge_list,G2,adj,K):
    data_features = []
    for edge in tqdm(edge_list, desc="Processing edges", unit="edge"):
        res = subgraph2vec(edge,G2,adj,K)
        data_features.append(res)
    
    return data_features



def graph2vector(G2, r_val, train_pos_edges,train_neg_edges,val_pos_edges,val_neg_edges,test_pos_edges,test_neg_edges,adj,K):
    
    train_pos_features = get_fea(train_pos_edges,G2,adj,K)
    train_neg_features = get_fea(train_neg_edges,G2,adj,K)
    test_pos_features = get_fea(test_pos_edges,G2,adj,K)
    test_neg_features = get_fea(test_neg_edges,G2,adj,K)
    
    train_features = train_pos_features+train_neg_features
    train_labels = [1]*len(train_pos_features)+[0]*len(train_neg_features)
    test_features = test_pos_features+test_neg_features
    test_labels = [1]*len(test_pos_features)+[0]*len(test_neg_features)
    
    train_features = np.array(train_features).astype(float)
    train_labels = np.array(train_labels).astype(float)
    test_features = np.array(test_features).astype(float)
    test_labels = np.array(test_labels).astype(float)
    
    if r_val>0:
        val_pos_features = get_fea(val_pos_edges,G2,adj,K)
        val_neg_features = get_fea(val_neg_edges,G2,adj,K)
        val_features = val_pos_features+val_neg_features
        val_labels = [1]*len(val_pos_features)+[0]*len(val_neg_features)
        val_features = np.array(val_features).astype(float)
        val_labels = np.array(val_labels).astype(float)
    else:
        val_features = None
        val_labels = None

    return train_features, train_labels, val_features, val_labels, test_features, test_labels


def subgraph2vec(ebunch,G2,A,K):
    u,v = ebunch
    V_K = np.array([u,v])
    fringe = np.array([u,v])
    nodes_dist = np.array([1,1])
    dist = 1
    while np.size(V_K)<K and np.size(fringe)>0:
        nei = np.array([]).astype(int)
        nei_out = sp.find(A[fringe,:])[1]
        nei_in = sp.find(A[:,fringe])[0]
        nei = np.concatenate((nei,nei_out,nei_in))
            
        nei = np.unique(nei)
        fringe = np.setdiff1d(nei,V_K)
        V_K = np.concatenate((V_K,fringe))
        dist = dist+1
        nodes_dist = np.concatenate(( nodes_dist,dist*np.ones(fringe.shape[0]) ))
        break
        
    sub = A[V_K,:][:,V_K].toarray()
    n_nodes = sub.shape[0]
    sub[0,1] = 0
    for j in range(n_nodes-1):
        sub[j,j+1:] = sub[j,j+1:]/nodes_dist[j]
        sub[j+1:,j] = sub[j+1:,j]/nodes_dist[j]
    a_p = sub.copy()
    a_p[a_p<0] = 0
    a_n = sub.copy()
    a_n[a_n>0] = 0
    a_n = -a_n
    s = []
    
    tmp = a_p.dot(a_p.T).dot(a_p)
    s.append(tmp)
    tmp = a_p.dot(a_n.T).dot(a_n)
    s.append(tmp)
    tmp = a_n.dot(a_p.T).dot(a_n)
    s.append(tmp)
    tmp = a_n.dot(a_n.T).dot(a_p)
    s.append(tmp)
    
    tmp = a_n.dot(a_n.T).dot(a_n)
    s.append(tmp)
    tmp = a_n.dot(a_p.T).dot(a_p)
    s.append(tmp)
    tmp = a_p.dot(a_n.T).dot(a_p)
    s.append(tmp)
    tmp = a_p.dot(a_p.T).dot(a_n)
    s.append(tmp)
    
    n_groups = len(s)
    orders = []
    
    for i in range(n_groups):
        tmp = np.abs(s[i])
        score = tmp[2:,0]+tmp[2:,1]+tmp[0,2:]+tmp[1,2:]
        orderi = np.concatenate(( np.array([0,1]),np.flip(np.argsort(score))+2 ))
        orders.append(orderi)
    
    if n_nodes>=K:
        res = []
        for i in range(n_groups):
            res.append(s[i][orders[i][:K],:][:,orders[i][:K]])
    else:
        res = []
        for i in range(n_groups):
            tmp = np.zeros([K,K])
            tmp[:n_nodes,:][:,:n_nodes] = s[i][orders[i],:][:,orders[i]]
            res.append(tmp)
    for i in range(n_groups):
        res[i] = res[i].reshape(-1)
    
    res = np.hstack(res)
    
    return res


def calculating_mean_std(auc):
    mean = np.mean(auc,1)*100
    std = np.std(auc,1)*100
    return mean, std


def evaluate(model,loader,device):
    model.eval()
    all_targets = []
    all_scores = []
    
    for data,target in loader:
        data = data.to(device)
        target = target.to(device)
        out,_ = model(data)
        all_scores.append(F.softmax(out,dim=1)[:, 1].cpu().detach())
        all_targets.extend(target.tolist())
    all_scores = torch.cat(all_scores).cpu().numpy()
    y_pred = np.zeros(np.size(all_scores))
    y_pred[np.where(all_scores>0.5)] = 1
    upt_res = [0]*7
    upt_res[0] = roc_auc_score(all_targets,all_scores)
    upt_res[1] = f1_score(all_targets,y_pred, average='macro')
    upt_res[2] = f1_score(all_targets,y_pred, average='micro')
    upt_res[3] = f1_score(all_targets,y_pred)
    # upt_res[4] = precision_score(all_targets,y_pred)
    # upt_res[5] = recall_score(all_targets,y_pred)
    # upt_res[6] = accuracy_score(all_targets,y_pred)
    return upt_res


class mlp(nn.Module):
    def __init__(self, in_dim,hid_dim,hid_dim2):
        super(mlp, self).__init__()
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, hid_dim2)
        self.lin4 = nn.Linear(hid_dim2, 2)
    
    def forward(self, fea):
        x = self.lin1(fea)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = F.relu(x)
        x = self.lin4(x)
        
        return x


class attmlp(nn.Module):
    def __init__(self, in_dim,hid_dim,hid_dim2,n_heads):
        super(attmlp, self).__init__()
        in_dim = int(in_dim/8)
        self.in_dim = in_dim
        self.n_heads = n_heads
        
        self.att_list = torch.nn.ModuleList()
        for i in range(n_heads):
            for j in range(8):
                self.att_list.append(nn.Linear(in_dim, 1))
        
        self.lin_enc = nn.Linear(in_dim, in_dim)
        
        self.lin1 = nn.Linear(in_dim*n_heads, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, hid_dim2)
        self.lin4 = nn.Linear(hid_dim2, 2)
        
        
    def forward(self, x):
        std_att = []
        z_cat = []
        for i in range(self.n_heads):
            z = 0
            for j in range(8):
                idx = i*self.n_heads+j
                xx = x[:,self.in_dim*j:self.in_dim*(j+1)]
                # xx = self.lin_enc(xx)
                score = self.att_list[idx](xx)
                score = F.tanh(score)
                zz = torch.mul(xx,score)
                z += zz
                std_att.append(score.std())
            z_cat.append(z)
        z_cat = torch.cat(z_cat,1)
        
        
        x = self.lin1(z_cat)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = F.relu(x)
        x = self.lin4(x)
        
        return x,std_att



if __name__=='__main__':
    edgepath = ['soc-sign-bitcoinalpha.csv','soc-sign-bitcoinotc.csv','wiki-RfA.txt',
                'soc-sign-Slashdot090221.txt','soc-sign-epinions.txt']
    
    n = 5
    seed = [i for i in range(1,n+1)]
    
    warnings.filterwarnings('ignore')
    count = np.loadtxt('input/count.txt').astype(int)
    
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if not os.path.exists('./save_model'):
        os.makedirs('./save_model')
    
    K = 5
    test_size = 0.2
    batch_size = 512
    r_val = 0.00
    lr = 0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    epochs = 100
    upd = 5
    hid_dim = 32
    hid_dim2 = 16
    n_heads = 3
    
    auc = np.zeros((len(edgepath),n))
    f1 = np.zeros((len(edgepath),n))
    f1_micro = np.zeros((len(edgepath),n))
    f1_macro = np.zeros((len(edgepath),n))
    mean = np.zeros((len(edgepath),4))
    std = np.zeros((len(edgepath),4))
    for i_data,datapath in enumerate(edgepath):
        for i_loop in range(n):
            print('dataset = '+datapath,'; test size = '+str(test_size),'; loop num = '+str(i_loop+1))
            split_seed = seed[i_loop]
            train_path = 'input/'+datapath[:-4]+'_train_'+str(split_seed)+'.txt'
            train_edges = np.loadtxt(train_path).astype(int)
            test_path = 'input/'+datapath[:-4]+'_test_'+str(split_seed)+'.txt'
            test_edges = np.loadtxt(test_path).astype(int)
            
            num_nodes = count[i_data][0]
            adj_train = sp.csr_matrix((train_edges[:,2].astype(float), (train_edges[:,0], train_edges[:,1])),
                           shape = (num_nodes, num_nodes))
            
            train_pos_edges = [[u,v] for u,v,s in train_edges if s==1]
            train_neg_edges = [[u,v] for u,v,s in train_edges if s==-1]
            test_pos_edges = [[u,v] for u,v,s in test_edges if s==1]
            test_neg_edges = [[u,v] for u,v,s in test_edges if s==-1]
            
            if r_val>0:
                random.shuffle(train_pos_edges)
                random.shuffle(train_neg_edges)
                n_val_pos = int(len(train_pos_edges)*r_val)
                n_val_neg = int(len(train_neg_edges)*r_val)
                val_pos_edges = train_pos_edges[:n_val_pos]
                val_neg_edges = train_neg_edges[:n_val_neg]
                train_pos_edges = train_pos_edges[n_val_pos:]
                train_neg_edges = train_neg_edges[n_val_neg:]
                edges = np.array(train_pos_edges+train_neg_edges)
                data = np.array([1]*len(train_pos_edges)+[-1]*len(train_neg_edges))
                adj_train = sp.csr_matrix((data.astype(float), (edges[:,0], edges[:,1])),
                               shape = (num_nodes, num_nodes))
            else:
                val_pos_edges = None
                val_neg_edges = None
            
            G2 = get_graph(adj_train)
            train_features, train_labels, val_features, val_labels, test_features, test_labels = \
                graph2vector(G2, r_val, train_pos_edges,train_neg_edges,val_pos_edges,val_neg_edges,test_pos_edges,test_neg_edges,adj_train,K)
            
            
            n_feas = train_features.shape[1]
            
            train_features = train_features*0.01
            test_features = test_features*0.01
            
    
            train_features = torch.tensor(train_features, dtype=torch.float)
            test_features = torch.tensor(test_features, dtype=torch.float)
            train_labels = torch.tensor(train_labels, dtype=torch.long)
            test_labels = torch.tensor(test_labels, dtype=torch.long)
            
            train_data = TensorDataset(train_features, train_labels)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            test_data = TensorDataset(test_features, test_labels)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
            
            if r_val>0:
                val_features = torch.tensor(val_features, dtype=torch.float)
                val_labels = torch.tensor(val_labels, dtype=torch.long)
                val_data = TensorDataset(val_features, val_labels)
                val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
            
            
            
            model = attmlp(n_feas, hid_dim,hid_dim2,n_heads).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            model.to(device)
            loss_func = nn.CrossEntropyLoss()
            mae_loss = nn.L1Loss(reduction='mean')
            
            best_res = 0
            for epoch in range(epochs):
                model.train()
                total_loss = []
                
                n_samples = 0
                for (data, target) in train_loader:
                    data = data.to(device)
                    target = target.to(device)
                    out,std_att = model(data)
                    loss = loss_func(out, target)
                    
                    std_att = sum(std_att).reshape(1,1)
                    loss2 = 0.2*mae_loss(std_att,torch.tensor([[0]], dtype=torch.float).to(device))
                    
                    loss = loss+loss2
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss.append( loss.item() * len(target))
                    n_samples += len(target)
                    
                total_loss = np.array(total_loss)
                avg_loss = np.sum(total_loss, 0) / n_samples
                
                if (epoch + 1) % upd == 0:
                    upt_res = evaluate(model,test_loader,device)
                    if r_val>0:
                        upt_res = evaluate(model,val_loader,device)
                    
                    if upt_res[0]+upt_res[1] > best_res and r_val>0:
                        torch.save(obj=model.state_dict(), f='save_model/model'+datapath[:-4]+'_'+str(i_loop+1)+'.pth')
                        best_res = upt_res[0]+upt_res[1]
                    
            if r_val>0:
                new_model = attmlp(n_feas, hid_dim,hid_dim2,n_heads).to(device)
                new_model.load_state_dict(torch.load('save_model/model'+datapath[:-4]+'_'+str(i_loop+1)+'.pth'))
                upt_res = evaluate(new_model,test_loader,device)
            else:
                upt_res = evaluate(model,test_loader,device)
            auc[i_data][i_loop],f1_macro[i_data][i_loop],f1_micro[i_data][i_loop],f1[i_data][i_loop] = \
                upt_res[0],upt_res[1],upt_res[2],upt_res[3]
            
            torch.cuda.empty_cache()
            
            
            mean[:,0], std[:,0] = calculating_mean_std(f1_micro)
            mean[:,1], std[:,1] = calculating_mean_std(f1)
            mean[:,2], std[:,2] = calculating_mean_std(f1_macro)
            mean[:,3], std[:,3] = calculating_mean_std(auc)
            
            
            
    print('mean reslults :',auc.mean(1))
    print('standard deviation reslults :',auc.std(1))