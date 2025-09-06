import numpy as np
import networkx as nx
import random
import scipy.sparse as sp
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import warnings
import os.path

from torch import optim
import torch.nn.functional as F
from torch import nn
import torch
from torch.utils.data import TensorDataset,DataLoader
import time
import multiprocessing as mp
from functools import partial
from tqdm import tqdm


def read_graph(train_filename):
    nodes = set()
    nodes_s = set()
    egs = []
    graph = [{}, {}]

    with open(train_filename) as infile:
        for line in infile.readlines():
            source_node, target_node = line.strip().split(' ')
            source_node = int(source_node)
            target_node = int(target_node)

            nodes.add(source_node)
            nodes.add(target_node)
            nodes_s.add(source_node)
            egs.append([source_node, target_node])

            if source_node not in graph[0]:
                graph[0][source_node] = []
            if target_node not in graph[1]:
                graph[1][target_node] = []

            graph[0][source_node].append(target_node)
            graph[1][target_node].append(source_node)

    n_node = len(nodes)
    return graph, n_node, list(nodes), list(nodes_s), egs


def read_test(test_filename):
    nodes = set()
    nodes_s = set()
    egs = []
    graph = [{}, {}]
    labels = []

    with open(test_filename) as infile:
        for line in infile.readlines():
            source_node, target_node, label = line.strip().split(' ')
            source_node = int(source_node)
            target_node = int(target_node)

            nodes.add(source_node)
            nodes.add(target_node)
            nodes_s.add(source_node)
            egs.append([source_node, target_node])
            labels.append(label)

            if source_node not in graph[0]:
                graph[0][source_node] = []
            if target_node not in graph[1]:
                graph[1][target_node] = []

            graph[0][source_node].append(target_node)
            graph[1][target_node].append(source_node)

    n_node = len(nodes)
    return graph, n_node, list(nodes), list(nodes_s), egs, labels


def read_split_data(data_file):
    neg_strategy = '0'
    s = '6' if 'twitter' in data_file else '5'
    
    graph, n_nodes,nodes,nodes_s,edges = read_graph(data_file+'train_0.'+s)
    graph2, n_nodes2,nodes2,nodes_s2,edges2,labels2 = read_test(data_file+'test_0.'+s+'_'+neg_strategy)
    
    edges_train = np.array(edges)
    adj_train = sp.csr_matrix((np.ones(edges_train.shape[0]), (edges_train[:,0], edges_train[:,1])),
                    shape = (n_nodes, n_nodes))
    
    train_pos_edges = edges.copy()
    test_pos_edges = [edges2[i] for i in range(len(edges2)) if labels2[i]=='1']
    test_neg_edges = [edges2[i] for i in range(len(edges2)) if labels2[i]=='0']
    
    if 'google' in data_file:
        G = nx.DiGraph()
        G.add_nodes_from(list(range(n_nodes)))
        train_pos_edges = [train_pos_edges[i] for i in np.random.permutation(
            len(train_pos_edges))[:int(len(test_pos_edges)/2)]]
        # print(len(train_pos_edges))
        G.add_edges_from(train_pos_edges)
        test_pos_edges_tmp = []
        for u,v in test_pos_edges:
            if not G.has_edge(u, v):
                test_pos_edges_tmp.append([u,v])
        test_pos_edges = test_pos_edges_tmp
        test_neg_edges = [test_neg_edges[i] for i in np.random.permutation(
            len(test_neg_edges))[:int(len(test_pos_edges))]]
    
    return n_nodes, train_pos_edges, test_pos_edges, test_neg_edges, adj_train


def graph2vector(train_pos_edges,test_pos_edges,test_neg_edges,adj,K,max_sub_nodes):
    
    train_pos_features = get_fea(train_pos_edges,adj,K,max_sub_nodes)
    test_pos_features = get_fea(test_pos_edges,adj,K,max_sub_nodes)
    test_neg_features = get_fea(test_neg_edges,adj,K,max_sub_nodes)
    
    test_features = test_pos_features+test_neg_features
    test_labels = [1]*len(test_pos_features)+[0]*len(test_neg_features)
    
    train_pos_features = np.array(train_pos_features).astype(float)
    test_features = np.array(test_features).astype(float)
    test_labels = np.array(test_labels).astype(float)
    
    return train_pos_features, test_features, test_labels


def get_fea(edge_list,adj,K,max_sub_nodes):
    features = []
    for edge in tqdm(edge_list, desc="Processing edges", unit="edge"):
        res = subgraph2vec(edge,adj,K,max_sub_nodes)
        features.append(res)
    
    return features


def subgraph2vec(ebunch,A,K,max_sub_nodes):
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
    
    n_nodes = np.size(V_K)
    if n_nodes>max_sub_nodes:
        idx = np.random.permutation(n_nodes-2)
        V_K = np.concatenate([V_K[:2],V_K[idx[:max_sub_nodes-2]+2]])
    
    sub = A[V_K,:][:,V_K].toarray()
    n_nodes = sub.shape[0]
    sub[0,1] = 0
    for j in range(n_nodes-1):
        sub[j,j+1:] = sub[j,j+1:]/nodes_dist[j]
        sub[j+1:,j] = sub[j+1:,j]/nodes_dist[j]
    a = sub
    s = []
    
    # tmp = a.dot(a).dot(a)
    # s.append(tmp)
    # tmp = a.T.dot(a).dot(a)
    # s.append(tmp)
    tmp = a.dot(a.T).dot(a)
    s.append(tmp)
    # tmp = a.dot(a).dot(a.T)
    # s.append(tmp)
    # tmp = a.dot(a.T).dot(a.T)
    # s.append(tmp)
    # tmp = a.T.dot(a).dot(a.T)
    # s.append(tmp)
    # tmp = a.T.dot(a.T).dot(a)
    # s.append(tmp)
    # tmp = a.T.dot(a.T).dot(a.T)
    # s.append(tmp)
    
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
    for i in range(len(res)):
        res[i] = res[i].reshape(-1)
    
    res = np.hstack(res)
    
    return res

def save_split_data(path_split,train_pos_edges, test_pos_edges, test_neg_edges):
    np.save(path_split+'/'+data_file[5:-1]+'_train_pos.npy',np.array(train_pos_edges))
    np.save(path_split+'/'+data_file[5:-1]+'_test_pos.npy',np.array(test_pos_edges))
    np.save(path_split+'/'+data_file[5:-1]+'_test_neg.npy',np.array(test_neg_edges))


def load_split_data(path_split,data_file,dict_n_nodes,i_loop):
    train_pos_edges = np.load(path_split+'/'+data_file[5:-1]+'_train_pos.npy')
    test_pos_edges = np.load(path_split+'/'+data_file[5:-1]+'_test_pos.npy')
    test_neg_edges = np.load(path_split+'/'+data_file[5:-1]+'_test_neg.npy')
    n_nodes = dict_n_nodes[data_file[5:-1]]
    adj_train = sp.csr_matrix((np.ones(train_pos_edges.shape[0]), (train_pos_edges[:,0], train_pos_edges[:,1])),
                    shape = (n_nodes, n_nodes))
    train_pos_edges = [list(i) for i in train_pos_edges]
    test_pos_edges = [list(i) for i in test_pos_edges]
    test_neg_edges = [list(i) for i in test_neg_edges]
    
    return n_nodes, train_pos_edges, test_pos_edges, test_neg_edges, adj_train
    

def split_neg(n_nodes, train_pos_edges, test_pos_edges, test_neg_edges, neg_seed):
    G = nx.DiGraph()
    G.add_nodes_from(list(range(n_nodes)))
    G.add_edges_from(train_pos_edges+test_pos_edges+test_neg_edges)
    
    np.random.seed(neg_seed)
    random.seed(neg_seed)
    train_neg_edges = []
    num_neg = 0
    num_neg_all = len(train_pos_edges)
    while num_neg<num_neg_all:
        u = random.randint(0,n_nodes-1)
        v = random.randint(0,n_nodes-1)
        if u==v:
            u = random.randint(0,n_nodes-1)
            v = random.randint(0,n_nodes-1)
        if not G.has_edge(u, v):
            train_neg_edges.append([u,v])
            num_neg += 1
            G.add_edge(u, v)
    
    return train_neg_edges


def evaluate(model,loader,device):
    model.eval()
    all_targets = []
    all_scores = []
    
    for data,target in loader:
        data = data.to(device)
        target = target.to(device)
        out = model(data)
        all_scores.append(F.softmax(out,dim=1)[:, 1].cpu().detach())
        all_targets.extend(target.tolist())
    all_scores = torch.cat(all_scores).cpu().numpy()
    upt_res = roc_auc_score(all_targets,all_scores)
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



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    
    K = 5
    data_list = ['cora','epinions','google','twitter']
    dict_n_nodes = {'cora':23166,'epinions':75879,'google':15763,'twitter':465017}
    n_loops = 10
    max_sub_nodes = 100
    auc = np.zeros((len(data_list),n_loops))
    
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    batch_size = 512
    lr = 0.001
    epochs = 100
    upd = 5
    
    
    for i_data, data_file_ori in enumerate(data_list):
        
        data_file = 'data/'+data_file_ori+'/'
        print('dataset =',data_file[5:-1])
        n_nodes, train_pos_edges, test_pos_edges, test_neg_edges, adj_train = \
            read_split_data(data_file)
        train_pos_features, test_features_ori, test_labels_ori = \
            graph2vector(train_pos_edges,test_pos_edges,test_neg_edges,adj_train,K,max_sub_nodes)
            
        n0 = 0
        for i_loop in range(n0,n_loops):
            print('dataset =',data_file[5:-1],'; loop =', i_loop+1)
            
            
            neg_seed = i_loop+1
            train_neg_edges = split_neg(n_nodes, train_pos_edges, test_pos_edges, test_neg_edges, neg_seed)
            
            
            train_neg_features = get_fea(train_neg_edges,adj_train,K,max_sub_nodes)
            train_neg_features = np.array(train_neg_features).astype(float)
            train_labels = [1]*train_pos_features.shape[0]+[0]*train_neg_features.shape[0]
            train_labels = np.array(train_labels).astype(float)
            train_features = np.vstack([train_pos_features,train_neg_features])
            
            train_features = torch.tensor(train_features, dtype=torch.float).to(device)
            test_features = torch.tensor(test_features_ori, dtype=torch.float).to(device)
            train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
            test_labels = torch.tensor(test_labels_ori, dtype=torch.long).to(device)
            
            n_feas = train_features.shape[1]
            
            train_data = TensorDataset(train_features,train_labels)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            test_data = TensorDataset(test_features,test_labels)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
            
            model = mlp(n_feas, 32, 16).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            model.to(device)
            loss_func = nn.CrossEntropyLoss()
            
            best_res = 0
            for epoch in range(epochs):
                model.train()
                total_loss = []
                
                n_samples = 0
                for (data, target) in train_loader:
                    data = data.to(device)
                    target = target.to(device)
                    out = model(data)
                    loss = loss_func(out, target)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss.append( loss.item() * len(target))
                    n_samples += len(target)
                    
            
                total_loss = np.array(total_loss)
                avg_loss = np.sum(total_loss, 0) / n_samples
                
                if (epoch + 1) % upd == 0:
                    upt_res = evaluate(model,test_loader,device)
                    
            upt_res = evaluate(model,test_loader,device)
            auc[i_data,i_loop] = upt_res
            torch.cuda.empty_cache()
    print('mean reslults :',auc.mean(1))
    print('standard deviation reslults :',auc.std(1))
    