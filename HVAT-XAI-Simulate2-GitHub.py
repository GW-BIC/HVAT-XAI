# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 13:18:14 2025

@author: VHAWASShaoY
"""

import numpy as np
import math
import torch
from torch import nn
import time
from sklearn import metrics
import matplotlib.pyplot as plt

class TemporalEncoding(nn.Module):
    def __init__(self,d_embed,max_len=30):
        super(TemporalEncoding,self).__init__()

        te = torch.zeros(max_len,d_embed)
        times = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        omegas = torch.exp(torch.arange(0,d_embed,2).float()*(-math.log(10000.0)/d_embed))
        te[:,0::2] = torch.sin(times*omegas)
        te[:,1::2] = torch.cos(times*omegas)
        self.register_buffer('te',te)

    def forward(self,time_idxs):
        return nn.functional.embedding(time_idxs,self.te)

class VAT(nn.Module):
    def __init__(self,n_concept,d_embed,nhead,d_hid,nblocks,dropout=0.1):
        super(VAT,self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.d_embed = d_embed
        self.token_embed = nn.Embedding(n_concept+1,d_embed)
        self.value_embed = nn.Embedding(n_concept+1,d_embed)
        self.time_encoder = TemporalEncoding(d_embed)
        encoder_layers = TransformerEncoderLayer(d_embed,nhead,d_hid,dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers,nblocks)

    def forward(self,time_ids,token_ids,token_vals,src_key_padding_mask=None):
        src = self.time_encoder(time_ids)+self.token_embed(token_ids)\
            +token_vals*self.value_embed(token_ids)
        output = self.transformer_encoder(src,src_key_padding_mask=src_key_padding_mask)
        return output[0] #get the 0th item in the sequence

class FFNN(nn.Module):
    def __init__(self,d_in,d_hid,d_out):
        super(FFNN,self).__init__()
        self.fc1 = nn.Linear(d_in,d_out)
        self.dropout1 = nn.Dropout(p=0.1)
        self.norm1 = nn.LayerNorm(d_out)
        self.fc2 = nn.Linear(d_out,d_hid)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(d_hid,d_out)
        self.dropout3 = nn.Dropout(p=0.1)
        self.norm2 = nn.LayerNorm(d_out)
    def forward(self,input):
        output = self.norm1(self.dropout1(self.fc1(input)))
        output2 = self.dropout2(nn.functional.relu(self.fc2(output)))
        output = output + self.dropout3(self.fc3(output2))
        output = self.norm2(output)
        return output

class HVAT(nn.Module):
    def __init__(self,n_concept,d_embed,nhead,d_hid,nblocks,d_ffnn_in):
        super(HVAT,self).__init__()
        self.transformer = VAT(n_concept = n_concept,
                                    d_embed=d_embed,nhead=nhead,d_hid=d_hid,nblocks=nblocks)
        self.ffnn = FFNN(d_ffnn_in,d_hid,d_embed)
        self.ffnn2 = FFNN(d_embed,d_hid,d_embed)
        self.clf = nn.Linear(d_embed,1)

    def forward(self,input,token_ids,time_ids,token_vals,src_key_padding_mask=None):
        output1 = self.ffnn(input)
        output2 = self.transformer(token_ids,time_ids,token_vals,src_key_padding_mask)
        output = output1+output2
        output = self.ffnn2(output)
        output = torch.sigmoid(self.clf(output)).flatten()
        return output

if __name__ == '__main__':
    
    # Generating simulated data
    sample_size = 20000
    np.random.seed(1)
    x1 = np.random.uniform(low=-1,high=1,size=sample_size)
    x2 = np.random.binomial(n=1,p=0.4,size=sample_size).astype(np.float)
    ts = np.arange(5).reshape((-1,1))-2
    k = np.random.uniform(low=-2,high=2,size=sample_size)
    b = np.random.uniform(low=-1,high=1,size=sample_size)
    e = np.random.normal(loc=0,scale=0.02,size=(5,sample_size))#*k.reshape((1,-1))*0.5
    x3 = b + k*ts + e
    x3_means = x3.mean(axis=0)
    x3_slopes = (ts*(x3-x3_means)).sum(axis=0)/(ts*ts).sum()
    y = -1 + 0.4*x1 - 0.3*x2 + 0.9*x3_means - 2*x3_slopes*(x3_slopes<0)
    p = 1/(1+np.exp(-y))
    z = np.random.binomial(n=1,p=p)

    slice_te = slice(0,int(0.1*sample_size+0.5))
    slice_vl = slice(int(0.1*sample_size+0.5),int(0.2*sample_size+0.5))
    slice_tr = slice(int(0.2*sample_size+0.5),sample_size)
    data_nonl = torch.stack([torch.tensor(x1),torch.tensor(x2)]).float().t()
    zz = torch.tensor(z).float()

    x3_mean = x3.mean()
    x3_std = x3.std()
    seqs = (x3-x3_mean)/x3_std
    seqs = np.concatenate([np.zeros((1,sample_size)),seqs],axis=0).astype(np.float32)
    data_seq = torch.tensor(seqs).unsqueeze(dim=-1)
    
    # Setting up mini-batches
    batches = []
    for i in range(sample_size//50):
        subidxs = torch.arange(i*50,(i+1)*50)
        batch_times = (torch.arange(6)*torch.ones((50,1),dtype=int)).T
        batch_tokens = torch.ones((6,50),dtype=int)
        batch_tokens[0] = 0
        batch_values = data_seq[:,subidxs]
        batches.append([subidxs,batch_times,batch_tokens,batch_values])

    batch_iidx_tr = []
    for batch in batches:
        batch_iidx_tr.append((batch[0]>=200).nonzero().flatten())
    
    # HVAT model initialization
    torch.manual_seed(1)
    model = HVAT(n_concept=1,d_embed=32,nhead=2,d_hid=30,nblocks=2,d_ffnn_in=2)
    model.clf.bias.data[:] = 0
    
    criterion = nn.BCELoss(reduction='sum')
    optimizer = torch.optim.SGD([
            {'params':model.transformer.parameters(),'lr':5e-5},
            {'params':model.ffnn.parameters(),'lr':5e-4},
            {'params':model.ffnn2.parameters()},
            {'params':model.clf.parameters()}
            ],lr=1e-4,momentum=0.9,nesterov=True)
    epochs = 0
    np.random.seed(1)
    torch.set_num_threads(3)
    AUCs_tr,AUCs_vl = [],[]
    no_improve = 0

    # HVAT model training
    num_epochs = 100
    start = time.time()
    print(time.strftime('%H:%M:%S'))
    for epoch in range(num_epochs+1):
        if epoch>=1:
            epochs += 1
            batch_idxs = list(range(len(batches)))
            np.random.shuffle(batch_idxs)
            model.train()
            for b,i in enumerate(batch_idxs):
                if b%30==0:
                    print('.',end='')
                idxs,time_ids,token_ids,token_vals = batches[i]
                ii = batch_iidx_tr[i]
                if ii.size(0)==0:
                    continue
                output = model(data_nonl[idxs[ii]],time_ids[:,ii],token_ids[:,ii],token_vals[:,ii],None)
                loss = criterion(output,zz[idxs[ii]])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print(' ',end='')
        elif epochs>0:
            continue
        scores = np.zeros(sample_size)
        model.eval()
        with torch.no_grad():
            for i,batch in enumerate(batches):
                if i%30==0:
                    print('.',end='')
                idxs,time_ids,token_ids,token_vals = batch
                output = model(data_nonl[idxs],time_ids,token_ids,token_vals)
                scores[idxs] = output
        print()
        fprs,tprs,_ = metrics.roc_curve(z[slice_tr],scores[slice_tr])
        auc_tr = metrics.auc(fprs,tprs)
        fprs,tprs,_ = metrics.roc_curve(z[slice_vl],scores[slice_vl])
        auc_vl = metrics.auc(fprs,tprs)

        print('{:2d}: {:.3f}    {:.3f}'.format(
            epochs,auc_tr*100,auc_vl*100))

    # XAI
    # Impact score for temporal mean of x3
    scores2 = np.zeros(sample_size)
    model.eval()
    with torch.no_grad():
        for i,batch in enumerate(batches):
            idxs,time_ids,token_ids,token_vals = batch
            token_vals2 = token_vals.clone()
            token_vals2[1:,:,0] = token_vals2[1:,:,0]-token_vals2[1:,:,0].mean(dim=0)
            scores2[idxs] = model(data_nonl[idxs],time_ids,token_ids,token_vals2)

    x3_mean_ref = 0
    idxs_nz = np.abs(x3_means-x3_mean_ref)>=0.01
    impacts = np.log(scores[idxs_nz]/(1-scores[idxs_nz]))-np.log(scores2[idxs_nz]/(1-scores2[idxs_nz]))
    impact_scores = impacts/(x3_means[idxs_nz]-x3_mean_ref)
    impact_score = impact_scores.mean()
    print(impact_score)
    
    # Impact by value for temporal mean of x3
    vs = np.arange(-1,1.2,0.2)
    v_impact= []
    for v in vs:
        idxs = np.abs(x3_means-v)<0.1
        v_impact.append(impacts[idxs[idxs_nz]].mean())

    plt.plot(vs,v_impact,color='tab:blue',
             lw=2,label='Model based')
    plt.plot([-1,1],[-0.9,0.9],ls='dashed',label='Ground truth')
    plt.legend()

    # Impact scores for temporal slope of x3
    scores2 = np.zeros(sample_size)
    model.eval()
    with torch.no_grad():
        for i,batch in enumerate(batches):
            idxs,time_ids,token_ids,token_vals = batch
            token_vals2 = token_vals.clone()
            token_vals2[1:,:,0] = token_vals2[1:,:,0].mean(dim=0)
            scores2[idxs] = model(data_nonl[idxs],time_ids,token_ids,token_vals2)

    x3_slope_ref = 0
    idxs_nz = np.abs(x3_slopes-x3_slope_ref)>=0.02
    impacts = np.log(scores[idxs_nz]/(1-scores[idxs_nz]))-np.log(scores2[idxs_nz]/(1-scores2[idxs_nz]))
    impact_scores = impacts/(x3_slopes[idxs_nz]-x3_slope_ref)
    impact_score = impact_scores.mean()
    print(impact_score)
    
    imsc_slope_neg = impact_scores[x3_slopes[idxs_nz]<0].mean()
    print(imsc_slope_neg)
    imsc_slope_pos = impact_scores[x3_slopes[idxs_nz]>0].mean()
    print(imsc_slope_pos)

    # Impact by value for temporal slope of x3
    vs = np.arange(-1.9,2.0,0.2)
    v_impact = []
    for v in vs:
        idxs = np.abs(x3_slopes-v)<0.1
        v_impact.append(impacts[idxs[idxs_nz]].mean())

    plt.plot(np.array(vs),np.array(v_impact),color='tab:blue',
             lw=2, label='Model based')
    plt.plot([-2,0,2],[4,0,0],ls='dashed', label='Ground truth')
    plt.legend()