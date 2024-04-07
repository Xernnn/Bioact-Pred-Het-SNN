# -*- coding: utf-8 -*-
"""Transfer_Learning_&_Case_Study.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1W9UKNzsuvbk_sUAzVF0tnPD_9VS43iVv

# Install Library

[RDKit ](https://github.com/rdkit/rdkit)

[DGL](https://github.com/dmlc/dgl/)

[DGL-LifeSci](https://github.com/awslabs/dgl-lifesci)

# Import Library
"""

import os

import dgl
import sys
import torch
import random
import cv2
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit import DataStructs

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import  History
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, AttentiveFPAtomFeaturizer
from sklearn.model_selection import train_test_split

from Utils.general import DATASET, get_dataset, separate_active_and_inactive_data, get_embedding_vector_class, count_lablel,data_generator, up_and_down_Samplenig
from Utils.gcnpretrained import get_sider_model
from Utils.specialfunctions import is_Membership

from Models.heterogeneous_siamese_sider import siamese_model_attentiveFp_sider

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

"""# Data"""

cache_path_tox21='./tox21_dglgraph.bin'

df_tox21 = get_dataset("tox21")
ids = df_tox21['mol_id']

df_tox21 = df_tox21.drop(columns=['mol_id'])

cache_path_sider='./sider_dglgraph.bin'

df = get_dataset("sider")

tox21_tasks = df_tox21.columns.values[:12].tolist()
tox21_tasks

tox21_smiles = np.array(df_tox21['smiles'])
sider_smiles = np.array(df['smiles'])

subscriber = []
for ts in tox21_smiles:
    for ss in sider_smiles:
        if ts == ss:
            subscriber.append(ts)

subscriber

"""# Required functions"""

def create_dataset_with_gcn_case_study(dataset, class_embed_vector, GCN, tasks):
    created_data = []
    data = np.arange(len(tasks))
    onehot_encoded = to_categorical(data)
    for i, data in enumerate(dataset):
        smiles, g, labels, mask = data
        g = g.to(device)
        g = dgl.add_self_loop(g)
        graph_feats = g.ndata.pop('h')
        embbed = GCN(g, graph_feats)
        embbed = embbed.to('cpu')
        embbed = embbed.detach().numpy()
        for j, label in enumerate(labels):
            a = (smiles, embbed, onehot_encoded[j], class_embed_vector[j], labels[j], tasks[j])
            created_data.append(a)
    print('Data created!!')
    return created_data


def create_dataset_with_gcn(dataset, subscriber, class_embed_vector, GCN, tasks, numberTask):

    created_data = []
    created_subscriber = []
    data = np.arange(len(tasks))
    onehot_encoded = to_categorical(data)

    for i, data in enumerate(dataset):
        smiles, g, label, mask = data
#         g = g.to(device)
        g = dgl.add_self_loop(g)
        graph_feats = g.ndata.pop('h')
        embbed = GCN(g, graph_feats)
        embbed = embbed.to('cpu')
        embbed = embbed.detach().numpy()
        a = (smiles, embbed, onehot_encoded[numberTask], class_embed_vector[numberTask], label, tasks[numberTask])
        if smiles in subscriber:
            created_subscriber.append(data)
        else:
            created_data.append(a)
    print('Data created!!')
    return created_data, created_subscriber

"""# Calculation of embedded vectors for each class"""

print(df_tox21, tox21_tasks)

df_positive, df_negative = separate_active_and_inactive_data(df_tox21, tox21_tasks)

for i,d in enumerate(zip(df_positive,df_negative)):
    print(f'{tox21_tasks[i]}=> positive: {len(d[0])} - negative: {len(d[1])}')

dataset_positive = [DATASET(d,smiles_to_bigraph, AttentiveFPAtomFeaturizer(), cache_file_path = cache_path_tox21) for d in df_positive]
dataset_negative = [DATASET(d,smiles_to_bigraph, AttentiveFPAtomFeaturizer(), cache_file_path = cache_path_tox21) for d in df_negative]

embed_class_tox21 = get_embedding_vector_class(dataset_positive, dataset_negative, radius=2, size = 512)

"""# Transfer Learning with BioAct-Het and AttentiveFp GCN"""

model_name = 'GCN_attentivefp_SIDER'
gcn_model = get_sider_model(model_name)
gcn_model.eval()
# gcn_model = gcn_model.to(device)

data_ds = []
subscriber_data_ds = []
for i, task in  enumerate(tox21_tasks):
    a = df_tox21[['smiles' , task]]
    a = a.dropna()
    ds = DATASET(a,smiles_to_bigraph, AttentiveFPAtomFeaturizer(), cache_file_path = cache_path_sider)
    data, subscriber_data = create_dataset_with_gcn(ds, subscriber, embed_class_tox21, gcn_model, tox21_tasks, i)
    for d in data:
        data_ds.append(d)
    for d in subscriber_data:
        subscriber_data_ds.append(d)

from sklearn.model_selection import KFold

Epoch_S = 10

def evaluate_model(dataset, subscriber_dataset, k = 10 , shuffle = False):
    result =[]

    kf = KFold(n_splits=10, shuffle= shuffle, random_state=None)

    for train_index, test_index in kf.split(dataset):

        train_ds = [dataset[index] for index in train_index]

        valid_ds = [dataset[index] for index in test_index]

        label_pos , label_neg, _ , _ = count_lablel(train_ds)
        print(f'train positive label: {label_pos} - train negative label: {label_neg}')

        # train_ds = up_and_down_Samplenig(train_ds, scale_downsampling = 0.5)

        label_pos , label_neg , _ , _ = count_lablel(train_ds)
        print(f'up and down sampling => train positive label: {label_pos} - train negative label: {label_neg}')

        label_pos , label_neg, _ , _ = count_lablel(valid_ds)
        print(f'Test positive label: {label_pos} - Test negative label: {label_neg}')

        l_train = []
        r_train = []
        lbls_train = []
        l_valid = []
        r_valid = []
        lbls_valid = []

        for i , data in enumerate(train_ds):
            smiles, embbed_drug, onehot_task, embbed_task, lbl, task_name = data
            l_train.append(embbed_drug[0])
            r_train.append(embbed_task)
            lbls_train.append(lbl.tolist())

        for i , data in enumerate(valid_ds):
            smiles, embbed_drug, onehot_task, embbed_task, lbl, task_name = data
            l_valid.append(embbed_drug[0])
            r_valid.append(embbed_task)
            lbls_valid.append(lbl.tolist())

        l_train = np.array(l_train).reshape(-1,1024,1)
        r_train = np.array(r_train).reshape(-1,512,1)
        lbls_train = np.array(lbls_train)

        l_valid = np.array(l_valid).reshape(-1,1024,1)
        r_valid = np.array(r_valid).reshape(-1,512,1)
        lbls_valid = np.array(lbls_valid)

        # create neural network model
        siamese_net = siamese_model_attentiveFp_sider()

        history = History()
        P = siamese_net.fit([l_train, r_train], lbls_train, epochs = Epoch_S, batch_size = 128, callbacks=[history])

        for j in range(100):
            C=1
            Before = int(P.history['accuracy'][-1]*100)
            for i in range(2,Epoch_S+1):
                if  int(P.history['accuracy'][-i]*100) == Before:
                    C=C+1
                else:
                    C=1
                Before=int(P.history['accuracy'][-i]*100)
                print(Before)
            if C==Epoch_S:
                break
            P = siamese_net.fit([l_train, r_train], lbls_train, epochs = Epoch_S, batch_size = 128, callbacks=[history])
        print(j+1)

        score  = siamese_net.evaluate([l_valid,r_valid], lbls_valid, verbose=1)
        a = (score[1],score[4])
        result.append(a)

    return result

scores = evaluate_model(data_ds, subscriber_data_ds, 10, True)

"""#### Dropout = 0.3 and downsampling = 0.5"""

scores

acc = []
auc = []
for i in scores:
    acc.append(i[0])
    auc.append(i[1])

print(f'accuracy= {np.mean(acc)} AUC= {np.mean(auc)} STD_AUC= {np.std(auc)}')

"""# **Case study with BioAct-Het**"""

model_name = 'GCN_attentivefp_SIDER'
gcn_model = get_sider_model(model_name)
gcn_model.eval()
gcn_model = gcn_model.to(device)

sider_smiles = df.smiles.to_numpy()

dir_path = 'C:/Users/stdso/Documents/USTH/Med/BioAct-Het-main/Data'

df_case_study = pd.read_csv(dir_path + '/group2.csv')

df_case_study

drug_name = df_case_study.Drug_Name.to_numpy()

candidate_smiles = df_case_study.smiles.to_numpy()

is_Membership(sider_smiles, candidate_smiles)

sider_tasks = df.columns.values[1:28].tolist()
sider_tasks

print(df, sider_tasks)

df_positive, df_negative = separate_active_and_inactive_data(df, sider_tasks)

for i,d in enumerate(zip(df_positive,df_negative)):
    print(f'{sider_tasks[i]}=> positive: {len(d[0])} - negative: {len(d[1])}')

dataset_positive = [DATASET(d,smiles_to_bigraph, AttentiveFPAtomFeaturizer(), cache_file_path = cache_path_sider) for d in df_positive]
dataset_negative = [DATASET(d,smiles_to_bigraph, AttentiveFPAtomFeaturizer(), cache_file_path = cache_path_sider) for d in df_negative]

embed_class_sider = get_embedding_vector_class(dataset_positive, dataset_negative, radius=2, size = 512)

dataset = DATASET(df,smiles_to_bigraph, AttentiveFPAtomFeaturizer(), cache_file_path = cache_path_sider)
ds_train = create_dataset_with_gcn_case_study(dataset, embed_class_sider, gcn_model, sider_tasks)

dataset_study = DATASET(df_case_study[df_case_study.columns[1:29]],smiles_to_bigraph,
                        AttentiveFPAtomFeaturizer(), cache_file_path = cache_path_sider)

ds_study = create_dataset_with_gcn_case_study(dataset_study, embed_class_sider, gcn_model, sider_tasks)

len(data_ds)

"""### Training algorithm"""

Epoch_S = 15

l, r , lbls = data_generator(ds_train)

l = np.array(l).reshape(-1,1024,1)
r = np.array(r).reshape(-1,512,1)
lbls=np.array(lbls)

history = History()

siamese_net = siamese_model_attentiveFp_sider()


s = siamese_net.fit([l, r], lbls, epochs = Epoch_S, shuffle=True, batch_size=128, callbacks=[history])

for j in range(1000):
    C=1
    Before = int(s.history['accuracy'][-1]*100)
    for i in range(2,Epoch_S+1):
        if  int(s.history['accuracy'][-i]*100)== Before:
            C=C+1
        else:
            C=1
        Before=int(s.history['accuracy'][-i]*100)
        print(Before)
    if C==Epoch_S:
        break
    s = siamese_net.fit([l, r], lbls, epochs = Epoch_S, shuffle=True, batch_size=128, callbacks=history)
print(j+1)

"""### Model evaluation"""

valid_ds = {}

for i, task in enumerate(sider_tasks):
    temp = []
    for j , data in enumerate(ds_study):
        smiles, embbed_drug, onehot_task, embbed_task, lbl, task_name = data
        if task ==  task_name:
            temp.append(data)

    valid_ds[task] = temp

task_scores = [sider_tasks for sider_tasks in range(len(sider_tasks))]

for i, task in enumerate(sider_tasks):

    l_val = []
    r_val = []
    lbls_valid = []
    for data in valid_ds[task]:
        smiles, embbed_drug, onehot_task, embbed_task, lbl, task_name = data
        l_val.append(embbed_drug[0])
        r_val.append(embbed_task)
        lbls_valid.append(lbl)

    l1 = np.array(l_val)
    r1 = np.array(r_val)
    lbls_valid = np.array(lbls_valid)

    y_pred = siamese_net.predict([l1,r1])

    result = (y_pred)
    task_scores[i] = task, result
    print(task_scores)

for task in task_scores:
    print(" --------------------------------- ")
    print(F'{task[0]}:')
    for i, drug in enumerate(task[1]):
        print(F'{i+1}- {drug_name[i]}: {drug}')
