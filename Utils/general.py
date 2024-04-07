import dgl
import random
import pandas as pd
import numpy as np

from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit import DataStructs

from dgllife.data import SIDER, Tox21, MUV
from numpy import array
from numpy import argmax
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, AttentiveFPAtomFeaturizer
from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive
from dgllife.data.csv_dataset import MoleculeCSVDataset
from dgllife.utils.mol_to_graph import smiles_to_bigraph

from Utils.gcnpretrained import get_fingerprint


class DATASET(MoleculeCSVDataset):

    def __init__(self,dataFram=None,
                 smiles_to_graph=smiles_to_bigraph,
                 node_featurizer=None,
                 edge_featurizer=None,
                 load=False,
                 log_every=1000,
                 cache_file_path = None,
                 n_jobs=1):


        super(DATASET, self).__init__(df=dataFram,
                                    smiles_to_graph=smiles_to_graph,
                                    node_featurizer=node_featurizer,
                                    edge_featurizer=edge_featurizer,
                                    smiles_column='smiles',
                                    cache_file_path = cache_file_path,
                                    load=load,
                                    log_every=log_every,
                                    init_mask=True,
                                    n_jobs=n_jobs)

    def __getitem__(self, item):

        return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]

def get_dataset(dataset_name = "sider"):

    if dataset_name == "tox21":
        _url = f'dataset/{dataset_name}.csv.gz'
        data_path = get_download_dir() + '/tox21.csv.gz'
        download(_get_dgl_url(_url), path=data_path, overwrite=False)
        df = pd.read_csv(data_path)

    else:
        url = f'dataset/{dataset_name}.zip'
        data_path = get_download_dir() + f'/{dataset_name}.zip'
        dir_path = get_download_dir() + f'/{dataset_name}'
        download(_get_dgl_url(url), path = data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + f'/{dataset_name}.csv')

    return df

def del_nan (datafarme,  tasks):
    dataset = []
    for task in tasks:
        a = datafarme[['smiles' , task]]
        a = a.dropna()
        dataset.append(a)

    return dataset

def separate_active_and_inactive_data (datafarme,  tasks):
    dataset_pos = []
    dataset_neg = []

    for task in tasks:
        a = datafarme[['smiles' , task]]
        b = a.loc[a[task]==0]
        a = a.loc[a[task]==1]
        dataset_pos.append(a)
        dataset_neg.append(b)

    return dataset_pos, dataset_neg

def get_embedding_vector_class(psitive_datafarme, negative_datafarame, radius=3, size = 1024 ):
    epsilon= 0.00001
    created_class_vector = []

    for i , df in enumerate(zip(psitive_datafarme, negative_datafarame)):

        temp_psitive = []
        temp_negative = []

        for data_pos in df[0]:
            smiles, g, label, mask = data_pos
            fingerprin_vector_psitive = get_fingerprint(smiles, radius, size)
            temp_psitive.append(fingerprin_vector_psitive)

        for data_neg in df[1]:
            smiles, g, label, mask = data_neg
            fingerprin_vector_negative = get_fingerprint(smiles, radius, size)
            temp_negative.append(fingerprin_vector_negative)

        psitive_vector = np.divide(np.sum(np.array(temp_psitive), axis = 0), len(df[0])) + epsilon
        negative_vector = np.divide(np.sum(np.array(temp_negative), axis = 0), len(df[1])) + epsilon

        created_class_vector.append(np.log(np.divide(psitive_vector, negative_vector)))

    print('class vector created!!')
    return created_class_vector

def count_lablel(dataset):

    label_pos = 0
    lable_neg = 0

    data_pos = []
    data_neg = []

    for i, data in enumerate(dataset):
        smiles, embbed_drug, embbed_task, lbl, task_number, task_name= data
        if (lbl == 1.).any():
            label_pos += 1
            data_pos.append(data)
        else:
            lable_neg += 1
            data_neg.append(data)

    return label_pos, lable_neg , data_pos, data_neg

def up_and_down_Samplenig(dataset, sacale_upsampling = 2, scale_downsampling = 0.7):
    data_up_pos = []
    data_down_neg = []
    pos, neg , data_pos, data_neg = count_lablel(dataset)

    upsample = int((neg // pos) // sacale_upsampling)
    downsample = int(neg * scale_downsampling)

    data_up_pos = upSamplenigData(data_pos, upsample)
    data_down_neg = random.sample(data_neg, downsample)
    data_up_pos.extend(data_neg)

    return data_up_pos


def upSamplenigData(dataset, scale):
    data_new = []

    for i , data in enumerate(dataset):
        embbed_drug, onehot_task, embbed_task, lbl, task_name = data
        for i in range(scale):
            data_new.append(data)

    return data_new

def data_generator(ds):

    l = []
    r = []
    lbls = []

    for i, data in enumerate(ds):
        smiles, embbed_drug, onehot_task, embbed_task, lbl, task_name = data
        l.append(embbed_drug[0])
        r.append(embbed_task)
        lbls.append(lbl.tolist())

    return l, r, lbls