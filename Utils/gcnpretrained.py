# -*- coding: utf-8 -*-

import numpy as np

import dgl
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit import DataStructs
from dgllife.model import load_pretrained
from dgllife.model import MLPPredictor

def get_sider_model (model_name = "GCN_attentivefp_SIDER"):
    if model_name ==  "GCN_attentivefp_SIDER":
        predictor_dropout = 0.08333992387843633
        predictor_hidden_feats = 1024
    else:
        predictor_dropout =  0.034959769945995006
        predictor_hidden_feats = 512

    gcn = load_pretrained(model_name)
    gnn_out_feats = gcn.gnn.hidden_feats[-1]
    gcn.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats, predictor_hidden_feats, predictor_dropout)

    return gcn

def get_muv_model (model_name = "GCN_attentivefp_MUV"):
    if model_name ==  "GCN_attentivefp_MUV":
        predictor_dropout = 0.24997398695768708
        predictor_hidden_feats = 128
    else:
         predictor_dropout = 0.10811886971338101
         predictor_hidden_feats = 128

    gcn = load_pretrained(model_name)
    gnn_out_feats = gcn.gnn.hidden_feats[-1]
    gcn.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats, predictor_hidden_feats, predictor_dropout)

    return gcn

def get_tox21_model (model_name = "GCN_attentivefp_Tox21"):
    if model_name ==  "GCN_attentivefp_Tox21":
        predictor_dropout = 0.5432104441360837
        predictor_hidden_feats = 512
    else:
         predictor_dropout =  0.18118350615245202
         predictor_hidden_feats = 128

    gcn = load_pretrained(model_name)
    gnn_out_feats = gcn.gnn.hidden_feats[-1]
    gcn.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats, predictor_hidden_feats, predictor_dropout)

    return gcn

def get_fingerprint(smile, radius = 2, size = 1024 ):

    smile_convert = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(smile_convert, radius , size)
    fingerprin_vector = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, fingerprin_vector)

    return fingerprin_vector