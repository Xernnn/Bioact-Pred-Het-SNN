# -*- coding: utf-8 -*-

import random
import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax

def select_data_support_set(dataset, size, positve, negative, task):

    support_set = []
    for i in range(size):
        temp_list = []

        for p in range(positve):
            """
            smile => d[0]
            embbed => d[1]
            onehot_task => d[2]
            embbed_task => d[3]
            lbl => d[4]
            task_name  => d[5]
            """
            pos = random.choice([d for d in dataset if d[5] == task and d[4] == 1.])
            temp_list.append(pos)

        for n in range(negative):
            """
            smile => d[0]
            embbed => d[1]
            onehot_task => d[2]
            embbed_task => d[3]
            lbl => d[4]
            task_name  => d[5]
            """
            neg = random.choice([d for d in dataset if d[5] == task and d[4] == 0.])
            temp_list.append(neg)

        support_set.append(temp_list)

    return support_set

def is_exist(item , listDrug):
    for data in listDrug:
        if item[0] == data[0] and item[5] == data[5] and item[4]==data[4]:
            print('true')
            return True

    return False

def create_support_set(ds, size, pos, neg, tasks):
    suport_set = {}
    for i, task in enumerate(tasks):
        s = select_data_support_set(ds, size, pos, neg, task)
        suport_set[task] = s
    return suport_set

def evaluation(model, tasks_test, suport_set ):

    task_scores = [tasks_test for tasks_test in range(len(tasks_test))]

    for i, task in enumerate(tasks_test):
        acc = []
        auc = []
        for data in suport_set[task]:
            y_test = []
            l_val = []
            r_val = []
            lbls_valid = []
            for d in data:
                smiles, embbed_drug, onehot_task, embbed_task, lbl, task_name = d
                l_val.append(embbed_drug[0])
                r_val.append(embbed_task)
                lbls_valid.append(lbl)

            l1 = np.array(l_val)
            r1 = np.array(r_val)
            lbls_valid = np.array(lbls_valid)

            score = model.evaluate([l1,r1], lbls_valid, verbose=0)
            acc.append(score[1])
            auc.append(score[4])

        result = (np.mean(acc), statistics.pstdev(acc), np.mean(auc), statistics.pstdev(auc))

        task_scores[i] = task, result

    return task_scores

def is_Membership(smiles, candidate_smiles):
    for s in smiles:
        if s in candidate_smiles:
            print("true: " , s)
            return True
        else:
            return False