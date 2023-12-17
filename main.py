import numpy as np
import os, sys
from tdc.single_pred import ADME
from tdc.single_pred import Tox
from sklearn.svm import SVC
from sklearn.svm import SVR
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


from models import KNNClassifier
from models import KernelRegressionNN

from Gbuilder import *
from OTtools import *
from utils import *

def main():
    #----------- Model Inputs ------------------------
    set_name ='half_life_obach'       # Choose dataset
    sets = 'ADME'                     # The prediction task, (ADME, Tox)
    task = 'regression'               # The type of the task to be done, (classification vs regression)
    #---------------- Specify the measure/embdgs + and the distance metrix used in the cost matrix ---------
    embdgs = False                    # If you apply an embedding scheme at the extracted features or not
    otsolver = 'wasserstein'         # The used optimal transport method, GWasserstein, wasserstein, FGW
    fm = 'euclidean'                  # The distance metrix to be applied in order to generate the cost matrix needed for optimal transport
    gamma = 3                        # The scaling parameter used in the kernel
    
    norm = False                      # Normalize the features on the nodes
    save_sim = False
    run_MLP = True
    #-------------------------------------------------------------------------------
    if sets == 'ADME':
       data = ADME(name = set_name)
    else:
       data = Tox(name = set_name)
    
    dall = data.get_data()
    split = data.get_split(method = 'scaffold', seed = 42, frac = [0.8, 0.0, 0.2])
    Train = split['train']
    Test = split['test']
    y_train = list(Train['Y'])
    y_test = list(Test['Y'])
    print('===========================================')
    print(f'Number of data points = {len(y_train) + len(y_test)}')
    if task == 'classification':
        print(f'Percent of positive data points = {np.sum(y_train) / len(y_train) * 100:.2f}%')

    #------------------- extract graph features ---------------------
    compound_iso_smiles = list(dall['Drug'])
    smile_graph = {}                      # Dictinary of features
    structure_graph = {}
    for smile in compound_iso_smiles:
        g, feat_, _, _  = smile_to_graph2(smile, normalize= norm)
        smile_graph[smile] = feat_
        if otsolver != 'Wasserstein':
            #g, feat_, _, _  = smile_to_graph2(smile, normalize= norm)
            C = structure_matrix(g, method='adjency')     # harmonic_distance, adjency, shortest_path
            structure_graph[smile] = C
    
    ############################# get the similarity matrix for the training set ############################
    unique_drug1 = np.asarray(list(Train['Drug']))
    unique_drug2 = unique_drug1

    print('------------ Compute Similarity matrix required for training')

    M = compute_distances_btw_graphs_with_embdgs(unique_drug1,unique_drug2,smile_graph = smile_graph,structure_graph = structure_graph , embdgs = embdgs, fm = fm, otsolver=otsolver)
    
    if save_sim:
        np.save('TDC_M_train_FGW_'+ str(sets) + '_' + str(set_name) +'.npy', M)
    
    ################################ get the matrix for the testing part ####################################
    print('------------ Compute Similarity matrix required for testing')

    unique_drug1_test = np.asarray(list(Test['Drug']))
    M_test = compute_distances_btw_graphs_with_embdgs(unique_drug1_test,unique_drug2,smile_graph = smile_graph,structure_graph = structure_graph, embdgs = embdgs, fm = fm, otsolver=otsolver, Train=False)
    
    if save_sim:
        np.save('TDC_M_test_FGW_'+ str(sets) + '_' + str(set_name) +'.npy', M)
 

    ################## Learning PArt #################
    Z=np.exp(-gamma*(M))
    if not assert_all_finite(Z):
        Z = np.nan_to_num(Z, nan=0)
        print('There is Nan in M')
        #raise InfiniteException('There is Nan')

    #### get the kernel matrix for the testing part
    Z_test=np.exp(-gamma*(M_test))
    if not assert_all_finite(Z_test):
        Z_test = np.nan_to_num(Z_test, nan=0)
        print('There is Nan in M_test')
        #raise InfiniteException('There is Nan')


    #### Train, text and evaluate using SVM (classification) or SVR (regression)

    if task == 'classification':
        ##--------------- Classification using SVM
        # train
        C=1
        verbose = False
        svc=SVC(C=C,kernel="precomputed",verbose=verbose,max_iter=10000000, probability=True)
        classes_ =np.array(y_train)
        svc.fit(Z, classes_)
        # test
        preds_t =svc.predict(Z_test)
        y_preds0 = svc.predict_proba(Z_test)
        y_preds = y_preds0[:, 1]
    

    elif task == 'regression':
        #--------------- Regression using SVR
        C=1
        verbose = False
        # train
        svr = SVR(kernel="precomputed")
        svr.fit(Z, np.array(y_train))
        # test
        y_preds = svr.predict(Z_test)


    results = evaluate(y_preds,y_test, task)
    print(f'--------------- SV {task} Results---------------')
    for key in results:
        print(f"{key}: {results[key]}")


    if run_MLP:

        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(y_train)

        # Convert data to PyTorch tensors
        X_train = torch.Tensor(Z)
        X_test = torch.Tensor(Z_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        
        num_epochs = 200
        batch_size = 20


        if task =="classification":
            # Create the model
            input_size = M.shape[1]  # Assuming input size is the number of features
            num_classes = len(np.unique(labels_encoded))
            model = KNNClassifier(input_size, num_classes)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(num_epochs):
                for i in range(0, len(X_train), batch_size):
                    inputs = X_train[i:i+batch_size]
                    targets = y_train[i:i+batch_size]

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                _,predicted = torch.max(outputs, 1)

        else:
            similarity_matrix = torch.Tensor(Z)
            test_data = torch.Tensor(Z_test)
            y_test = torch.Tensor(y_test)
            targets = torch.Tensor(y_train)
            input_size = similarity_matrix.size(1)

            dataset = TensorDataset(similarity_matrix, targets)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


            model = KernelRegressionNN(input_size, 1)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            for epoch in range(num_epochs):
                for inputs, targets in dataloader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs.float(), targets.unsqueeze(1).float())  # Ensure targets have the right shape
                    loss.backward()

                    optimizer.step()

            predicted = model(test_data)
            y_test = y_test.tolist()
            predicted = predicted.tolist()


        results = evaluate(predicted,y_test, task)
        print(f'--------------- MLP {task} Results---------------')
        for key in results:
            print(f"{key}: {results[key]}")





if __name__ == "__main__":
    main()


