import numpy as np
import os, sys
from tdc.single_pred import ADME
from tdc.single_pred import Tox
from sklearn.svm import SVC
from sklearn.svm import SVR



from Gbuilder import *
from OTtools import *
from utils import *

def main():
    #----------- Model Inputs ------------------------
    set_name ='hia_hou'       # Choose dataset
    sets = 'ADME'                     # The prediction task, (ADME, Tox)
    task = 'classification'               # The type of the task to be done, (classification vs regression, classification2, regression2)
    #---------------- Specify the measure/embdgs + and the distance metrix used in the cost matrix ---------
    embdgs = False                    # If you apply an embedding scheme at the extracted features or not
    otsolver = 'wasserstein'         # The used optimal transport method, GWasserstein, wasserstein, FGW
    fm = 'euclidean'                  # The distance metrix to be applied in order to generate the cost matrix needed for optimal transport
    gamma = 3                        # The scaling parameter used in the kernel
    
    norm = False                      # Normalize the features on the nodes
    save_sim = False
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
    for key in results:
        print(f"{key}: {results[key]}")


    










if __name__ == "__main__":
    main()


