import numpy as np
import os, sys
from tdc.single_pred import ADME
from tdc.single_pred import Tox
import Gbuilder 


def main():
    #----------- Model Inputs ------------------------
    set_name ='ppbr_az'       # Choose dataset
    sets = 'ADME'                     # The prediction task, (ADME, Tox)
    task = 'regression'               # The type of the task to be done, (classification vs regression, classification2, regression2)
    #---------------- Specify the measure/embdgs + and the distance metrix used in the cost matrix ---------
    embdgs = False                    # If you apply an embedding scheme at the extracted features or not
    otsolver = 'FGW'         # The used optimal transport method, GWasserstein, wasserstein, FGW
    fm = 'euclidean'                  # The distance metrix to be applied in order to generate the cost matrix needed for optimal transport
    gamma = 12                        # The scaling parameter used in the kernel
    
    norm = False                      # Normalize the features on the nodes
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








if __name__ == "__main__":
    main()


