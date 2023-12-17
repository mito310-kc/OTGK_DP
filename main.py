import os, sys

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




if __name__ == "__main__":
    main()
