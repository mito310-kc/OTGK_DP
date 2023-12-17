import ot
import ot.plot
from ot.gromov import gromov_wasserstein, fused_gromov_wasserstein
import time

from itertools import islice

import pandas as pd
import numpy as np

from Gbuilder import *


def compute_distances_btw_graphs_with_embdgs(X,Y,fm, smile_graph, otsolver, structure_graph, embdgs =False, Train = True):

    X = X.reshape(X.shape[0],)
    Y = Y.reshape(Y.shape[0],)
    if Train:
        M = np.zeros((X.shape[0], Y.shape[0]))      # similarity matrix
        Q = np.zeros((X.shape[0], Y.shape[0]))
        for i, x1 in enumerate(X):
            for j,x2 in enumerate(Y):
                if j>=i:
                    dist_ = compute_matching_similarity_embds(x1, x2, dist="uniform", smile_graph = smile_graph, structure_graph=structure_graph, features_metric=fm, alg=otsolver , embdgs = embdgs )
                    M[i, j] = dist_
            if i % 20 == 0:
                print(f'Processed {i} graphs out of {len(X)}')

        np.fill_diagonal(Q,np.diagonal(M))     # this is to avoid
        M = M+M.T-Q

    else:
        M = np.zeros((X.shape[0], Y.shape[0]))
        for i, x1 in enumerate(X):
            row=[compute_matching_similarity_embds(x1, x2, dist="uniform", smile_graph = smile_graph, structure_graph=structure_graph, features_metric=fm, alg=otsolver , embdgs = embdgs) for j,x2 in enumerate(Y)]
            M[i,:]=row

            if i % 20 == 0:
                print(f'Processed {i} graphs out of {len(X)}')

    M[np.abs(M)<=1e-15]=0 #threshold due to numerical precision
    return M



def compute_matching_similarity_embds(g1,g2, smile_graph,structure_graph, dist = "uniform", features_metric ='hamming', embdgs = False , discrete = True, alg = 'wasserstein', sinkhorn_lambda=1e-2 ):

    embd_lvl = 1

    features1 = smile_graph[g1]
    features2 = smile_graph[g2]
    nodes1 = len(features1)
    nodes2 = len(features2)

    if structure_graph:
        C1 = structure_graph[g1]
        C2 = structure_graph[g2]
        #nodes1 = len(C1)
        #nodes2 = len(C2)

    h = 5
    if embdgs:
        if embd_lvl == 1:
            embd1 = generate_embeddings_graph(g1,h)
            embd2 = generate_embeddings_graph(g2,h)
        else:
            embd1 = generate_embeddings_graph2(g1,h)
            embd2 = generate_embeddings_graph2(g2,h)
    else:
        embd1 = np.asarray([list(array1) for array1 in features1])
        embd2 = np.asarray([list(array2) for array2 in features2])


    startstruct=time.time()

    if dist == "uniform":
        t1masses = np.ones(nodes1)/(nodes1)
        t2masses = np.ones(nodes2)/(nodes2)

    elif dist == "node_cent":
        A1=nx.adjacency_matrix(g1.nx_graph)
        D1=np.sum(A1,axis=0)
        A2=nx.adjacency_matrix(g2.nx_graph)
        D2=np.sum(A2,axis=0)

        t1masses = D1/sum(D1)
        t2masses = D2/sum(D2)

    elif dist == "btws":
        centrality1 = nx.betweenness_centrality(g1.nx_graph, normalized=True, k=len(g1.nodes())-1)
        centrality2 = nx.betweenness_centrality(g2.nx_graph, normalized=True, k=len(g2.nodes())-1)

        t1masses = [x/sum(list(centrality1.values())) for x in list(centrality1.values())]
        t2masses = [x/sum(list(centrality2.values())) for x in list(centrality2.values())]

    #ground_distance = 'hamming' if discrete else 'euclidean'

    if features_metric=='euclidean':
        costs = ot.dist(embd1, embd2, metric=features_metric)

    elif features_metric=='hamming':
        f = lambda x,y: hamming_dist(x,y)
        costs = ot.dist(embd1, embd2, metric=f)

    if alg == "sinkhorn":
        # mat = ot.sinkhorn( t1masses  , t2masses, costs, sinkhorn_lambda,  numItermax=50)
        # dist_ = np.sum(np.multiply(mat, costs))
        dist_ = ot.sinkhorn2( t1masses  , t2masses, costs, sinkhorn_lambda,  numItermax=50)

    elif alg == "pWasserstein":
        p = np.ones(len(nodes1))
        q = np.ones(len(nodes2))
        w0, log0 = ot.partial.partial_wasserstein(p, q, costs, m=1, log=True)
        dist_ = log0['partial_w_dist']

    elif alg == "pGWasserstein":
        C1 = ot.dist(embd1, embd1, metric=features_metric)
        C2 = ot.dist(embd2, embd2, metric=features_metric)
        p = np.ones(len(nodes1))
        q = np.ones(len(nodes2))
        res0, log0 = ot.partial.partial_gromov_wasserstein(C1, C2, p, q, m=1, log=True)
        dist_ = log0['partial_gw_dist']

    elif alg == "GWasserstein":

        #C1 = ot.dist(embd1, embd1, metric=features_metric)
        #C2 = ot.dist(embd2, embd2, metric=features_metric)

        gw0, log0 = ot.gromov.gromov_wasserstein(C1, C2, t1masses, t2masses, 'square_loss', verbose=False, log=True)
        dist_ = log0['gw_dist']

    elif alg == "FGW":
        alpha = 1e-3
        #print("SMILE = ", g1)

        #print("SMILE = ", g2)
        '''
        print("Nodes1 = ", nodes1)
        print("Nodes2 = ", nodes2)
        print("C1 = ", C1.shape)
        print("C2 = ", C2.shape)
        print("SMILE = ", g2)
        print("t1masses = ", len(t1masses))
        print("t2masses = ", len(t2masses))
        print("Features1 = ", len(features1))
        print("Features2 = ", len(features2))
        '''
        Gwg, logw = fused_gromov_wasserstein(costs, C1, C2, t1masses, t2masses, loss_fun='square_loss', alpha=alpha, verbose=False, log=True)
        print(logw)
        dist_ = logw['fgw_dist']

    elif alg == 'stochasticGWasserstein':
        C1 = ot.dist(embd1, embd1, metric=features_metric)
        C2 = ot.dist(embd2, embd2, metric=features_metric)
        pgw, plog = ot.gromov.pointwise_gromov_wasserstein(C1, C2, t1masses, t2masses, loss, max_iter=100, log=True)
        dist_ = plog['gw_dist_estimated']
    else:
        dist_ = ot.emd2(t1masses,t2masses , costs)

    return dist_
