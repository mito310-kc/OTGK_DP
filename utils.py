import numpy as np
from tdc import Evaluator


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))



def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def create_adj_avg(adj_cur):
    '''
    create adjacency
    '''
    deg = np.sum(adj_cur, axis = 1)
    deg = np.asarray(deg).reshape(-1)

    deg[deg!=1] -= 1

    deg = 1/deg
    deg_mat = np.diag(deg)
    adj_cur = adj_cur.dot(deg_mat.T).T

    return adj_cur

def assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    a=X.dtype.char in np.typecodes['AllFloat']
    b=np.isfinite(X.sum())
    c=np.isfinite(X).all()

    if (a and not b and not c):
        return False
    else :
        return True


class InfiniteException(Exception):
    pass

def hamming_dist(x,y):
    #print('x',len(x[-1]))
    #print('y',len(y[-1]))
    return len([i for i, j in zip(x, y) if i != j])





def evaluate(y_preds,y_test, task):
    if task == 'classification':
        evaluator1 = Evaluator(name = 'PR-AUC')
        score1 =  evaluator1(y_test, y_preds)
        evaluator2 = Evaluator(name = 'ROC-AUC')
        score2 =  evaluator2(y_test, y_preds)
        results = {'PR-AUC':score1 , 'ROC-AUC': score2}

    elif task == 'regression':
        evaluator1 = Evaluator(name = 'spearman')
        score1 = evaluator1(y_test, y_preds)
        evaluator2 = Evaluator(name = 'MAE')
        score2 = evaluator2(y_test, y_preds)
        results = {'Spearman':score1 , 'MAE': score2}
    return results




 

