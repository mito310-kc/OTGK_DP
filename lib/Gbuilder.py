from rdkit import Chem
import networkx as nx
from rdkit.Chem import Descriptors
from scipy.sparse.csgraph import shortest_path
from rdkit.Chem.Fingerprints import FingerprintMols
import pandas as pd



'''
!pip install git+https://github.com/bp-kelley/descriptastorus

try:
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
except:
    raise ImportError("Please install pip install git+https://github.com/bp-kelley/descriptastorus and pip install pandas-flavor")
'''


def generate_embeddings_graph2(g,h):
    graph_feat = []

    for it in range(h+1):
        if it == 0:
            node_features = g.all_matrix_attr()
            graph_feat.append(node_features)
        else:
            adj_mat = nx.adjacency_matrix(g.nx_graph).todense()
            adj_cur = adj_mat + np.identity(adj_mat.shape[0])
            adj_cur = create_adj_avg(adj_cur)
            np.fill_diagonal(adj_cur, 0)

            graph_feat_cur = 0.5*(np.dot(adj_cur, graph_feat[it-1]) + graph_feat[it-1])
            emds2 = generate_embeddings_graph_lvl_two2(graph_feat_cur,adj_mat)
            graph_feat.append(emds2)

    return graph_feat[-1]


def generate_embeddings_graph_lvl_two2(emds,adj_mat):
    emd2 =[]
    for i in range(len(emds)):     # iterate through the nodes
        ws_index = list(np.nonzero(adj_mat[i])[0])             # get the index of ws
        ws = adj_mat[ws_index]         # get ws
        ngbs = []
        for t in ws:
            filters = [list(a*np.asarray(b)) for a,b in zip(t,emds)]
            res = list(np.sum(filters, 0)/np.sum(t))
            ngbs.append(res)
        emd2.append(list(np.asarray(list(np.sum(ngbs, 0)/len(ngbs)) + emds[i])/2))
    emds = np.asarray(emd2)
    return emds


def atom_features1(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

'''
atom_properties = {
        "AtomicNumber": atom.GetAtomicNum(),
        "AtomicWeight": Descriptors.ExactMolWt(atom.GetSymbol()),
        "Valence": atom.GetTotalValence(),
        "FormalCharge": atom.GetFormalCharge(),
        "NumHs": atom.GetTotalNumHs(),
        "Hybridization": atom.GetHybridization(),
        "Degree": atom.GetDegree(),
        "TotalDegree": atom.GetTotalDegree(),
        "NumNonHNeighbors": atom.GetNumNonHNeighbors(),
        "Chirality": atom.GetChiralTag(),
        "PartialCharge": atom.GetPropsAsDict().get("_GasteigerCharge", None),
        "Aromatic": atom.GetIsAromatic(),
        "HBondAcceptor": atom.GetIsHbondAcceptor(),
        "HBondDonor": atom.GetIsHbondDonor(),
        "AtomType": atom.GetPropsAsDict().get("_TriposAtomType", None),
        "NumRadicalElectrons": atom.GetNumRadicalElectrons(),
        "IsInRing": atom.IsInRing()

        # Add more properties as needed
}


def atom_features(atom):
    return [atom.GetAtomicNum(),
                     atom.GetTotalValence(),
                     atom.GetFormalCharge(),
                     atom.GetTotalNumHs(),
                     #atom.GetHybridization(),
                     atom.GetDegree(),
                     atom.GetTotalDegree(),
                     #atom.GetChiralTag(),
                     atom.GetPropsAsDict().get("_GasteigerCharge", 0),
                     int(atom.GetIsAromatic()),
                     atom.GetPropsAsDict().get("_TriposAtomType", 0),
                     atom.GetImplicitValence(),
                     atom.GetNumRadicalElectrons(),
                     int(atom.IsInRing()),
                     ]
'''


# def atom_features(atom):
# return np.array([atom.GetDegree() , atom.GetTotalNumHs(), atom.GetImplicitValence(), atom.GetIsAromatic()])
def atom_features(atom):
    return np.array([atom.GetDegree(),
                     atom.GetTotalNumHs(),
                     atom.GetImplicitValence(),
                     int(atom.GetIsAromatic()),
                     #atom.GetAtomicNum(),
                     #atom.GetTotalValence(),
                     #atom.GetTotalDegree(),
                     #atom.GetFormalCharge(),
                     #atom.GetPropsAsDict().get("_GasteigerCharge", 0),
                     #atom.GetPropsAsDict().get("_TriposAtomType", 0),
                     #atom.GetNumRadicalElectrons(),
                     #int(atom.IsInRing())
                     ])


def smile_to_myfeatures(smile, normalize= False):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        if normalize:
            features.append(feature/ sum(feature))
        else:
            features.append(feature)

    return features


def smile_to_graph2(smile, normalize= False):

    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    g = nx.Graph()

    for atom in mol.GetAtoms():
        ### add the feature vector at the atom ####
        feature = atom_features(atom)
        if normalize:
            features.append(feature/sum(feature))
        else:
            features.append(feature)

        atom_idx = atom.GetIdx()
        g.add_node(atom_idx)

    # edges = []
    # g = nx.Graph()
    for bond in mol.GetBonds():
        #edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    nodes_del = []

    for node in g:
        if g.degree(node) == 0:
           nodes_del.append(node)



    for indx in nodes_del:
       g.remove_node(indx)

    F = [element for index, element in enumerate(features) if index not in nodes_del]



    # g = nx.Graph(edges).to_directed()

    return g, F, c_size, mol


def structure_matrix(g, method='shortest_path'):
    A=nx.adjacency_matrix(g)

    if method =='shortest_path':
        #A = A.astype(int)
        A = A.toarray()
        C = shortest_path(A)

    if method =='harmonic_distance':
        A=A.astype(np.float32)
        D=np.sum(A,axis=0)
        L=np.diag(D)-A

        ones_vector=np.ones(L.shape[0])
        fL=np.linalg.pinv(L)

        C=np.outer(np.diag(fL),ones_vector)+np.outer(ones_vector,np.diag(fL))-2*fL
        C=np.array(C)

    if method=='adjency':
        C = A.toarray()

    return C



def smiles2daylight(s):
    try:
        NumFinger = 2048
        mol = Chem.MolFromSmiles(s)
        bv = FingerprintMols.FingerprintMol(mol)
        temp = tuple(bv.GetOnBits())
        features = np.zeros((NumFinger, ))
        features[np.array(temp)] = 1
    except:
        print('rdkit not found this smiles: ' + s + ' convert to all 0 features')
        features = np.zeros((2048, ))
    return np.array(features)
 




def smiles2rdkit2d(s):
    try:
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = np.array(generator.process(s)[1:])
        NaNs = np.isnan(features)
        features[NaNs] = 0
    except:
        print('descriptastorus not found this smiles: ' + s + ' convert to all 0 features')
        features = np.zeros((200, ))
    return np.array(features)

def extract_features(X_drug, y, discriptor = 'rdkit2d'):
    df_data = pd.DataFrame(zip(X_drug, y))
    df_data.rename(columns={0:'SMILES',1: 'Label'}, inplace=True)

    if discriptor == 'rdkit2d':
        unique = pd.Series(df_data['SMILES'].unique()).apply(smiles2rdkit2d)
        unique_dict = dict(zip(df_data['SMILES'].unique(), unique))
        df_data['Extracted_features'] = [unique_dict[i] for i in df_data['SMILES']]

    elif discriptor == 'daylight':                  # this is a fingerprint
        unique = pd.Series(df_data['SMILES'].unique()).apply(smiles2daylight)
        unique_dict = dict(zip(df_data['SMILES'].unique(), unique))
        df_data['Extracted_features'] = [unique_dict[i] for i in df_data['SMILES']]

    elif discriptor == 'myfeatures':
        unique = pd.Series(df_data['SMILES'].unique()).apply(smile_to_myfeatures)
        unique_dict = dict(zip(df_data['SMILES'].unique(), unique))
        df_data['Extracted_features'] = [unique_dict[i] for i in df_data['SMILES']]

    return df_data.reset_index(drop=True)





#------------------- extract graph features ----------------
#drug_encoding = 'myfeatures'     # daylight: smiles2daylight , rdkit2d: smiles2rdkit2d , myfeatures

#train = extract_features(X_drug = Train.Drug.values, y = Train.Y.values,discriptor = drug_encoding)


