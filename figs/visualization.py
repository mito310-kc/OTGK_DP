################################################################################
################################# Visulization #################################
################################################################################
!pip install PyTDC
!pip install rdkit
!pip install py3Dmol

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import py3Dmol
import tdc
from tdc import utils
from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Tox
from tdc.single_pred import ADME
from tdc.single_pred import Develop
from tdc.single_pred import QM
import networkx as nx
import matplotlib.pyplot as plt

def smile_to_graph_vis(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

        ### add the feature vector at the atom ####
        atom_index = atom.GetIdx()
        custom_feature = "Feature_{}".format(atom_index)
        atom.SetProp(custom_feature, ','.join(map(str, feature.astype(int) ) ))


    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    gg = nx.Graph()
    gg.add_edges_from(edges)

    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index, mol , gg

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


utils.retrieve_dataset_names('Tox')    # (ADME, Tox)


#----- Model Inputs -------------
set_name ='herg'         # Choose dataset
sets = 'Tox'                       # The prediction task, (ADME, Tox)
drug_idx = 15
#-------------------------------------------------------------------------------

if sets == 'ADME':
   data = ADME(name = set_name)
elif sets == 'Tox':
   data = Tox(name = set_name)
elif sets == 'Develop':
    if set_name == 'tap':
        label_list = retrieve_label_name_list('TAP')
        data = Develop(name = 'TAP', label_name = label_list[0])
    else:
        data = Develop(name = set_name)
elif sets == 'QM':
    label_list = retrieve_label_name_list('QM7b')
    data = QM(name = 'QM7b', label_name = label_list[0])
else:
    data = HTS(name = set_name)   #sarscov2_3clpro_diamond ,sarscov2_vitro_touret


dall = data.get_data()
split = data.get_split(method = 'scaffold', seed = 42, frac = [0.8, 0.0, 0.2])
Train = split['train']
Test = split['test']




#=================== 2D #===================

s = Train['Drug'][drug_idx]
mol1 = Chem.MolFromSmiles(s)
img = Draw.MolToImage(mol1)

#mol = Chem.AddHs(mol)  # Add hydrogens for a complete 3D structure
AllChem.EmbedMolecule(mol1, randomSeed=42)  # Generate 3D coordinates

dpi = 300

# Generate a 2D depiction of the molecule with the specified DPI
img = Draw.MolToImage(mol1, size=(300, 300), kekulize=True, wedgeBonds=True, wedgeFontSize=9, dpi=dpi)

img.save("molecule2d.png")

# Optionally, you can also display the image
img




#=================== 3D #===================
s = Train['Drug'][drug_idx]
mol = Chem.MolFromSmiles(s)

# Convert the RDKit molecule to a PDB format string
pdb_block = Chem.MolToMolBlock(mol)

# Create a Py3Dmol view
viewer = py3Dmol.view(width=500, height=500)

# Add the 3D molecular structure from the PDB block
viewer.addModel(pdb_block, "mol")

# Style the molecular representation (e.g., sticks)
viewer.setStyle({"stick": {}})

# Zoom to fit the molecule
viewer.zoomTo()
viewer.show()

img_data = viewer.toImage(format='jpeg')

# Show the 3D visualization in a Jupyter notebook

#=================== 3D #===================


def MolTo3DView(mol, size=(300, 300), style="stick", surface=False, opacity=0.5):
    """Draw molecule in 3D

    Args:
    ----
        mol: rdMol, molecule to show
        size: tuple(int, int), canvas size
        style: str, type of drawing molecule
               style can be 'line', 'stick', 'sphere', 'carton'
        surface, bool, display SAS
        opacity, float, opacity of surface, range 0.0-1.0
    Return:
    ----
        viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    assert style in ('line', 'stick', 'sphere', 'carton')
    mblock = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(mblock, 'mol')
    viewer.setStyle({style:{}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer


def smi2conf(smiles):
    '''Convert SMILES to rdkit.Mol with 3D coordinates'''
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        return mol
    else:
        return None

smi = Train['Drug'][drug_idx]
#smi = 'COc3nc(OCc2ccc(C#N)c(c1ccc(C(=O)O)cc1)c2P(=O)(O)O)ccc3C[NH2+]CC(I)NC(=O)C(F)(Cl)Br'
conf = smi2conf(smi)
viewer = MolTo3DView(conf, size=(700, 500), style='sphere')
viewer.show()



#=================== 2D #===================

### Show the corresponding graph

smi = Train['Drug'][drug_idx]

c_size, features, edge_index, mol , g = smile_to_graph_vis(smi)

# Visualize the undirected graph
plt.figure(figsize=(8, 6))
nx.draw(g, with_labels=False, node_color='cornflowerblue',edge_color='cornflowerblue',width=4 , node_size=500, font_size=10)
plt.show()







#================== Rank Plot =========================
############################ PLOT RANK FIGURE ##################################

from scipy.stats import rankdata

# caco2 = [0.908, 0.393, 0.446, 0.599, 0.401, 0.546, 0.502, 0.398, 0.521, 0.368]
# sorted_indices_caco2 = sorted(range(len(caco2)), key=lambda i: caco2[i])

caco2 = [0.908, 0.393, 0.446, 0.599, 0.401, 0.546, 0.502, 0.390, 0.521, 0.368]
opposite_ranks_caco2 = list(rankdata(caco2).astype(int))

hiv = [0.807, 0.972, 0.869, 0.936, 0.974, 0.978, 0.975, 0.928, 0.911, 0.945]
sorted_indices_hiv = sorted(range(len(hiv)), key=lambda i: hiv[i])
opposite_ranks_hiv = [len(sorted_indices_hiv) - sorted_indices_hiv.index(i) for i in range(len(sorted_indices_hiv))]

Pgp = [0.880, 0.918, 0.908, 0.895, 0.892, 0.929, 0.923, 0.882, 0.830, 0.886]
sorted_indices_Pgp = sorted(range(len(Pgp)), key=lambda i: Pgp[i])
opposite_ranks_Pgp  = [len(sorted_indices_Pgp) - sorted_indices_Pgp.index(i) for i in range(len(sorted_indices_Pgp))]

# m = np.stack((opposite_ranks_caco2, opposite_ranks_hiv,opposite_ranks_Pgp ) , axis=1)


Bioav = [0.581, 0.672, 0.613, 0.566, 0.632, 0.577, 0.671, 0.748, 0.646, 0.76]
sorted_indices_Bioav = sorted(range(len(Bioav)), key=lambda i: Bioav[i])
opposite_ranks_Bioav  = [len(sorted_indices_Bioav) - sorted_indices_Bioav.index(i) for i in range(len(sorted_indices_Bioav))]


Lipo = [0.701, 0.574, 0.743, 0.541, 0.572, 0.547, 0.535, 0.809, 1,1]
opposite_ranks_Lipo = list(rankdata(Lipo).astype(int))


AqSol = [1.203, 0.827, 1.023, 0.907, 0.776, 1.026, 1.040, 0.992,10,10]
opposite_ranks_AqSol = list(rankdata(AqSol).astype(int))


PPBR = [12.848, 9.994, 11.106, 10.194, 9.373, 10.075, 9.445, 8.556, 8.733, 8.58]
opposite_ranks_PPBR = list(rankdata(PPBR).astype(int))


BBB = [0.823, 0.889, 0.781, 0.842, 0.855, 0.892, 0.897, 0.857,0,0]
sorted_indices_BBB = sorted(range(len(BBB)), key=lambda i: BBB[i])
opposite_ranks_BBB  = [len(sorted_indices_BBB) - sorted_indices_BBB.index(i) for i in range(len(sorted_indices_BBB))]


VD = [0.493, 0.561, 0.226, 0.457, 0.241, 0.559, 0.485, 0.722, 0.412, 0.729]
sorted_indices_VD = sorted(range(len(VD)), key=lambda j: VD[j])
opposite_ranks_VD  = [len(sorted_indices_VD) - sorted_indices_VD.index(i) for i in range(len(sorted_indices_VD))]


cyp2d6_s = [0.671, 0.677, 0.485, 0.617, 0.574, 0.704, 0.736, 0.784, 0.575, 0.814]
sorted_indices_cyp2d6_s = sorted(range(len(cyp2d6_s)), key=lambda j: cyp2d6_s[j])
opposite_ranks_cyp2d6_s  = [len(sorted_indices_cyp2d6_s) - sorted_indices_cyp2d6_s.index(i) for i in range(len(sorted_indices_cyp2d6_s))]


cyp3d4_s = [0.633, 0.639, 0.662, 0.590, 0.576, 0.582, 0.609, 0.641, 0.639, 0.651]
sorted_indices_cyp3d4_s = sorted(range(len(cyp3d4_s)), key=lambda j: cyp3d4_s[j])
opposite_ranks_cyp3d4_s  = [len(sorted_indices_cyp3d4_s) - sorted_indices_cyp3d4_s.index(i) for i in range(len(sorted_indices_cyp3d4_s))]


cyp2c9_s = [0.380, 0.360, 0.367, 0.344, 0.375, 0.381, 0.392, 0.448, 0.385, 0.47]
sorted_indices_cyp2c9_s = sorted(range(len(cyp2c9_s)), key=lambda j: cyp2c9_s[j])
opposite_ranks_cyp2c9_s  = [len(sorted_indices_cyp2c9_s) - sorted_indices_cyp2c9_s.index(i) for i in range(len(sorted_indices_cyp2c9_s))]



Half_Life = [0.329, 0.184, 0.038, 0.239 ,0.085, 0.151, 0.129, 0.372 ,0.269, 0.414]
sorted_indices_Half_Life = sorted(range(len(Half_Life)), key=lambda j: Half_Life[j])
opposite_ranks_Half_Life = [len(sorted_indices_Half_Life) - sorted_indices_Half_Life.index(i) for i in range(len(sorted_indices_Half_Life))]



CL_Micro = [0.492, 0.586, 0.252, 0.532, 0.365, 0.585, 0.578, 0.512, 0.533, 0.552]
sorted_indices_CL_Micro = sorted(range(len(CL_Micro)), key=lambda j: CL_Micro[j])
opposite_ranks_CL_Micro = [len(sorted_indices_CL_Micro) - sorted_indices_CL_Micro.index(i) for i in range(len(sorted_indices_CL_Micro))]


CL_Hepa = [0.272, 0.382, 0.235, 0.366, 0.289, 0.413, 0.439, 0.341, 0.314, 0.324]
sorted_indices_CL_Hepa = sorted(range(len(CL_Hepa)), key=lambda j: CL_Hepa[j])
opposite_ranks_CL_Hepa = [len(sorted_indices_CL_Hepa) - sorted_indices_CL_Hepa.index(i) for i in range(len(sorted_indices_CL_Hepa))]



hERG = [0.736, 0.841, 0.754, 0.738, 0.825, 0.778, 0.756, 0.779, 0.762, 0.853]
sorted_indices_hERG = sorted(range(len(hERG)), key=lambda j: hERG[j])
opposite_ranks_hERG = [len(sorted_indices_hERG) - sorted_indices_hERG.index(i) for i in range(len(sorted_indices_hERG))]


AMES = [0.794, 0.823, 0.776, 0.818, 0.814, 0.842, 0.837, 0.789,0,0]
sorted_indices_AMES = sorted(range(len(AMES)), key=lambda j: AMES[j])
opposite_ranks_AMES = [len(sorted_indices_AMES) - sorted_indices_AMES.index(i) for i in range(len(sorted_indices_AMES))]


DILI =[0.832, 0.875, 0.792, 0.859, 0.886, 0.919, 0.861, 0.887, 0.862, 0.904]
sorted_indices_DILI = sorted(range(len(DILI)), key=lambda j: DILI[j])
opposite_ranks_DILI = [len(sorted_indices_DILI) - sorted_indices_DILI.index(i) for i in range(len(sorted_indices_DILI))]


LD50 =[0.649,0.678, 0.675, 0.6491, 0.6781, 0.685, 0.669, 0.648,1,1]
opposite_ranks_LD50 = list(rankdata(LD50).astype(int))


ranks = np.asarray([opposite_ranks_caco2,
                    opposite_ranks_hiv,
                    opposite_ranks_Pgp,
                    opposite_ranks_Bioav,
                    opposite_ranks_Lipo,
                    opposite_ranks_AqSol,
                    opposite_ranks_PPBR,
                    opposite_ranks_BBB,
                    opposite_ranks_VD,
                    opposite_ranks_cyp2d6_s,
                    opposite_ranks_cyp3d4_s,
                    opposite_ranks_cyp2c9_s,
                    opposite_ranks_Half_Life,
                    opposite_ranks_CL_Micro,
                    opposite_ranks_CL_Hepa,
                    opposite_ranks_hERG,
                    opposite_ranks_AMES,
                    opposite_ranks_DILI,
                    opposite_ranks_LD50
                    ])


import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your actual data)
num_methods = 10
num_datasets = 19

# Generating random ranks for each method on 19 datasets (replace this with your actual data)

# Set colors for different methods
method_colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'darkblue']

# Create a ranked plot for each method across datasets
plt.figure(figsize=(8,8))
plt.grid(axis='y')

for dataset in range(num_datasets):
    # Scatter plot for each dataset's rank for all methods
    for method in range(num_methods):
        plt.scatter(ranks[dataset,method], dataset, color=method_colors[method], s=500,  label=f'Method {method}', alpha=0.7)

# Customize plot labels, titles, and legend


names_=  ['caco2','HIV','Pgp','Bioav','Lipo','AqSol','PPBR','BBB','VD','cyp2d6_s','cyp3d4_s','cyp2c9_s','Half_Life','CL-Micro','CL-Hepa','hERG','AMES','DILI','LD50']


plt.xlabel('Rank' , fontsize = 20)
plt.ylabel('Dataset' , fontsize =20 )

#plt.title('Ranked Plot of Methods Across Datasets')
plt.yticks(range(num_datasets), [i for i in names_])  # Setting y-tick labels

methods_labels = ['Morgan', 'RDKit2D', 'CNN', 'GCN', 'AttFP', 'AttrM', 'CPred', 'W', 'GW', 'FGW']

legend_handles = []
for i, color in enumerate(method_colors):
    legend_handles.append(plt.scatter([], [], color=color, label=methods_labels[i], alpha=0.7))

plt.legend(handles=legend_handles, title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(-1, 19.5)
plt.xticks(range(1,11))
plt.tight_layout()


#plt.savefig('/content/drive/MyDrive/Colab_Notebooks/Project_4_graph_matching/rank_plot.png', dpi=300)
plt.show()
