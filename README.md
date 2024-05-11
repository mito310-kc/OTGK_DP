# Optimal Transport Based Graph Kernels for Drug Property Prediction (OTGK_DP)

This is the complete code for the OTGK_DP framework to predict ADMET durg properties. The work is pubished in the following papers: <br />
 <br />
Comments/Bugs/Problems: maburidi@ucmerced.edu  maburidi@gmail.com  <br />
[OTGK](https://ieeexplore.ieee.org/abstract/document/10504311))

```
@INPROCEEDINGS{10504311,
  author={Aburidi, Mohammed and Marcia, Roummel},
  booktitle={2024 IEEE First International Conference on Artificial Intelligence for Medicine, Health and Care (AIMHC)}, 
  title={Wasserstein Distance-Based Graph Kernel for Enhancing Drug Safety and Efficacy Prediction *}, 
  year={2024},
  volume={},
  number={},
  pages={113-119},
  keywords={Drugs;Proteins;Adaptation models;Toxicology;Pharmacodynamics;Predictive models;Safety;Optimal Transport;Wasserstein Distance;Graph Matching;Drug Discovery;ADMET Properties},
  doi={10.1109/AIMHC59811.2024.00029}}
```



December, 2023. Initial release <br />


<br />
<br />


<img width="1103" alt="main_fig" src="https://github.com/Maburidi/OTGK_DP/assets/48891624/a52d6c2f-9334-4af2-b596-b3213db96935">

<br />
<br />
<br />
<br />


Find the Colab tutorial in the tutorial folder to run this repository and to predict the drug properties. <br /> 
The data used in this project is   [Therapeutics Data Commons - TDC](https://tdcommons.ai/)) <br /> 
To list all of the datasets in AMDE, run the following 

```
from tdc import utils
utils.retrieve_dataset_names('ADME')
```

To list all of the datasets in TOX, run the following 

```
from tdc import utils
utils.retrieve_dataset_names('TOX')
```



Make sure to install the following dependincies:  

```
PyTDC
rdkit
torch_geometric
POT
py3Dmol

```



 <br /> 
  <br /> 
   <br /> 


## Main Results: 

<img width="911" alt="Screen Shot 2023-12-18 at 10 33 32 AM" src="https://github.com/Maburidi/OTGK_DP/assets/48891624/174fa587-06e1-46c3-b32b-399fdea34d2c">



 <br /> 
  <br /> 
   <br /> 



Cite as:
```
@inproceedings{aburidi2024_1,
   author = {M. Aburidi and R. Marica},
   journal = {Scientific Reports},
   title = {Optimal Transport-Based Graph Kernels for Drug Property Prediction},
   url = {},
   year = {2024},
}



@inproceedings{aburidi2024_2,
   author = {M. Aburidi and R. Marica},
   journal = {First IEEE International Conference on AI for Medicine, Health, and Care. AIMHC 2024},
   title = {Enhancing Drug Safety and Efficacy: Wasserstein Distance-Based Graph Kernel for Drug Property Prediction}, 
   url = {},
   year = {2024},
}
```

