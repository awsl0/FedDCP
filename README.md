# Introduction

This is the implementation of our paper 《FedDCP: Personalized Federated Learning Based
on Dual Classifiers and Prototypes》 (accepted by PRCV 2024).  We show the code of the representative FedDCP (`FedDCP`).


# Dataset

Due to the file size limitation, we only upload the fmnist dataset with the default practical setting ($\beta=0.1$). Please refer to our project [PFLlib](https://github.com/TsingZ0/PFLlib). 


# System (based on PFL-Non-IID)

- `main.py`: configurations of **FedDCP**.
- `env_linux.yaml`: python environment to run **FedDCP** on Linux. 
- `./flcore`: 
    - `./clients/clientAvgDBE.py`: the code on the client. 
    - `./servers/serverAvgDBE.py`: the code on the server. 
    - `./trainmodel/models.py`: the code for models. 
- `./utils`:
    - `data_utils.py`: the code to read the dataset. 

# Simulation

## Environments
With the installed [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh), we can run this platform in a conda virtual environment called *fl_torch*. 
```
conda env create -f env_linux.yaml # for Linux
```