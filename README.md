Code for our paper ["The Sparse Recovery Autoencoder"](https://arxiv.org/abs/1806.10175). 

# Toy example
Under the folder **toy_example**, we provide a jupyter notebook `jan22_toy_example.ipynb` that works through the training and evaluation of our autoencoder (as well as other baseline algorithms) for the synthetic1 dataset. We highly recommend interested readers to take a look at the provided notebook before diving deep into our code.

# Overview

The source code contains four parts:
1. Core
   - model.py
   - utils.py
   - datasets.py
   - baselines.py
2. Code for each dataset
   - synthetic_main.py
   - synthetic_powerlaw_main.py
   - amazon_main.py
   - amazon_parallel_l1.py
   - rcv1_main.py
   - rcv1_parallel_l1.py
3. Scripts for reproducing our results
   - scripts/synthetic1.sh
   - scripts/synthetic2.sh
   - scripts/amazon.sh
   - scripts/rcv1.sh
   - scripts/synthetic_powerlaw.sh
4. Code and scripts for one of the baselines `Simple AE + l1-min`
   - synthetic_simpleAE.py
   - amazon_simpleAE.py
   - rcv1_simpleAE.py
   - scripts are under simpleAE_scripts/


# Run
To reproduce our experimental results, first run `chmod +x scripts/*.sh` to make the scripts executable. After that, run the given scripts:
- `$ ./scripts/synthetic1.sh`
- `$ ./scripts/synthetic2.sh`
- `$ ./scripts/amazon.sh`
- `$ ./scripts/rcv1.sh`
- `$ ./scripts/synthetic_powerlaw.sh`

Note:
1. The results are stored in a python dictionary which is then saved under the folder `ckpts/`. They can be used to reproduce the figures shown in our paper.
2. Before running `amazon.sh`, download `train.csv` from [this kaggle competition](https://www.kaggle.com/c/amazon-employee-access-challenge/data) and specify its location via --data_dir.
3. The RCV1 dataset will be fetched automatically using the `sklearn.datasets.fetch_rcv1` function.
4. To reproduce results of one of the baselines `Simple AE + l1-min`, run scripts under the folder simpleAE_scripts/.
5. For high-dimensional vectors, solving `l1-min` using [Gurobi](http://www.gurobi.com/) takes a long time on a single CPU. To speed up, we solve `l1_min` in parallel on a multi-core machine. In `amazon_main.py` and `rcv1_main.py`, performance evaluation is performed on a small set of the test samples (while training is still done using the complete training set). After training the autoencoder, we use a multi-core machine and solve `l1_min` in parallel on the complete test set using `amazon_parallel_l1.py` and `rcv1_parallel_l1.py`. Depending on your multi-core machine, solving `l1_min` in parallel on the *complete* test set may still take a long time, I would recommend running `amazon_parallel_l1.py` and `rcv1_parallel_l1.py` first with a small subset (by setting a small number for the parameters `num_core` and `batch` in the python file).  


# Environment
Here is our software environment. 

1. Python 2.7.12
   - numpy 1.13.3
   - sklearn 0.19.1
   - scipy 1.0.0
   - joblib 0.10.0
2. Tensorflow r1.4
3. Gurobi 7.5.1
