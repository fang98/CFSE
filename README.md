# Collaborative Filtering Enhanced Subgraph Embedding for Link Direction and Sign Prediction

Instructions for reproducing the experiments reported in Table 3 and Table 4.


## Requirements
- Python 3.7
- PyTorch 1.8.1
- CUDA 11.1

## Link Sign Prediction

To preprocess data, follow these steps:

1. Execute `python preprocessing.py` to create the train/test sets for the link sign prediction task.
2. Execute `python main.py` for link sign prediction in signed directed networks.

## Link Prediction

To conduct experiments, follow these steps:

1. Execute `python main.py` for link prediction in directed networks.

