# GUM
GUM(Group Utility Maximization) is the official repository for the paper "Summarizing User-Item Matrix By Group Utility Maximization"(in submission). 

# Requirements

```python
pip install captum, numpy, pandas, pytorch, sklearn
```

# Getting Start

The listed four  folders. 

- preprocess/titanic_importance.py specifies how to train the model and obtain the feature importance matrix. The pretrained model is provided too. 
- data/ folder consists of the raw titanic dataset, feature importance matrix of titanic and netflix-200 data processed by ours. Limited by the GitHub capacity,  users can download the netflix-prize and Movielens datasets from their official links [netflix-prize-data](https://www.kaggle.com/netflix-inc/netflix-prize-data) and [Movielens](https://grouplens.org/datasets/movielens/).
- algorithm/ folder contains the CELF (accelerated greedy), stochastic greedy, k-max, brute_force, and baselines in our paper.
- run/ folder provides examples to run the algorithms on different datasets.  

# Example

The such code snippet shows how to obtain the group summarization on Titanic dataset. You can reproduce the experiments by replacing the dataset and setting the parameters $k$ and $l$ of our examplar codes.

```bash
cd run;
python titanic_run.py
```
# Cite Us
