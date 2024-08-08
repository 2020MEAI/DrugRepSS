# DrugRepSS: DrugRepSS: Textual Semantic and Graph Structural Progressive Representation Learning for Drug Repositioning

# 1.Introduction

This repository contains source code and datasets for paper "DrugRepSS: Textual Semantic and Graph Structural Progressive Representation Learning for Drug Repositioning".In this study, we propose a drug repositioning framework called DrugRepSS, which leverages the effectiveness comparison relationships (ECR) between drugs to identify new drug indications based on graph neural networks and ranking learning. 

# 2.Install Python libraries needed

```bash
conda create -n DrugRepSS python=3.8
conda activate DrugRepSS
pip install -r requirements.txt
```

# 3.Methods

We first obtain the semantic embedding representations of drugs and diseases through the large language model and then obtain the structural embedding representations of drugs on this basis by using different graph neural networks. Finally, we obtain the ranking of candidate drugs for a disease through different ranking learning methods. To ensure different source code could run successfully in our framework, we modify part of their source code.

## 3.1 LLM for  semantic feature representations of drugs and diseases

### 3.1.1 Dataset

- LLM/dataset/disease_cui_id.xlsx: The names and ids of the disease
- LLM/dataset/drug_id.xlsx: The names and ids of the drug

### 3.1.2 Running

- run LLM/chatGPT.py

## 3.2 GNN for more rich and highly characterized drug features 

### 3.2.1 Dataset

- embedding/dataset/disease_drug_vector.xlsx: Features output by the large language model(The first 263 features are disease features and the rest are drug features)

- embedding/dataset/dis_drug_list_new.csv: drug-disease association data 

### 3.2.2 Running

- run GIN_Recover model

  ```python
  run embedding/gnnrecoveremb/yModel.py
  ```

- run Pro_G model 

  ```python
  run embedding/ProGemb/NE.py
  ```

- run MA_GCL model

  ```python
  run embedding/MAGCLemb/main.py
  ```

## 3.3  Learn to rank for ranking the relevant candidate drugs

### 3.3.1 Dataset

**We need to go through the following process to generate the dataset for the LTR model:**

1. We fill the drug features output by the GNN into the emb column in the training and test set files **(emb column example: "1:feature_value 2:feature_value3:feature_value   ···  100:feature_value")**
2. We extracted three columns of data from the training set and test set files: disease id, drug effectiveness rank, and drug feature to form the training set and test set

### 3.3.2 Running

​	You can run LTR models under LTR folder

- run LambdaRank model

  ```python
  python -u main.py --method LambdaRank --input_train ./mydata2/output训练集magcl.txt --input_test ./mydata2/output测试集magcl.txt --output ./resultdata/LambRank/example_LambRank_1v5.txt
  LambdaMART
  ```

- run LambdaMART model 

  ```python
  python -u main.py --method LambdaMART --input_train ./mydata2/output训练集gnnrecover.txt --input_test ./mydata2/output测试集gnnrecover.txt --lr_LM 0.001 --output ./resultdata/LambdaMart/example_LambdaMART_1v5.txt
  ```

- run RankNet model

  ```python
  python -u main.py --method RankNet --input_train ./mydata2/output训练集gnnrecover.txt --input_test ./mydata2/output测试集gnnrecover.txt --output ./resultdata/example_rankNet_ran_1v1.txt
  ```

- run PLRank model

  ```python
  run LTR/PLRank/runrank.py
  run LTR/PLRank/plltr.py
  ```
  
  
