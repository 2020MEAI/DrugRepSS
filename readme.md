# DrugRepSS: Textual Semantic and Graph Structural Progressive Representation Learning for Drug Repositioning

# 1.Introduction

This repository contains source code and datasets for paper "DrugRepSS: Textual Semantic and Graph Structural Progressive Representation Learning for Drug Repositioning".The method introduces text semantics into drug repositioning modeling for the first time, proposing the graph contrastive learning module to capture a deep structural representation of diseases and drugs, and subsequently designing the learning-to-rank module to accurately predict the drug-disease relationship.

# 2. Overview

![image](https://github.com/2020MEAI/DrugRepSS/blob/master/framework_img.png)

# 3.Install Python libraries needed

```bash
conda create -n DrugRepSS python=3.8
conda activate DrugRepSS
pip install -r requirements.txt
```

# 4.Methods

We first obtain the textual semantic representations of drugs and diseases through the large language model and then obtain the deep structural representations of drugs on this basis by graph contrastive learning module. Finally, we obtain the ranking of candidate drugs for a disease through  learning-to-rank module. To ensure different source code could run successfully in our framework, we modify part of their source code.

## 4.1 LLM for textual semantic representation

### 4.1.1 Dataset

- semantic_feature/dataset/disease_id.xlsx: The names and ids of the disease
- semantic_feature/dataset/drug_id.xlsx: The names and ids of the drug

### 4.1.2 Running

```python
run semantic_feature/chatGPT.py
```

### 4.1.3 Output

- drug_id_vector.xlsx: Drug features output by the large language model
- disease_id_vector.xlsx: Disease features output by the large language model

## 4.2 Graph contrastive learning for structural representation 

### 4.2.1 Dataset

- tructural_feature/dataset/disease_drug_vector.xlsx: Features output by the large language model(The first 263 features are disease features and the rest are drug features)

- tructural_feature/dataset/dis_drug_list.csv: drug-disease association data 

### 4.2.2 Running

- run GIN_Recover model

  ```python
  run tructural_feature/gnnrecoveremb/MyModel.py
  ```

- run Pro_G model 

  ```python
  run tructural_feature/ProGemb/NE.py
  ```

- run MA_GCL model

  ```python
  run tructural_feature/MAGCLemb/main.py
  ```

### 4.2.3 Output

- output _features.xlsx: drug features and disease features output by the GNN

## 4.3  Learning to rank for drug repositioning

### 4.3.1 Dataset

- tructural_feature/dataset/train_origin.xlsx: the training dataset without drug features for LTR model
- tructural_feature/dataset/test_origin.xlsx: the testing dataset without drug features for LTR model
- output_features.xlsx: drug features and disease features output by the GNN

**We need to go through the following process to generate the dataset for the LTR model（We can refer to the file "tructural_feature/handle_file.py"）:**

1. We fill the drug features output by the GNN (output _features.xlsx) into the emb column in the train_origin.xlsx and test_origin.xlsx files **(emb column example: "1:feature_value 2:feature_value3:feature_value   ···  100:feature_value")**
2. We extracted three columns of data from train_origin.xlsx and test_origin.xlsx: disease id, drug effectiveness rank, and drug feature to form the training set and test set. Then we generate the file train_dataset.txt and test_dataset.txt

### 4.3.2 Running

​	You can run LTR models under LTR folder

- run LambdaRank model

  ```python
  python -u main.py --method LambdaRank --input_train ./mydata2/train_dataset.txt --input_test ./mydata2/test_dataset.txt --output ./resultdata/LambRank/example_LambRank_1v5.txt
  LambdaMART
  ```

- run LambdaMART model 

  ```python
  python -u main.py --method LambdaMART --input_train ./mydata2/train_dataset.txt --input_test ./mydata2/test_datset.txt --lr_LM 0.001 --output ./resultdata/LambdaMart/example_LambdaMART_1v5.txt
  ```

- run RankNet model

  ```python
  python -u main.py --method RankNet --input_train ./mydata2/train_dataset.txt --input_test ./mydata2/test_dataset.txt --output ./resultdata/example_rankNet_ran_1v1.txt
  ```

- run PLRank model

  ```python
  run LTR/PLRank/runrank.py
  run LTR/PLRank/plltr.py
  ```
  
- --method: the method of LTR model
- --input_train: the path of train dataset file
- --input_test: the path of test dataset file
- -- output: the path of output