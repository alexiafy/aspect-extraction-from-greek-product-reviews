# Aspect Extraction from Greek Product Reviews

This is the github repository of the "Aspect Extraction from Greek Product Reviews" thesis, submitted for the partial fullfilment of the M.Sc. Data & Web Science

----------------------------------------------------
**Abstract**:

Aspect-based Sentiment Analysis is a research field of Opinion Mining that focuses on the detection of the sentiment expressed towards specific aspects of a document. In a sentence, different sentiment might be expressed for different aspects, therefore, the extraction of those aspects and the detection of their sentiment is a significant task. A large number of studies have investigated this research field, however most of those studies have focused on the English language. This thesis focuses on aspect extraction from Greek product reviews, a task that has not been previously explored. Explicitly mentioned, single and multi-word aspects are detected on this purpose. Moreover, the task of aspect-based sentiment analysis is also explored, providing an end to end solution. A novel dataset that consists of product reviews in Greek language was leveraged. Different preprocessing steps were employed and their contribution in the evaluated performance is reported. Two baseline approaches, namely a feature engineering model and embedding-based neural network architectures, are firstly introduced. As an attempt to improve the performance, Bidirectional Encoder Representations from Transformers (BERT) model was also employed. The pre-trained BERT model was fine tuned and the contribution of different layers on top was investigated. The Conditional Random Field (CRF) layer was also utilized as an attempt to explore the advantage provided by applying constraint rules in the predicted labels. The BERT model contributed to a great performance gain compared to the baseline approaches, with which, the highest F1-Score for the aspect extraction task was 84.22\%.

----------------------------------------------------
Data

| First Header                          | Precision     | Recall        | F1-Score      |
| ------------------------------------- | ------------- | ------------- | ------------- |
| CRF                                   | 82.98         | 74.58         | 78.51         |


----------------------------------------------------
Model

![BERT CRF](https://user-images.githubusercontent.com/33041542/165788823-c5cdc1bb-cf8f-438e-9566-731ce1352e96.jpg)


The model summary:


----------------------------------------------------
Run

----------------------------------------------------
Results

| First Header                          | Precision     | Recall        | F1-Score      |
| ------------------------------------- | ------------- | ------------- | ------------- |
| CRF                                   | 82.98         | 74.58         | 78.51         |
| GRU + Softmax                         | 79.00         | 73.96         | 76.38         |
| BiGRU + Softmax                       | 83.37         | 74.13         | 78.39         |
| BiGRU + CRF                           | 83.81         | 74.38         | 78.75         |
| BERT + Softmax                        | 85.52         | 81.80         | 83.59         |
| BERT + CRF                            | 87.97         | 79.37         | 83.40         |
| BERT + L1 + Softmax                   | 86.31         | 82.24         | **84.22**     |
| BERT + L1 + CRF                       | 86.30         | 81.99         | 84.09         |
| BERT (concat middle layers) + Softmax | 84.43         | 80.85         | 82.58         |
| BERT (concat last layers) + Softmax   | 81.47         | 82.13         | 81.80         |


----------------------------------------------------
Requirements

----------------------------------------------------
