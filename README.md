# SNI-ML

Code for the paper Research on Twitter Summary Based on  Social Network Information and Multimodal Learning.

Requirements and Installation：

Python version >= 3.6
numpy version >= 1.18.5

  
Generate summary by SNI-ML:

python main.py  -eta 0.3  -lam 0.9  -lamada 0.8 -mu 0.2  -sn ./data/SN/ -sum_path ./data/SNI-ML-sum/  -words ./data/filed_words/

Test for rouge:

python  test.py -gold_sum_path ./data/gold-sum -sum_path ./data/SNI-ML-sum/  


Test for Topic relevance:

python  Relevence_topic.py -re_path ./data/re_s/  -sum_path ./data/SNI-ML-sum/  -vector ../GoogleNews-vectors-negative300.bin



Citation:





Description： Because of the overall data is too much to upload, we only upload part of the data used in the paper is in the Folder data.

Doesn't work?
Please contact Yangyang at yangyang_cao2021@163.com
