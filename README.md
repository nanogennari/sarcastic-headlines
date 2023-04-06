# Sarcastic headlines classification

![Brazil deaths](https://github.com/nanogennari/sarcastic-headlines/blob/main/models-comparison.png?raw=true)

## Introduction

This work is an exploration of different classification strategies for sarcastic and non sarcastic news headlines. We will compare the performance of tree different vectorizers: simple frequency counter (bag of words or BoW), a term frequency–inverse document frequency (TF-IDF) and a pre-trained BERT model's embeddings, associated with a K-nearest neighbors (kNN) classifier and at the end contrat with the performance of using transferred learning by retraining an BERT classifier to our dataset.

## Companion Article

This analysis has a companion article publised on Medium: [Sarcastic headlines classification](https://nanogennari.medium.com/sarcastic-headlines-classification-9738b1541229)


## Used data

* News Headlines Dataset For Sarcasm Detection: https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection

## Used libraries

* Pandas
* Numpy
* Plotly
* Nltk
* sklearn
* Ktrain
* Pytorch
* sentence_transformers

## Files

* `starcastic-headlines.ipynb`: contains the executed analysis.
* `Sarcasm_Headlines_Dataset_v2.json`: headlines dataset.

## Results

For the tested classification strategies we obtained the following accuracy:

| Method           | Accuracy |
| ---------------- | -------- |
| BoW + kNN        | 67.4%    |
| TF-IDF + kNN     | 67.2%    |
| Pre-trained Bert | 77.8%    |
| Bert finetunning | 93.1%    |

## Acknowledgments

First and foremost, praises and thanks to Rishabh Misra for collectiong and making the data necessary for this analysis available and easy to work with.

This analysis was made as a part of the Udacity's Data Scientist Nanodegree Program.

## References

Maiya, A. S. (2020). ktrain: A low-code library for augmented machine learning. arXiv preprint arXiv:2004.10703. Misra, R. (2019). News Headlines Dataset For Sarcasm Detection. https://www.kaggle.com/datasets/rmisra/
news-headlines-dataset-for-sarcasm-detection. [Online; accessed 16-October-2022].

Misra, R. and Arora, P. (2019). Sarcasm detection using hybrid neural network. arXiv preprint arXiv:1908.07414. Misra, R. and Grover, J. (2021). Sculpting Data for ML: The first act of Machine Learning.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830.

Reimers, N. and Gurevych, I. (2019). Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.
