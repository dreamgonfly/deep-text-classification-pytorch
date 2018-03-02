# Deep Text Classification in PyTorch
PyTorch implementation of deep text classification models including:

- [WordCNN : Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- [CharCNN : Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)
- [VDCNN : Very Deep Convolutional Networks for Text Classification](https://arxiv.org/abs/1606.01781)
- [QRNN : Quasi-Recurrent Neural Networks](https://arxiv.org/abs/1611.01576)

## Requirements
- python 3.5+
- PyTorch 0.3
- gensim 3.2

## Usage
Download datasets with:
```
$ python download_dataset.py all
```
You can also download a specific dataset by specifying its name instead of `all`. Available datasets are `amazon_review_full`, `yelp_review_polarity`, `yahoo_answers`, `dbpedia`, `ag_news`, `yelp_review_full`, `amazon_review_polarity`, `sogou_news`, `MR`, `SST-1`, and `SST-2`

Download word vectors with:
```
$ python download_wordvector.py word2vec
$ python download_wordvector.py glove
```

#### WordCNN
To train WordCNN with rand mode:
```
$ python main.py --dataset MR --use_gpu WordCNN --mode rand --vector_size 128 --epochs 256
```
To train WordCNN with multichannel mode:
```
$ python main.py --dataset MR --use_gpu WordCNN --mode multichannel --wordvec_mode word2vec --epochs 256
```
Available modes are `rand`, `static`, `non-static`, and `multichannel`

## Results
|                                 |  MR  |     SST_1      |     SST_2      |       ag_news  |     sogu_news     |      db_pedia      | yelp_review_polarity | amazon_review_full | yahoo_answer | amazon_review_full | amazon_review_polarity |
|:-------------------------------:|:----:|:--------------:|:--------------:|:--------------:|:-----------------:|:------------------:|:--------------------:|:------------------:|:------------:|:------------------:|:----------------------:|
|WordCNN (rand)                   |      |                |                |    88.3        |                   |                    |           92.5       |                    |              |                    |                        |

## References
- [nlp-benchmarks](https://github.com/ArdalanM/nlp-benchmarks)
