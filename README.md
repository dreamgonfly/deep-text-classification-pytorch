# Deep Text Classification in PyTorch
PyTorch implementation of deep text classification models including:

- [WordCNN : Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- [CharCNN : Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)
- [VDCNN : Very Deep Convolutional Networks for Text Classification](https://arxiv.org/abs/1606.01781)
- [QRNN : Quasi-Recurrent Neural Networks](https://arxiv.org/abs/1611.01576)

## Requirements
- Python 3.5+
- [PyTorch 0.3](pytorch.org)
- [gensim 3.2](https://github.com/RaRe-Technologies/gensim)
- [tqdm](https://github.com/tqdm/tqdm)
- [requests](https://github.com/requests/requests)

## Usage
To begin, you will need to download datasets as follows:
```
$ python download_dataset.py all
```
You can also download a specific dataset by specifying its name instead of `all`. Available datasets are `MR`, `SST-1`, `SST-2`, `ag_news`, `sogou_news`, `dbpedia`, `yelp_review_full`,  `yelp_review_polarity`, `yahoo_answers`, `amazon_review_full`, and `amazon_review_polarity`

To download word vectors, run the following:
```
$ python download_wordvector.py word2vec
$ python download_wordvector.py glove
```

#### WordCNN
To train WordCNN with rand mode:
```
$ python main.py --dataset MR --use_gpu WordCNN --mode rand --vector_size 128 --epochs 300
```
To train WordCNN with multichannel mode:
```
$ python main.py --dataset MR --use_gpu WordCNN --mode multichannel --wordvec_mode word2vec --epochs 300
```
Available modes are `rand`, `static`, `non-static`, and `multichannel`

#### CharCNN
To train CharCNN with small mode:
```
$ python main.py --dataset MR --use_gpu CharCNN --mode small --epochs 300
```
To train CharCNN with large mode:
```
$ python main.py --dataset MR --use_gpu CharCNN --mode large --epochs 300
```

#### VDCNN
To train VDCNN with depth = 29:
```
$ python main.py --dataset MR --use_gpu VDCNN --depth 29
```

#### QRNN
To train QRNN with hidden_size = 300:
```
$ python main.py --dataset MR --use_gpu QRNN --hidden_size 300 --epochs 300
```

#### TF-IDF (benchmark)
You can train a multinomial logistic regression with TF-IDF features as a benchmark.
```
$ python tf-idf.py --dataset MR
```

#### Help
Refer to `python main.py --help` and `python main.py {WordCNN, CharCNN, VDCNN, QRNN} --help` for full description of how to use.


## Experiments
Results are reported as follows: **Test accuracy reproduced here (Test accuracy reported by the paper)**

To find the settings for experiments, refer to `experiments.sh`.

|                                 |  MR  |     SST_1      |     SST_2      |       ag_news  |     sogu_news     |      db_pedia      | yelp_review_full | yelp_review_polarity | yahoo_answer | amazon_review_full | amazon_review_polarity |
|:-------------------------------:|:----:|:--------------:|:--------------:|:--------------:|:-----------------:|:------------------:|:--------------------:|:------------------:|:------------:|:------------------:|:----------------------:|
|WordCNN (rand)                   |      |                |                |    88.3        |                   |                    |           92.5       |                    |              |                    |                        |

## References
- [Shawn1993's CNNs for Sentence Classification in PyTorch](https://github.com/Shawn1993/cnn-text-classification-pytorch)
- [ArdalanM's nlp-benchmarks](https://github.com/ArdalanM/nlp-benchmarks)
- [salesforce's Quasi-Recurrent Neural Network (QRNN) for PyTorch](https://github.com/salesforce/pytorch-qrnn)
