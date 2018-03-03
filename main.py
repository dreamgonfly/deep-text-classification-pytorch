import argparse
from os.path import dirname, abspath, join, exists
import os

import torch
from torch.optim import Adadelta, Adam, lr_scheduler
from torch import nn
import numpy as np

from download_dataset import DATASETS
from preprocessors import DATASET_TO_PREPROCESSOR
import dictionaries
from dataloaders import TextDataset, TextDataLoader
from trainers import Trainer
from evaluators import Evaluator

from models.CharCNN import CharCNN
from models.WordCNN import WordCNN
from models.VDCNN import VDCNN
from models.QRNN import QRNN

import utils

# Random seed
np.random.seed(0)
torch.manual_seed(0)

# Arguments parser
parser = argparse.ArgumentParser(description="Deep NLP Models for Text Classification")
parser.add_argument('--dataset', type=str, default='ag_news', choices=DATASETS)
parser.add_argument('--use_gpu', type=bool, default=torch.cuda.is_available())
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--initial_lr', type=float, default=0.01)
parser.add_argument('--lr_schedule', action='store_true')
parser.add_argument('--optimizer', type=str, default='Adam')

subparsers = parser.add_subparsers(help='NLP Model')

## WordCNN
WordCNN_parser = subparsers.add_parser('WordCNN')
# WordCNN_parser.set_defaults(preprocess_level='word')
WordCNN_parser.add_argument('--preprocess_level', type=str, default='word', choices=['word', 'char'])
WordCNN_parser.add_argument('--dictionary', type=str, default='WordDictionary', choices=['WordDictionary', 'AllCharDictionary'])
WordCNN_parser.add_argument('--max_vocab_size', type=int, default=50000)
WordCNN_parser.add_argument('--min_count', type=int, default=None)
WordCNN_parser.add_argument('--start_end_tokens', type=bool, default=False)
group = WordCNN_parser.add_mutually_exclusive_group()
group.add_argument('--vector_size', type=int, default=128, help='Only for rand mode')
group.add_argument('--wordvec_mode', type=str, default=None, choices=['word2vec', 'glove'])
WordCNN_parser.add_argument('--min_length', type=int, default=5)
WordCNN_parser.add_argument('--max_length', type=int, default=300)
WordCNN_parser.add_argument('--sort_dataset', action='store_true')
WordCNN_parser.add_argument('--mode', type=str, default='rand', choices=['rand', 'static', 'non-static', 'multichannel'])
WordCNN_parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[3,4,5])
WordCNN_parser.add_argument('--epochs', type=int, default=10)
WordCNN_parser.set_defaults(model=WordCNN)

## CharCNN
CharCNN_parser = subparsers.add_parser('CharCNN')
CharCNN_parser.set_defaults(preprocess_level='char')
CharCNN_parser.add_argument('--dictionary', type=str, default='CharCNNDictionary', choices=['CharCNNDictionary', 'VDCNNDictionary', 'AllCharDictionary'])
CharCNN_parser.add_argument('--min_length', type=int, default=1014)
CharCNN_parser.add_argument('--max_length', type=int, default=1014)
CharCNN_parser.add_argument('--mode', type=str, default='small')
CharCNN_parser.add_argument('--epochs', type=int, default=3)
CharCNN_parser.set_defaults(model=CharCNN)

## VDCNN
VDCNN_parser = subparsers.add_parser('VDCNN')
VDCNN_parser.set_defaults(preprocess_level='char')
VDCNN_parser.add_argument('--dictionary', type=str, default='VDCNNNDictionary', choices=['CharCNNDictionary', 'VDCNNDictionary', 'AllCharDictionary'])
VDCNN_parser.add_argument('--min_length', type=int, default=1014)
VDCNN_parser.add_argument('--max_length', type=int, default=1014)
VDCNN_parser.add_argument('--epochs', type=int, default=3)
VDCNN_parser.add_argument('--depth', type=int, default=29, choices=[9, 17, 29, 49])
VDCNN_parser.add_argument('--embed_size', type=int, default=16)
VDCNN_parser.add_argument('--optional_shortcut', type=bool, default=True)
VDCNN_parser.add_argument('--k', type=int, default=8)
VDCNN_parser.set_defaults(model=VDCNN)

args = parser.parse_args()

# Logging
model_name = args.model.__name__
logger = utils.get_logger(model_name)

logger.info('Arguments: {}'.format(args))

logger.info("Preprocessing...")
Preprocessor = DATASET_TO_PREPROCESSOR[args.dataset]
preprocessor = Preprocessor(args.dataset)
train_data, val_data, test_data = preprocessor.preprocess(level=args.preprocess_level)

logger.info("Building dictionary...")
Dictionary = getattr(dictionaries, args.dictionary)
dictionary = Dictionary(args)
dictionary.build_dictionary(train_data)

logger.info("Making dataset & dataloader...")
train_dataset = TextDataset(train_data, dictionary, args.sort_dataset, args.min_length, args.max_length)
train_dataloader = TextDataLoader(dataset=train_dataset, dictionary=dictionary, batch_size=args.batch_size)
val_dataset = TextDataset(val_data, dictionary, args.sort_dataset, args.min_length, args.max_length)
val_dataloader = TextDataLoader(dataset=val_dataset, dictionary=dictionary, batch_size=64)
test_dataset = TextDataset(test_data, dictionary, args.sort_dataset, args.min_length, args.max_length)
test_dataloader = TextDataLoader(dataset=test_dataset, dictionary=dictionary, batch_size=64)

logger.info("Constructing model...")
model = args.model(n_classes=preprocessor.n_classes, dictionary=dictionary, args=args)
if args.use_gpu:
    model = model.cuda() 
    
logger.info("Training...")
trainable_params = [p for p in model.parameters() if p.requires_grad]
if args.optimizer == 'Adam':
    optimizer = Adam(params=trainable_params, lr=args.initial_lr)
if args.optimizer == 'Adadelta':
    optimizer = Adadelta(params=trainable_params, lr=args.initial_lr, weight_decay=0.95)
lr_plateau = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=5, min_lr=0.0001)
criterion = nn.CrossEntropyLoss
trainer = Trainer(model, train_dataloader, val_dataloader, 
                  criterion=criterion, optimizer=optimizer, 
                  lr_schedule=args.lr_schedule, lr_scheduler=lr_plateau, 
                  use_gpu=args.use_gpu, logger=logger)
trainer.run(epochs=args.epochs)

logger.info("Evaluating...")
logger.info('Best Model: {}'.format(trainer.best_checkpoint_filepath))
model.load_state_dict(torch.load(trainer.best_checkpoint_filepath)) # load best model
evaluator = Evaluator(model, test_dataloader, use_gpu=args.use_gpu, logger=logger)
evaluator.evaluate()