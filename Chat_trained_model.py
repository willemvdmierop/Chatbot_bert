import os, sys, re
import time
import torch
import torchvision
# import nltk
# import pandas
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# pytorch layers
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential, Embedding, Linear, Conv1d, MaxPool1d, LSTM
from torch.nn import CrossEntropyLoss as CrossEntropyLoss
import torch.utils.data as data
from torch.jit import script, trace
from torch.utils.data import DataLoader as DataLoader
from collections import Counter, OrderedDict
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel, BertForMaskedLM
from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import AdamW
from torch.utils.tensorboard import SummaryWriter
# load the dataset interface
import utils
import dataset

start = time.time()
voc = []
# tokens for OOV, start and end
OOV = '<UNK>'
START_TOKEN = "<S>"
END_TOKEN = "</S>"
max_phrase_length = 40
minibatch_size = 1

device = 'cpu'
if (torch.cuda.is_available()):
    device = torch.device('cuda')

# Todo this implementation is for scibert
#tokenizer_scibert = AutoTokenizer.from_pretrained("./scibert_scivocab_uncased")
#model_scibert = BertForMaskedLM.from_pretrained("./scibert_scivocab_uncased")
#print('scibert model', model_scibert)
# return the list of OrderedDicts:
# a total of 83097 dialogues
full_data = utils.create_dialogue_dataset()

end = time.time()
print("\n"+96 * '#')
print(
    'Total data preprocessing time is {0:.2f} and the length of the vocabulary is {1:d}'.format(end - start, len(voc)))
print(96 * '#')

def load_data(**train_pars):
    stage = train_pars['stage']
    data = dataset.MoviePhrasesData(full_data)
    train_dataset_params = {'batch_size': minibatch_size, 'shuffle': True}
    dataloader = DataLoader(data, **train_dataset_params)
    return dataloader


load_data_pars = {'stage': 'train', 'num_workers': 3}
dataset = load_data(**load_data_pars) #this returns a dataloader
print('\n' + 40 * '#',"Now FineTuning", 40 * '#')
# Load the BERT tokenizer.
print('Loading BERT model...')
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

weights = torch.load("/Users/willemvandemierop/Documents/Master AI/Pycharm/Chatbot_bert/chatbot.pth", map_location= 'cpu')
model = BertForMaskedLM.from_pretrained("/Users/willemvandemierop/Documents/Master AI/Pycharm/Chatbot_bert/bertbot.pth")


params = list(model.named_parameters())

params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('======= Embedding Layer =======\n')
for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n======= First Transformer =======\n')
for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n======= Output Layer =======\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
