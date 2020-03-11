import os, sys, re
import time
import torch
import torchvision
# import nltk
# import pandas
# Sentiment analysys
# Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011)
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# pytorch layers
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Embedding, Linear, Conv1d, MaxPool1d, LSTM
from torch.nn import CrossEntropyLoss as CrossEntropyLoss
import torch.utils.data as data
from torch.jit import script, trace
from torch.utils.data import DataLoader as DataLoader
from collections import Counter, OrderedDict
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel, BertForPreTraining
from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import AdamW
# load the dataset interface
import utils
import dataset

start = time.time()
voc = []
# tokens for OOV, start and end
OOV = '<UNK>'
START_TOKEN = "<S>"
END_TOKEN = "</S>"
max_phrase_length = 20
minibatch_size = 1

device = 'cpu'
if (torch.cuda.is_available()):
    device = torch.device('cuda')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = BertForPreTraining.from_pretrained('bert-base-uncased')

# Todo this implementation is for scibert
tokenizer_scibert = AutoTokenizer.from_pretrained("/Users/willemvandemierop/Google Drive/DL Prediction (706)/scibert_scivocab_uncased")
model_scibert = BertForPreTraining.from_pretrained("/Users/willemvandemierop/Google Drive/DL Prediction (706)/scibert_scivocab_uncased")
#print('scibert model', model_scibert)
# return the list of OrderedDicts:
# a total of 83097 dialogues
full_data = utils.create_dialogue_dataset()

#voc = utils.create_vocab()
#voc.append(OOV)
#voc.append(START_TOKEN)
#voc.append(END_TOKEN)
#f = open("vocab.txt","w+")
# for i in voc:
#    f.write(i + '\n')
# f.close()
#print("We load the vocab from the text file and get:")
voc = [line.rstrip('\n') for line in
       open("/Users/willemvandemierop/Documents/Master AI/Pycharm/DL Prediction/Coursework/vocab.txt")]
print(voc[5000:5010])
voc_idx = OrderedDict()
for idx, w in enumerate(voc):
    voc_idx[w] = idx

end = time.time()
print("\n"+96 * '#')
print(
    'Total data preprocessing time is {0:.2f} and the length of the vocabulary is {1:d}'.format(end - start, len(voc)))
print(96 * '#')

def load_data(**train_pars):
    stage = train_pars['stage']
    data = dataset.MoviePhrasesData(max_phrase_length, voc_idx, full_data, OOV, START_TOKEN, END_TOKEN)
    train_dataset_params = {'batch_size': minibatch_size, 'shuffle': True}
    dataloader = DataLoader(data, **train_dataset_params)
    return dataloader


load_data_pars = {'stage': 'train', 'num_workers': 3}
dataset = load_data(**load_data_pars) #this returns a dataloader
print('\n' + 40 * '#',"Now FineTuning", 40 * '#')

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

device = 'cpu'
if device == 'cuda':
    device = 'cuda'

model.to(device)
#forward(input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None)
# class transformers.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.0, correct_bias=True)
lrate = 1e-6
optim_pars = {'lr': lrate, 'weight_decay': 1e-3}
optimizer = AdamW(model.parameters(), **optim_pars)

current_batch = 0
epochs = 1
for epoch in range(epochs):
    for idx, val in enumerate(dataset):
        data = val[0][1]
        print("gittest")
        tokenized = tokenizer.tokenize((val))


        optimizer.zero_grad()

        break
    break


