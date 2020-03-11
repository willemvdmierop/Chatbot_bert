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
tokenizer_scibert = AutoTokenizer.from_pretrained("/Users/willemvandemierop/Google Drive/DL Prediction (706)/scibert_scivocab_uncased")
model_scibert = BertForMaskedLM.from_pretrained("/Users/willemvandemierop/Google Drive/DL Prediction (706)/scibert_scivocab_uncased")
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
    data = dataset.MoviePhrasesData(voc_idx, full_data, max_phrase_length)
    train_dataset_params = {'batch_size': minibatch_size, 'shuffle': True}
    dataloader = DataLoader(data, **train_dataset_params)
    return dataloader


load_data_pars = {'stage': 'train', 'num_workers': 3}
dataset = load_data(**load_data_pars) #this returns a dataloader
print('\n' + 40 * '#',"Now FineTuning", 40 * '#')
# Load the BERT tokenizer.
print('Loading BERT model...')
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')


params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')
for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== First Transformer ====\n')
for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Output Layer ====\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


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
total_phrase_pairs = 0
epochs = 1
loss_list = []
for epoch in range(epochs):
    t0 = time.time()
    print('======== Training =======')
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    total_loss = 0
    for idx, X in enumerate(dataset):
        model.zero_grad()
        # number of phrases
        # X[0] is just the index
        # X[1] is the dialogue, X[1][0] are input phrases
        batch_size = len(X[1])
        # number of tokens in a sequence 
        seq_length = len(X[1][0]['input_ids'].squeeze())
        total_phrase_pairs += batch_size
        input_tensor = torch.zeros((batch_size,seq_length), dtype = torch.long)
        token_id_tensor = torch.zeros((batch_size,seq_length), dtype = torch.long)
        attention_mask_tensor = torch.zeros((batch_size,seq_length), dtype = torch.long)
        lm_labels_tensor = torch.zeros((batch_size,seq_length), dtype = torch.long)
        for i in range(batch_size):
            input_tensor[i] = X[1][i]['input_ids'].squeeze()
            token_id_tensor[i] = X[1][i]['token_type_ids'].squeeze()
            attention_mask_tensor[i] = X[1][i]['attention_mask'].squeeze()
            lm_labels_tensor[i] = X[1][i]['lm_labels'].squeeze()


        outputs = model(input_ids = input_tensor, attention_mask = attention_mask_tensor, token_type_ids = token_id_tensor, lm_labels= lm_labels_tensor)

        loss = outputs[0]
        total_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    average_loss = total_loss/len(dataset)
    loss_list.append(average_loss)
    print("average loss '{}' and train time '{}'".format(average_loss,(time.time()-t0)))



