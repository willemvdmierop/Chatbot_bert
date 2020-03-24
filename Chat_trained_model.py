import os, sys, re
import time
import numpy as np
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
# tokenizer_scibert = AutoTokenizer.from_pretrained("./scibert_scivocab_uncased")
# model_scibert = BertForMaskedLM.from_pretrained("./scibert_scivocab_uncased")
# print('scibert model', model_scibert)
# return the list of OrderedDicts:
# a total of 83097 dialogues
full_data = utils.create_dialogue_dataset()

end = time.time()
print("\n" + 96 * '#')
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
dataset = load_data(**load_data_pars)  # this returns a dataloader
print('\n' + 40 * '#', "Now FineTuning", 40 * '#')
# Load the BERT tokenizer.
print('Loading BERT model...')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForMaskedLM.from_pretrained(
    "/Users/willemvandemierop/Documents/Master AI/Pycharm/Chatbot_bert/my_saved_model_directory_final")

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

PAD, MASK, CLS, SEP = '[PAD]', '[MASK]', '[CLS]', '[SEP]'


def to_bert_input(tokens):
    token_idx = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
    sep_idx = tokens.index(['SEP'])
    segment_idx = token_idx * 0
    segment_idx[(sep_idx + 1):] = 1
    mask = (token_idx != 0)
    return token_idx.unsqueeze(0), segment_idx.unsqueeze(0), mask.unsqueeze(0)


if __name__ == '__main__':
    done = True
    model.eval()
    # while True:
    while done:
        # message = input('\nEnter your Question here: ').strip()
        message = "hello how are you"
        tokens_message = tokenizer.encode_plus(message)
        # lm_labels = -100 * (tokens_message['attention_mask'] - tokens_message['token_type_ids'])
        # tokens_message['lm_labels'] = lm_labels
        print(tokens_message)
        message_ = tokenizer.decode(tokens_message['input_ids'])
        print(message_)
        batch_size = 1
        # number of tokens in a sequence
        seq_length = len(tokens_message['input_ids'])
        input_tensor = torch.zeros(seq_length, dtype=torch.long).to(device)
        token_id_tensor = torch.zeros(seq_length, dtype=torch.long).to(device)
        attention_mask_tensor = torch.zeros(seq_length, dtype=torch.long).to(device)
        # lm_labels_tensor = torch.zeros(seq_length, dtype=torch.long).to(device)
        for i in range(batch_size):
            input_tensor[i] = tokens_message[1][i]['input_ids'].squeeze()
            token_id_tensor[i] = tokens_message[1][i]['token_type_ids'].squeeze()
            attention_mask_tensor[i] = tokens_message[1][i]['attention_mask'].squeeze()
            lm_labels_tensor[i] = tokens_message[1][i]['lm_labels'].squeeze()
        # lm_labels_tensor = tokens_message['lm_labels']

        outputs = model(input_ids=input_tensor, attention_mask=attention_mask_tensor, token_type_ids=token_id_tensor,
                        lm_labels=None)

        logits = model(tokens_message)
        logits = logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        word_idx = np.argmax(probs)
        prob = probs[word_idx]
        top_tokens = tokenizer.convert_ids_to_tokens(word_idx)
        print("prob", prob, "tokens", top_tokens)

        done = False
