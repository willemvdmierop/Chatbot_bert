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
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# load the dataset interface
import utils
import dataset
import numpy as np

start = time.time()
voc = []
# tokens for OOV, start and end
OOV = '<UNK>'
START_TOKEN = "<S>"
END_TOKEN = "</S>"
max_phrase_length = 40
minibatch_size = 200

device = 'cpu'
if (torch.cuda.is_available()):
    device = torch.device('cuda')

# Todo this implementation is for scibert
# tokenizer_scibert = AutoTokenizer.from_pretrained("./scibert_scivocab_uncased")
# model_scibert = BertForMaskedLM.from_pretrained("./scibert_scivocab_uncased")
# print('scibert model', model_scibert)
# return the list of OrderedDicts:
# a total of 83097 dialogues
#question_data, answer_data = utils.question_answers_dataset()
full_data = utils.create_dialogue_dataset()

end = time.time()
print("\n" + 96 * '#')
print(
    'Total data preprocessing time is {0:.2f} and the length of the vocabulary is {1:d}'.format(end - start, len(voc)))
print(96 * '#')


def load_data(**train_pars):
    stage = train_pars['stage']
    #data = dataset.MoviePhrasesData(full_data)
    data = dataset.MoviePhrasesData(all_dialogues = full_data)
    train_dataset_params = {'batch_size': minibatch_size, 'shuffle': True}
    dataloader = DataLoader(data, **train_dataset_params)
    return dataloader


load_data_pars = {'stage': 'train', 'num_workers': 3}
dataset = load_data(**load_data_pars)  # this returns a dataloader
print('\n' + 40 * '#', "Now FineTuning", 40 * '#')
# Load the BERT tokenizer.
print('Loading BERT model...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

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

tb = SummaryWriter()
model.to(device)
# forward(input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None)
# class transformers.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.0, correct_bias=True)
lrate = 1e-4
optim_pars = {'lr': lrate}
#optimizer = AdamW(model.parameters(), **optim_pars)
optimizer = optim.Adam(model.parameters(), **optim_pars)
wd = os.getcwd()
if not os.path.exists(wd + "/my_saved_model_directory"):
    os.mkdir(wd + "/my_saved_model_directory")
if not os.path.exists(wd + "/my_saved_model_directory_final"):
    os.mkdir(wd + "/my_saved_model_directory_final")
current_batch = 0
total_phrase_pairs = 0
epochs = 1
loss_list = []
for epoch in range(epochs):
    t0 = time.time()
    print(30 * "#" + ' Training ' + 30 * "#")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    total_loss = 0
    counter = 0
    print("length of dataset: ", len(dataset))
    for idx, X in enumerate(dataset):
        model.zero_grad()
        # number of phrases
        # X[0] is just the index
        # X[1] is the dialogue, X[1][0] are input phrases
        batch_size = len(X[0])
        # number of tokens in a sequence 
        seq_length = len(X[1][0]['input_ids'][0].squeeze())
        total_phrase_pairs += batch_size
        input_tensor = torch.zeros((batch_size, seq_length), dtype=torch.long).to(device)
        token_id_tensor = torch.zeros((batch_size, seq_length), dtype=torch.long).to(device)
        attention_mask_tensor = torch.zeros((batch_size, seq_length), dtype=torch.long).to(device)
        masked_lm_labels_tensor = torch.zeros((batch_size, seq_length), dtype=torch.long).to(device)
        new_input_eval = torch.zeros((batch_size, seq_length), dtype=torch.long).to(device)
        for i in range(batch_size):
            input_tensor[i] = X[1][0]['input_ids'][i].squeeze()
            token_id_tensor[i] = X[1][0]['token_type_ids'][i].squeeze()
            attention_mask_tensor[i] = X[1][0]['attention_mask'][i].squeeze()
            masked_lm_labels_tensor[i] = X[1][0]['masked_lm_labels'][i].squeeze()
            new_input_eval[i] = X[1][0]['new_input_eval'][i].squeeze()

        outputs = model(input_ids=input_tensor, attention_mask=attention_mask_tensor, token_type_ids=token_id_tensor,
                        masked_lm_labels=masked_lm_labels_tensor)
        loss = outputs[0]
        total_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # check what the model is doing
        # model.eval()
        # output_model = model(new_input_eval)
        # output_model = output_model[0].detach()
        # idx = torch.argmax(output_model, dim = -1)
        # given_text = tokenizer.convert_ids_to_tokens(new_input_eval[0])
        # generated_text = tokenizer.convert_ids_to_tokens(idx[0])
        # original_text = tokenizer.convert_ids_to_tokens(input_tensor[0])

        # model.train()
        counter += 1
        tb.add_scalar('Loss_Bert_model', loss, epoch)
        if counter % 4000 == 0:
            print("*", end='')

    average_loss = total_loss / len(dataset)
    loss_list.append(average_loss)
    model.save_pretrained(wd + "/my_saved_model_directory/")
    print("average loss '{}' and train time '{}'".format(average_loss, (time.time() - t0)))

tb.close()
model.save_pretrained(wd + "/my_saved_model_directory_final/")
print(model.get_output_embeddings())
