import os, sys, re
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader as DataLoader
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel, BertForMaskedLM
from transformers import AutoModelWithLMHead, AutoTokenizer
from torch.optim import Adam
from transformers import AdamW
from torch.utils.tensorboard import SummaryWriter
# load the dataset interface
import utils
import dataset_Q_A as dataset
import numpy as np
device = 'cpu'
if (torch.cuda.is_available()):
    device = torch.device('cuda')


start = time.time()
OOV = '<UNK>'
START_TOKEN = "<S>"
END_TOKEN = "</S>"

question_data, answer_data = utils.question_answers_dataset()

max_length_questions = 0
mean_length_q = 0
for i in range(len(question_data)):
    max_length_questions = max(max_length_questions, len(question_data[i]))
    mean_length_q += len(question_data[i])
mean_length_q /= len(question_data)

mean_length_a = 0
max_length_answers = 0
for i in range(len(answer_data)):
    max_length_answers = max(max_length_answers, len(answer_data[i]))
    mean_length_a += len(answer_data[i])
mean_length_a /= len(answer_data)

end = time.time()
print("\n" + 96 * '#')
print('Total data preprocessing time is {0:.2f} and the length of the vocabulary is {1:d}'.format(end - start, len(question_data)))
print('## Question data[0] : {} , \n## Answer data[0] :  {}'.format(question_data[0], answer_data[0]))
print('The max lenght of the Questions is: {}, the max length of the answers is: {}'.format(max_length_questions, max_length_answers))
print('The mean lenght of the Questions is: {0:.2f}, the mean length of the answers is: {1:.2f}'.format(mean_length_q, mean_length_a))
print(96 * '#')


def load_data(**train_pars):
    stage = train_pars['stage']
    data = dataset.MoviePhrasesData(questions_data= question_data, answers_data = answer_data, scibert = scibert_train)
    train_dataset_params = {'batch_size': minibatch_size, 'shuffle': True}
    dataloader = DataLoader(data, **train_dataset_params)
    return dataloader

Q_above_30 = []
A_above_30 = []
for i in range(len(question_data)):
    if len(question_data[i]) > 30:
        Q_above_30.append(question_data[i])
        A_above_30.append(answer_data[i])


############# SCIBERT ###########
scibert_train = True ############
#################################
max_phrase_length = 40
minibatch_size = 100
load_data_pars = {'stage': 'train', 'num_workers': 3}
dataLoader = load_data(**load_data_pars)  # this returns a dataloader
print('\n' + 40 * '#', "Loading the Bert Tokenizer", 40 * '#')
#################################### Load the BERT tokenizer. ########################################
wd = os.getcwd()
print('Loading BERT model...')
if scibert_train:
    print("Attention we are initializing the scibert model with scibert tokenizer!")
    tokenizer = BertTokenizer.from_pretrained('./scibert_scivocab_uncased', do_lower_case=True)
    model_Q_A = BertForMaskedLM.from_pretrained('./scibert_scivocab_uncased')
    model_Q_A.to(device)
    if not os.path.exists(wd + "/my_saved_model_dir_QA_Scibert_tmp"):
        os.mkdir(wd + "/my_saved_model_dir_QA_Scibert_tmp")
    if not os.path.exists(wd + "/my_saved_model_dir_QA_Scibert_final"):
        os.mkdir(wd + "/my_saved_model_dir_QA_Scibert_final")
    # !Attention we need the full size of the new vocabulary!!
    model_Q_A.resize_token_embeddings(len(tokenizer))
else:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model_Q_A = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model_Q_A.to(device)
    if not os.path.exists(wd + "/my_saved_model_dir_QA_tmp"):
        os.mkdir(wd + "/my_saved_model_dir_QA_tmp")
    if not os.path.exists(wd + "/my_saved_model_dir_QA_final"):
        os.mkdir(wd + "/my_saved_model_dir_QA_final")


params = list(model_Q_A.named_parameters())
print('The BERT model_Q_A has {:} different named parameters.\n'.format(len(params)))
print('======= Embedding Layer =======\n')
for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n======= First Transformer =======\n')
for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n======= Output Layer =======\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


print('\n' + 40 * '#', "Training on dataset", 40 * '#')
############################ Training the Question and answer dataset Model #########################

tb = SummaryWriter()

lrate = 1e-4
optim_pars = {'lr': lrate, 'weight_decay': 1e-3}
optimizer = Adam(model_Q_A.parameters(), **optim_pars)

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
    print("length of DataLoader: ", len(dataLoader))
    for idx, X in enumerate(dataLoader):
        model_Q_A.zero_grad()
        # number of phrases
        # X[0] is just the index
        # X[1] is the dialogue, X[1][0] are input phrases
        batch_size = len(X['input_ids'].squeeze())
        # number of tokens in a sequence
        seq_length = len(X['input_ids'][2].squeeze())
        total_phrase_pairs += batch_size
        input_tensor = torch.zeros((batch_size, seq_length), dtype=torch.long).to(device)
        token_id_tensor = torch.zeros((batch_size, seq_length), dtype=torch.long).to(device)
        attention_mask_tensor = torch.zeros((batch_size, seq_length), dtype=torch.long).to(device)
        masked_lm_labels_tensor = torch.zeros((batch_size, seq_length), dtype=torch.long).to(device)
        new_input_eval = torch.zeros((batch_size, seq_length), dtype=torch.long).to(device)
        for i in range(batch_size):
            input_tensor[i] = X['input_ids'][i].squeeze()
            token_id_tensor[i] = X['token_type_ids'][i].squeeze()
            attention_mask_tensor[i] = X['attention_mask'][i].squeeze()
            masked_lm_labels_tensor[i] = X['masked_lm_labels'][i].squeeze()
            new_input_eval[i] = X['new_input_eval'][i].squeeze()

        outputs = model_Q_A(input_ids=input_tensor, attention_mask=attention_mask_tensor, token_type_ids=token_id_tensor,
                        masked_lm_labels=masked_lm_labels_tensor)
        loss = outputs[0]
        total_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        counter += 1
        if scibert_train:
            tb.add_scalar('Loss_Bert_model_Q_A_Scibert', loss, epoch)
        else:
            tb.add_scalar('Loss_Bert_model_Q_A', loss, epoch)
        if counter % 400 == 0:
            print("*", end='')

    average_loss = total_loss / len(dataLoader)
    loss_list.append(average_loss)
    if scibert_train:
        model_Q_A.save_pretrained(wd + "/my_saved_model_dir_QA_Scibert_tmp/")
    else:
        model_Q_A.save_pretrained(wd + "/my_saved_model_dir_QA_tmp/")
    print("average loss '{}' and train time '{}'".format(average_loss, (time.time() - t0)))

tb.close()
if scibert_train:
    model_Q_A.save_pretrained(wd + "/my_saved_model_dir_QA_Scibert_final/")
else:
    model_Q_A.save_pretrained(wd + "/my_saved_model_dir_QA_final/")

