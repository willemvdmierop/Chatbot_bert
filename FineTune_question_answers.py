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


#################################
### Training Hyper-Parameters ###
#################################

max_phrase_length = 40
minibatch_size = 200
lrate = 1e-4
lrate_str = '0001'
w_decay = 1e-3
w_decay_str = '001'
epochs = 10

######## SCIBERT /ARXIV ##########
scibert_train = True ############
arxiv_train = False ##############
##################################

##########################
### Folder/File Naming ###
##########################
if scibert_train:
    dirname = 'model_scibert_lr' + lrate_str + '_wd' + w_decay_str + '_batch' + str(minibatch_size) + '_ep' + str(epochs)
    lossname = 'loss_scibert_lr' + lrate_str + '_wd' + w_decay_str + '_batch' + str(minibatch_size) + '_ep' + str(epochs)
elif arxiv_train:
    dirname = 'model_arxiv_lr' + lrate_str + '_wd' + w_decay_str + '_batch' + str(minibatch_size) + '_ep' + str(epochs)
    lossname = 'loss_arxiv_lr' + lrate_str + '_wd' + w_decay_str + '_batch' + str(minibatch_size) + '_ep' + str(epochs)
else:
    dirname = 'model_bert_lr' + lrate_str + '_wd' + w_decay_str + '_batch' + str(minibatch_size) + '_ep' + str(epochs)
    lossname = 'loss_bert_lr' + lrate_str + '_wd' + w_decay_str + '_batch' + str(minibatch_size) + '_ep' + str(epochs)
wd = os.getcwd()
dirname_final = os.path.join(wd,dirname + '_final')
dirname =  os.path.join(wd,dirname + '_tmp')

tb = SummaryWriter(log_dir = 'runs/AdamW')

###########################

start = time.time()
OOV = '<UNK>'
START_TOKEN = "<S>"
END_TOKEN = "</S>"

question_data, answer_data = utils.question_answers_dataset()

#cornell_vocab = utils.create_vocab() #use this for dynamic creation of vocabulary (but takes long time)
with open('simp_cornell_vocab.txt', 'r') as f:
   cornell_vocab = [word.rstrip('\n') for word in f]

end = time.time()
print('Total data preprocessing time is {0:.2f} and the length of the dataset is {1:d}'.format(end - start, len(question_data)))
'''
utils.print_dialogue_data_metrics(question_data, answer_data)
'''



#################################### Load the BERT tokenizer. ########################################

print('Loading BERT model...')

if os.path.exists(dirname) and len(os.listdir(dirname)) != 0:
    print("Attention we are initializing the model with an already trained tokenizer from dir: {}!".format(dirname))
    tokenizer = BertTokenizer.from_pretrained(dirname, do_lower_case=True)
    model_Q_A = BertForMaskedLM.from_pretrained(dirname)
    # !Attention we need the full size of the new vocabulary!!
    #model_Q_A.resize_token_embeddings(len(tokenizer)) #should already be extended from first initialization
else:
    if not os.path.exists(dirname): os.mkdir(dirname)    
    if scibert_train:
        print("Attention we are initializing the scibert model with extended scibert tokenizer!")
        tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
        num_added = tokenizer.add_tokens(cornell_vocab) # extend normal tokenizer with cornell vocabulary
        model_Q_A = BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased')
        # !Attention we need the full size of the new vocabulary!!
        model_Q_A.resize_token_embeddings(len(tokenizer))
    # elif arxiv_train:
    #   TODO : add loading of arxiv model
    elif arxiv_train:
        print("Attention we are initializing the arxiv model with extended arxiv tokenizer!")
        tokenizer = BertTokenizer.from_pretrained('lysandre/arxiv', do_lower_case=True)
        num_added = tokenizer.add_tokens(cornell_vocab) # extend normal tokenizer with cornell vocabulary
        model_Q_A = BertForMaskedLM.from_pretrained('lysandre/arxiv')
        # !Attention we need the full size of the new vocabulary!!
        model_Q_A.resize_token_embeddings(len(tokenizer))
    else:
        print("Attention we are initializing the bert model with extended bert tokenizer!")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        num_added = tokenizer.add_tokens(cornell_vocab) # extend normal tokenizer with cornell vocabulary
        model_Q_A = BertForMaskedLM.from_pretrained('bert-base-uncased')
        # !Attention we need the full size of the new vocabulary!!
        model_Q_A.resize_token_embeddings(len(tokenizer))

model_Q_A.to(device)

params = list(model_Q_A.named_parameters())
print('The BERT model_Q_A has {:} different named parameters.\n'.format(len(params)))
'''
print('======= Embedding Layer =======\n')
for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n======= First Transformer =======\n')
for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n======= Output Layer =======\n')
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
'''

def load_data(**train_pars):
    stage = train_pars['stage']
    data = dataset.MoviePhrasesData(questions_data= question_data, answers_data = answer_data, tokenizer = tokenizer)
    train_dataset_params = {'batch_size': minibatch_size, 'shuffle': True}
    dataloader = DataLoader(data, **train_dataset_params)
    return dataloader

load_data_pars = {'stage': 'train', 'num_workers': 3}
dataLoader = load_data(**load_data_pars)  # this returns a dataloader
#print('\n' + 40 * '#', "Loading the Bert Tokenizer", 40 * '#')


############################ Training the Question and answer dataset Model #########################
print('\n' + 40 * '#', "Training on dataset", 40 * '#')


optim_pars = {'lr': lrate, 'weight_decay': w_decay}
optimizer = AdamW(model_Q_A.parameters(), **optim_pars)

current_batch = 0
total_phrase_pairs = 0
loss_list = []
e = 0
if os.path.exists(os.path.join(dirname,'checkpoint.pth')):
    checkpoint = torch.load(os.path.join(dirname,'checkpoint.pth'))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    e = checkpoint['epoch'] + 1
    loss_list = checkpoint['loss_list']

for epoch in range(e, epochs):
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
        tb.add_scalar(lossname, loss, epoch)
        if counter % 400 == 0:
            print("*", end='')

    average_loss = total_loss / len(dataLoader)
    loss_list.append(average_loss)
    tokenizer.save_pretrained(dirname)
    model_Q_A.save_pretrained(dirname)
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_list':loss_list,
    }, os.path.join(dirname,'checkpoint.pth'))
    print("average loss '{}' and train time '{}' min".format(average_loss, (time.time() - t0)/60))

tb.close()
### Save final trained model/optimizer
if not os.path.exists(dirname_final): os.mkdir(dirname_final) 
tokenizer.save_pretrained(dirname_final)
model_Q_A.save_pretrained(dirname_final)

