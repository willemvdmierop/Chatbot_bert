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
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
import bert_score
from bert_score import BERTScorer
import utils_generation as ugen
# load the dataset interface
import utils
import dataset_Q_A as dataset
import numpy as np
import pandas as pd
import pickle
device = 'cpu'
cuda = False
if (torch.cuda.is_available()):
    device = torch.device('cuda')
    cuda = True

# ========================================= Training Hyper-Parameters ======================================== #
#################################
### Training Hyper-Parameters ###
#################################

max_phrase_length = 40
minibatch_size = 200
lrate = 1e-4
lrate_str = '001'
w_decay = 1e-2
w_decay_str = '001'
epochs = 60

######## SCIBERT /ARXIV ##########
scibert_train = False ############
arxiv_train = False ##############
##################################
######## SCIBERT /ARXIV ##########
Gradient_clipping_on = True ######
Schedule_ON = False  #############
##################################

##########################
### Folder/File Naming ###
##########################
if scibert_train:
    dirname = 'model_scibert_lr' + lrate_str + '_wd' + w_decay_str + '_batch' + str(minibatch_size) + '_ep' + str(epochs) + '_mPlenght' + str(max_phrase_length) + '_grad_clip_' + str(Gradient_clipping_on) + '_Schedule_' + str(Schedule_ON)
    lossname = 'loss_scibert_lr' + lrate_str + '_wd' + w_decay_str + '_batch' + str(minibatch_size) + '_ep' + str(epochs) + '_mPlenght' + str(max_phrase_length) + '_grad_clip_' + str(Gradient_clipping_on) + '_Schedule_' + str(Schedule_ON)
elif arxiv_train:
    dirname = 'model_arxiv_lr' + lrate_str + '_wd' + w_decay_str + '_batch' + str(minibatch_size) + '_ep' + str(epochs) + '_mPlenght' + str(max_phrase_length) + '_grad_clip_' + str(Gradient_clipping_on) + '_Schedule_' + str(Schedule_ON)
    lossname = 'loss_arxiv_lr' + lrate_str + '_wd' + w_decay_str + '_batch' + str(minibatch_size) + '_ep' + str(epochs) + '_mPlenght' + str(max_phrase_length) + '_grad_clip_' + str(Gradient_clipping_on) + '_Schedule_' + str(Schedule_ON)
else:
    dirname = 'model_bert_lr' + lrate_str + '_wd' + w_decay_str + '_batch' + str(minibatch_size) + '_ep' + str(epochs) + '_mPlenght' + str(max_phrase_length) + '_grad_clip_' + str(Gradient_clipping_on) + '_Schedule_' + str(Schedule_ON)
    lossname = 'loss_bert_lr' + lrate_str + '_wd' + w_decay_str + '_batch' + str(minibatch_size) + '_ep' + str(epochs) + '_mPlenght' + str(max_phrase_length) + '_grad_clip_' + str(Gradient_clipping_on) + '_Schedule_' + str(Schedule_ON)
wd = os.getcwd()
dirname_final = os.path.join(wd,dirname + '_final')
dirname =  os.path.join(wd,dirname + '_tmp')

tb = SummaryWriter(log_dir = 'runs/AdamW')


# ========================================= Load Bert tokenizer ======================================== #

print('Loading BERT model...')

if os.path.exists(os.path.join(dirname, 'pytorch_model.bin')) and os.path.exists(os.path.join(dirname, 'tokenizer_config.json')):
    print("Attention we are initializing the model with an already trained tokenizer from dir: {}!".format(dirname))
    tokenizer = BertTokenizer.from_pretrained(dirname, do_lower_case=True)
    model_Q_A = BertForMaskedLM.from_pretrained(dirname)
    # !Attention we need the full size of the new vocabulary!!
    #model_Q_A.resize_token_embeddings(len(tokenizer)) #should already be extended from first initialization
else:
    if not os.path.exists(dirname): os.mkdir(dirname)    
    if scibert_train:
        print("Attention we are initializing the scibert model with scibert tokenizer!")
        tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
        #num_added = tokenizer.add_tokens(cornell_vocab) # extend normal tokenizer with cornell vocabulary
        model_Q_A = BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased')
        # !Attention we need the full size of the new vocabulary!!
        #model_Q_A.resize_token_embeddings(len(tokenizer))
    # elif arxiv_train:
    #   TODO : add loading of arxiv model
    elif arxiv_train:
        print("Attention we are initializing the arxiv model with arxiv tokenizer!")
        tokenizer = BertTokenizer.from_pretrained('lysandre/arxiv', do_lower_case=True)
        #num_added = tokenizer.add_tokens(cornell_vocab) # extend normal tokenizer with cornell vocabulary
        model_Q_A = BertForMaskedLM.from_pretrained('lysandre/arxiv')
        # !Attention we need the full size of the new vocabulary!!
        #model_Q_A.resize_token_embeddings(len(tokenizer))
    else:
        print("Attention we are initializing the bert model with bert tokenizer!")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        #num_added = tokenizer.add_tokens(cornell_vocab) # extend normal tokenizer with cornell vocabulary
        model_Q_A = BertForMaskedLM.from_pretrained('bert-base-uncased')
        # !Attention we need the full size of the new vocabulary!!
        #model_Q_A.resize_token_embeddings(len(tokenizer))

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

# ================================================ Load Cornell dataset ======================================== #

start = time.time()
OOV = '<UNK>'
START_TOKEN = "<S>"
END_TOKEN = "</S>"

question_data, answer_data = utils.question_answers_dataset()

# utils.save_data_csv(question_data,answer_data)
end = time.time()
print('Total data preprocessing time is {0:.2f} and the length of the dataset is {1:d}'.format(end - start, len(question_data)))
'''
utils.print_dialogue_data_metrics(question_data, answer_data)
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
# ========================================== Metrics during training ======================================== #

n_samples = 10
max_len = 20
top_k = 50
temperature = 1.5
#question = 1

scorer = BERTScorer(model_type='bert-base-uncased')
q_refs = pickle.load(open('Q_refs.pkl', 'rb'))
q3_refs = q_refs['q3_refs']
q2_refs = q_refs['q2_refs']
q1_refs = q_refs['q1_refs']
all_q_refs = [q1_refs,q2_refs,q3_refs]
all_q_cands = ['Who is she?', 'Are you okay?', 'Why?']



# ========================================= Optimizer and scheduler ======================================== #

optim_pars = {'lr': lrate, 'weight_decay': w_decay}
optimizer = AdamW(model_Q_A.parameters(), **optim_pars)
if Schedule_ON:
    num_training_steps = len(dataLoader)*epochs
    num_warmup_steps = int(len(dataLoader)*epochs / 10)
    #scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                                   #num_training_steps= num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps= num_training_steps)
    print("\n========= attention we are using a cosin with hard restarts lr schedule =========")

current_batch = 0
total_phrase_pairs = 0
loss_list = []
e = 0
Q_metrics = [[],[],[]]
if os.path.exists(os.path.join(dirname,'checkpoint.pth')):
    checkpoint = torch.load(os.path.join(dirname,'checkpoint.pth'))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if Schedule_ON:
        scheduler.load_state_dict(checkpoint['schedule_state_dict'])
    e = checkpoint['epoch'] + 1
    loss_list = checkpoint['loss_list']
if os.path.exists(os.path.join(dirname,'metrics.pkl')):
    metrics = torch.load(os.path.join(dirname,'metrics.pkl'))
    Q_metrics = metrics['q_metrics']

# ============================================ Training the model ======================================== #
print('\n' + 40 * '#', "Training on dataset", 40 * '#')
max_grad_norm = 1.0
for epoch in range(e, epochs):
    t0 = time.time()
    print(30 * "#" + ' Training ' + 30 * "#")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    total_loss = 0
    counter = 0
    print("length of DataLoader: ", len(dataLoader))
    print("will evaluate every ", int(len(dataLoader)/42), ' batches')
    for idx, X in enumerate(dataLoader):
        model_Q_A.train()
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
        if Gradient_clipping_on:
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model_Q_A.parameters(), max_grad_norm)
        optimizer.step()
        if Schedule_ON:
            scheduler.step()
        #optimizer.zero_grad() #not necessary since optimizer made from model parameters and we zero_grad the model at start of iteration
        counter += 1
        tb.add_scalar(lossname, loss, epoch)
        if counter % int(len(dataLoader)/42) == 0:
            ugen.load_model_tokenizer(model_path=model_Q_A, tokenizer_path=tokenizer, is_path=False)
            print('Metrics (BLEU, P, R, F1, epoch, counter)')
            for i in range(len(all_q_cands)):
                seed_text = tokenizer.tokenize(all_q_cands[i].lower())
                refs = all_q_refs[i]
                bleu, P, R, F1 = utils.return_metrics(scorer=scorer, refs=refs, seed_text=seed_text,
                                                        n_samples=n_samples, top_k=top_k,
                                                        temperature=temperature, max_len=max_len, cuda=cuda)
                Q_metrics[i].append([bleu, P, R, F1,epoch, counter])
                print('Q'+str(i+1),' Metrics: ', Q_metrics[i][-1])
            model_Q_A.train()
        
    
    ### End of Epoch ###
    
    average_loss = total_loss / len(dataLoader)
    loss_list.append(average_loss)
    tokenizer.save_pretrained(dirname)
    model_Q_A.save_pretrained(dirname)
    if Schedule_ON:
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'schedule_state_dict': scheduler.state_dict(),
            'loss_list':loss_list,
        }, os.path.join(dirname,'checkpoint.pth'))
    else:
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_list': loss_list,
        }, os.path.join(dirname, 'checkpoint.pth'))

    
    metrics = {'q_metrics': Q_metrics} #{'q1_metrics': Q1_metrics 'q2_metrics': Q2_metrics, 'q3_metrics': Q3_metrics}
    torch.save(metrics, os.path.join(dirname,'metrics.pkl'))
        
    print("average loss '{}' and train time '{}' min".format(average_loss, (time.time() - t0)/60))
### End of Training ###


tb.close()
### Save final trained model/optimizer
if not os.path.exists(dirname_final): os.mkdir(dirname_final) 
tokenizer.save_pretrained(dirname_final)
model_Q_A.save_pretrained(dirname_final)
metrics = {'q_metrics': Q_metrics} #{'q1_metrics': Q1_metrics 'q2_metrics': Q2_metrics, 'q3_metrics': Q3_metrics}
torch.save(metrics, os.path.join(dirname_final,'metrics.pkl'))

