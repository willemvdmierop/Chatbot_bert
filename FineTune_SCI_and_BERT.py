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
import numpy as np
import pandas as pd
import pickle
# import utility and dataset files
import utils_generation as ugen
import utils
import Dataset_Q_A as dataset

device = 'cpu'
cuda = False
if (torch.cuda.is_available()):
    device = torch.device('cuda')
    cuda = True


# ============================================================================================================ #
# ========================================= Training Hyper-Parameters ======================================== #
# ============================================================================================================ #

max_phrase_length = 40
minibatch_size = 200
lrate = 1e-4
lrate_str = '001'
w_decay = 1e-2
w_decay_str = '001'
epochs = 60
max_grad_norm = 1.0

### Training with Scibert (True) of with original Bert (False) ###
scibert_train = False  

### Choose to apply gradient clipping or learning rate schedules
Gradient_clipping_on = True 
Schedule_ON = False  


# ============================================================================================================ #
# ========================================= Filename/Directory Naming Schemes ================================ #
# ============================================================================================================ #

if scibert_train:
    dirname = 'model_scibert_lr' + lrate_str + '_wd' + w_decay_str + '_batch' + str(minibatch_size) + '_ep' + str(epochs) + '_mPlenght' + str(max_phrase_length) + '_grad_clip_' + str(Gradient_clipping_on) + '_Schedule_' + str(Schedule_ON)
    lossname = 'loss_scibert_lr' + lrate_str + '_wd' + w_decay_str + '_batch' + str(minibatch_size) + '_ep' + str(epochs) + '_mPlenght' + str(max_phrase_length) + '_grad_clip_' + str(Gradient_clipping_on) + '_Schedule_' + str(Schedule_ON)
else:
    dirname = 'model_bert_lr' + lrate_str + '_wd' + w_decay_str + '_batch' + str(minibatch_size) + '_ep' + str(epochs) + '_mPlenght' + str(max_phrase_length) + '_grad_clip_' + str(Gradient_clipping_on) + '_Schedule_' + str(Schedule_ON)
    lossname = 'loss_bert_lr' + lrate_str + '_wd' + w_decay_str + '_batch' + str(minibatch_size) + '_ep' + str(epochs) + '_mPlenght' + str(max_phrase_length) + '_grad_clip_' + str(Gradient_clipping_on) + '_Schedule_' + str(Schedule_ON)
wd = os.getcwd()
dirname_final = os.path.join(wd,dirname + '_final')
dirname =  os.path.join(wd,dirname + '_tmp')

tb = SummaryWriter(log_dir = 'runs/AdamW')


# ============================================================================================================ #
# ========================================= Load BERT pre-trained model ====================================== #
# ============================================================================================================ #

print('Loading BERT model...')

### Continue fine-tuning
if os.path.exists(os.path.join(dirname, 'pytorch_model.bin')) and os.path.exists(os.path.join(dirname, 'tokenizer_config.json')):
    print("Attention we are initializing the model with an already trained tokenizer from dir: {}!".format(dirname))
    tokenizer = BertTokenizer.from_pretrained(dirname, do_lower_case=True)
    model_Q_A = BertForMaskedLM.from_pretrained(dirname)
else:
    if not os.path.exists(dirname): os.mkdir(dirname)    
    if scibert_train:
        print("Attention we are initializing the scibert model with scibert tokenizer!")
        tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
        model_Q_A = BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased')
    else:
        print("Attention we are initializing the bert model with bert tokenizer!")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        model_Q_A = BertForMaskedLM.from_pretrained('bert-base-uncased')

model_Q_A.to(device)

params = list(model_Q_A.named_parameters())
print('The BERT model_Q_A has {:} different named parameters.\n'.format(len(params)))


# ============================================================================================================ #
# ========================================= Load Cornell dataset ============================================= #
# ============================================================================================================ #

start = time.time()

### Load Question-Answers (phrase pairs) dataset
question_data, answer_data = utils.question_answers_dataset()

### Save the dataset in a csv format
# utils.save_data_csv(question_data,answer_data)

### Print analysis of dataset
# utils.print_dialogue_data_metrics(question_data, answer_data)

def load_data(**train_pars):
    stage = train_pars['stage']
    data = dataset.MoviePhrasesData(questions_data= question_data, answers_data = answer_data, tokenizer = tokenizer)
    train_dataset_params = {'batch_size': minibatch_size, 'shuffle': True}
    dataloader = DataLoader(data, **train_dataset_params)
    return dataloader

### Transform dataset into an iteratable dataloader
load_data_pars = {'stage': 'train', 'num_workers': 3}
dataLoader = load_data(**load_data_pars)

end = time.time()
print('Total data preprocessing time is {0:.2f} and the length of the dataset is {1:d}'.format(end - start, len(question_data)))


# ============================================================================================================ #
# ========================================= Metrics during training ========================================== #
# ============================================================================================================ #

### Parameters used for word generation for training evaluation
n_samples = 10
max_len = 20
top_k = 50
temperature = 1.5

### Scorer which returns Precision, Recall, and F1 scores
scorer = BERTScorer(model_type='bert-base-uncased')

### Load references for potential answers to the three questions used for evaluation
q_refs = pickle.load(open('Metrics_files/Q_refs.pkl', 'rb'))
q3_refs = q_refs['q3_refs']
q2_refs = q_refs['q2_refs']
q1_refs = q_refs['q1_refs']
all_q_refs = [q1_refs,q2_refs,q3_refs]
all_q_cands = ['Who is she?', 'Are you okay?', 'Why?']


# ============================================================================================================ #
# ========================================= Optimizer and scheduler ========================================== #
# ============================================================================================================ #

optim_pars = {'lr': lrate, 'weight_decay': w_decay}
optimizer = AdamW(model_Q_A.parameters(), **optim_pars)
if Schedule_ON:
    num_training_steps = len(dataLoader)*epochs
    num_warmup_steps = int(len(dataLoader)*epochs / 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps= num_training_steps)
    print("\n========= attention we are using a cosine with warmup lr schedule =========")

### Tracking mechanisms for metrics
current_batch = 0
total_phrase_pairs = 0
loss_list = []
e = 0
Q_metrics = [[],[],[]]

### Load optimizer and scheduler if continuing finetuning
if os.path.exists(os.path.join(dirname,'checkpoint.pth')):
    checkpoint = torch.load(os.path.join(dirname,'checkpoint.pth'))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if Schedule_ON:
        scheduler.load_state_dict(checkpoint['schedule_state_dict'])
    e = checkpoint['epoch'] + 1
    loss_list = checkpoint['loss_list']

### Load metrics if continuing finetuning
if os.path.exists(os.path.join(dirname,'metrics.pkl')):
    metrics = torch.load(os.path.join(dirname,'metrics.pkl'))
    Q_metrics = metrics['q_metrics']


# ============================================================================================================ #
# ========================================= Training the model =============================================== #
# ============================================================================================================ #

print('\n' + 40 * '#', "Training on dataset", 40 * '#')
print("length of DataLoader: ", len(dataLoader))
print("will evaluate every ", int(len(dataLoader)/42), ' batches')

### Training Loop
for epoch in range(e, epochs):
    t0 = time.time()
    print(30 * "#" + ' Training ' + 30 * "#")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))

    ### Reset epoch metrics
    total_loss = 0
    counter = 0

    ### iterate through dataset by batch
    for idx, X in enumerate(dataLoader):
        model_Q_A.train()
        model_Q_A.zero_grad()
        
        batch_size = len(X['input_ids'].squeeze())
        seq_length = len(X['input_ids'][2].squeeze()) # number of tokens in a sequence
        total_phrase_pairs += batch_size

        ### Construct input to the model
        input_tensor = torch.zeros((batch_size, seq_length), dtype=torch.long).to(device)
        token_id_tensor = torch.zeros((batch_size, seq_length), dtype=torch.long).to(device)
        attention_mask_tensor = torch.zeros((batch_size, seq_length), dtype=torch.long).to(device)
        masked_lm_labels_tensor = torch.zeros((batch_size, seq_length), dtype=torch.long).to(device)
        for i in range(batch_size):
            input_tensor[i] = X['input_ids'][i].squeeze()
            token_id_tensor[i] = X['token_type_ids'][i].squeeze()
            attention_mask_tensor[i] = X['attention_mask'][i].squeeze()
            masked_lm_labels_tensor[i] = X['masked_lm_labels'][i].squeeze()

        ### Feed input to model and get output
        outputs = model_Q_A(input_ids=input_tensor, attention_mask=attention_mask_tensor, token_type_ids=token_id_tensor,
                        masked_lm_labels=masked_lm_labels_tensor)

        ### Back-propagate loss
        loss = outputs[0]
        total_loss += loss
        loss.backward()

        ### Apply gradient clipping
        if Gradient_clipping_on:
            torch.nn.utils.clip_grad_norm_(model_Q_A.parameters(), max_grad_norm)

        ### Optimizer/LR Scheduler step
        optimizer.step()
        if Schedule_ON:
            scheduler.step()
        
        ### Tensorboard logging
        tb.add_scalar(lossname, loss, epoch)
        
        ### Evaluate model 42 times in a epoch
        counter += 1
        if counter % int(len(dataLoader)/42) == 0:
            ### Pass the trained tokenizer to the generation utility file
            ugen.load_model_tokenizer(model_path=model_Q_A, tokenizer_path=tokenizer, is_path=False)

            print('Metrics (BLEU, P, R, F1, epoch, counter)')

            ### Get scores for all three questions (candidates)
            for i in range(len(all_q_cands)):
                seed_text = tokenizer.tokenize(all_q_cands[i].lower())
                refs = all_q_refs[i]
                bleu, P, R, F1 = utils.return_metrics(scorer=scorer, refs=refs, seed_text=seed_text,
                                                        n_samples=n_samples, top_k=top_k,
                                                        temperature=temperature, max_len=max_len, cuda=cuda)
                Q_metrics[i].append([bleu, P, R, F1,epoch, counter])
                print('Q'+str(i+1),' Metrics: ', Q_metrics[i][-1])

            ### Set model to training mode before continuing
            model_Q_A.train()    
    
    ### Save model and tokenizer
    tokenizer.save_pretrained(dirname)
    model_Q_A.save_pretrained(dirname)

    ### Save loss, optimizer, and LR scheduler
    average_loss = total_loss / len(dataLoader)
    loss_list.append(average_loss)
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

    ### Save evaluation scores
    metrics = {'q_metrics': Q_metrics}
    torch.save(metrics, os.path.join(dirname,'metrics.pkl'))
        
    print("average loss '{}' and train time '{}' min".format(average_loss, (time.time() - t0)/60))
### End of Training ###

tb.close()

### Save final trained model/optimizer/metrics
if not os.path.exists(dirname_final): os.mkdir(dirname_final) 
tokenizer.save_pretrained(dirname_final)
model_Q_A.save_pretrained(dirname_final)
metrics = {'q_metrics': Q_metrics}
torch.save(metrics, os.path.join(dirname_final,'metrics.pkl'))

