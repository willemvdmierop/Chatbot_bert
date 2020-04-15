from transformers import AutoTokenizer, AutoModel, BertForMaskedLM, AutoModelWithLMHead, BertTokenizer
import torch
from torch.utils.tensorboard import SummaryWriter
import utils_generation as ugen
import pandas as pd
import numpy as np
import time
import utils
import bert_score
from bert_score import score, BERTScorer
import pickle

device = 'cpu'
cuda = False
if (torch.cuda.is_available()):
    device = torch.device('cuda')
    cuda = True

##### calculate indexes ########
### Questions: #################
Q1 = "who is she?".lower()
Q2 = "are you okay?".lower()
Q3 = "why?".lower()
###############################
######## data #################
question_data, answer_data = utils.question_answers_dataset()
###############################
index_Q1 = []
index_Q2 = []
index_Q3 = []

count_Q1 = 0
count_Q2 = 0
count_Q3 = 0
t0 = time.time()

print(len(question_data))
scorer = BERTScorer(model_type='bert-base-uncased')
print('test_score:' , scorer.score(["are you okay?"],[["are you good?"]]))

for i in range(len(question_data)):
    if i%50==0: print("*", end='')
    P, R, F1 = scorer.score([Q1], [question_data[i]])
    if F1.item() > 0.9:
        count_Q1 += 1
        
        index_Q1.append(i)
    P, R, F1 = scorer.score([Q2], [question_data[i]])
    if F1.item() > 0.9:
        count_Q2 += 1
        
        index_Q2.append(i)
    P, R, F1 = scorer.score([Q3], [question_data[i]])
    if F1.item() > 0.9:
        count_Q3 += 1
        
        index_Q3.append(i)
    if i % 1000 == 0:
        print('\nwe have found {} similar questions in total'.format(count_Q1+count_Q2+count_Q3))
        print('calculating took {} seconds for 100'.format(time.time() - t0))
        with open('index_Q.pkl','wb') as myfile:
            pickle.dump([index_Q1,index_Q2,index_Q3], myfile)
        t0 = time.time()
print('we have found {} similar questions to Q1'.format(count_Q1))
print('we have found {} similar questions to Q1'.format(count_Q2))
print('we have found {} similar questions to Q1'.format(count_Q3))


with open('index_Q.pkl','wb') as myfile:
    pickle.dump({'q1': index_Q1,'q2':index_Q2, 'q3':index_Q3}, myfile)