from transformers import AutoTokenizer, AutoModel, BertForMaskedLM, AutoModelWithLMHead, BertTokenizer
import torch
from torch.utils.tensorboard import SummaryWriter
import utils_generation as ugen
import pandas as pd
import numpy as np
import time
import utils
import bert_score
from bert_score import score

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
for i in range(len(question_data)):
    print("*", end='')
    P, R, F1 = score([Q1], [question_data[i]], lang='en')
    if F1.item() > 0.9:
        count_Q1 += 1
        print('we have found {} similar questions to Q1'.format(count_Q1))
        index_Q1.append(i)
    P, R, F1 = score([Q2], [question_data[i]], lang='en')
    if F1.item() > 0.9:
        count_Q2 += 1
        print('we have found {} similar questions to Q1'.format(count_Q2))
        index_Q2.append(i)
    P, R, F1 = score([Q3], [question_data[i]], lang='en')
    if F1.item() > 0.9:
        count_Q3 += 1
        print('we have found {} similar questions to Q1'.format(count_Q3))
        index_Q3.append(i)
    if i % 100 == 0:
        print('we have found {} similar questions in total'.format(count_Q1+count_Q2+count_Q3))
        print('calculating took {} seconds'.format(time.time() - t0))
        t0 = time.time()

