import collections
import os, sys, re
import time
import torch
import torchvision
import numpy as np
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
import time


#text = "Wagner was a German composer, his inoculation was ambiguous. Most of his life he was deeply depressed, because he had too much serotonin".lower().split()
text = "what is an atom ?"
text = text.lower().split()

tokenizer_sci = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
vocab_sci = tokenizer_sci.get_vocab()
#tokenizer_sci.save_vocabulary("vocab_scibert.txt")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_bert = tokenizer.get_vocab()
#tokenizer.save_vocabulary("vocab_bert.txt")

tokenizer_arxiv = AutoTokenizer.from_pretrained("lysandre/arxiv")
vocab_arxiv = tokenizer.get_vocab()
#tokenizer.save_vocabulary("vocab_arxiv.txt")

print('Bert vocab has a length of {} and Scibert has a length of {}'.format(len(vocab_bert), len(vocab_sci)))

count_bert = 0
count_sci = 0
count_arxiv = 0
expert = False
for txt in text:
    if txt == "expert":
        print("This person wants to talk to an expert!")
        expert = True
    if txt in vocab_bert:
        count_bert +=  1
    if txt in vocab_sci:
        count_sci += 1
    if txt in vocab_arxiv:
        count_arxiv += 1

print("\nThe sentence we received was: ", text)
print("Base on the sentence, the person wants an expert: ", expert)
print('In this sentence we counted {} words in bert vocab, {} words in scibert and {} words in Arxiv'.format(count_bert,count_sci, count_arxiv))