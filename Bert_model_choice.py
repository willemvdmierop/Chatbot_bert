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


text = "Wagner was a German composer".lower().split()


tokenizer_sci = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
vocab_sci = tokenizer_sci.get_vocab()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_bert = tokenizer.get_vocab()

np.savetxt("vocab_bert.txt", vocab_bert)
count_bert = 0
count_sci = 0
for txt in text:
    if txt in vocab_bert:
        count_bert +=  1
    if txt in vocab_sci:
        count_sci += 1
