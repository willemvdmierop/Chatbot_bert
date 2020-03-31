from __future__ import absolute_import, division, print_function, unicode_literals

import collections
from io import open
import transformers
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import AutoTokenizer


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab

vocab_corn = load_vocab("vocab_cornell.txt")
print(len(vocab_corn))

vocab_sci = load_vocab("/Users/willemvandemierop/Documents/Master AI/Pycharm/Chatbot_bert/scibert_scivocab_uncased/vocab.txt")
print(len(vocab_sci))

tokens_to_add = []
for _, value in enumerate(vocab_corn):
    if value not in vocab_sci:
        tokens_to_add.append(value)

tokenizer = AutoTokenizer.from_pretrained("/Users/willemvandemierop/Documents/Master AI/Pycharm/Chatbot_bert/scibert_scivocab_uncased")
num_added_toks = tokenizer.add_tokens(tokens_to_add)
print('We have added', num_added_toks, 'tokens')

tokenizer.save_pretrained("/Users/willemvandemierop/Documents/Master AI/Pycharm/Chatbot_bert/scibert_scivocab_uncased")
print("We have updated the scibert Tokenizer, let's see what the new length is of the vocab file")

tokenizer = AutoTokenizer.from_pretrained("/Users/willemvandemierop/Documents/Master AI/Pycharm/Chatbot_bert/scibert_scivocab_uncased")
print("the vocab size is:" , tokenizer.vocab_size)

print("we tokenize a sentence", tokenizer.tokenize("hello how are you doing ? My name is willem and I am your creator. You will not harm a living object"))