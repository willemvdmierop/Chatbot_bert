import torch
import torch.utils
from torch.utils import data
import utils
import re


class MoviePhrasesData(data.Dataset):

    # voc: vocabulary, word:idx
    # all dialogues:
    def __init__(self, max_seq_len, voc, all_dialogues, unk_token, start_token, end_token):
        # why init superclass? access methods from data.Dataset
        # no need to rewrite methods from the superclass
        # first argument: subclass, second: instance of a subclass
        # in Python3 this is equivalent to super()
        super(MoviePhrasesData, self).__init__()
        self.max_seq_len = max_seq_len
        self.voc = voc
        self.all_dialogues = all_dialogues
        self.unk_token = unk_token
        self.end_token = end_token
        self.start_token = start_token

    # loads one full dialogue (K phrases in a dialogue), an OrderedDict
    # If there are a total K phrases, the data point will be
    # ((K-1) x (MAX_SEQ + 2), (K-1) x (MAX_SEQ + 2))
    # zero-pad after EOS
    def load_dialogue(self, dialogue):
        dial = dialogue
        # k: phrase
        all_inputs = []
        all_outputs = []
        # get keys (first phrase) from the dialogues
        keys = dial.keys()
        for k in keys:
            input_phrase = []
            output_phrase = []
            # tokenize here, both key and reply
            tokenized_k = re.sub('\W+', ' ', k).split()
            tokenized_r = re.sub('\W+', ' ', dial[k]).split() #dial is a dict so dial[k] returns the value associated with k
            # pad or truncate, both key and reply
            if len(tokenized_k) > self.max_seq_len:
                tokenized_k = tokenized_k[:self.max_seq_len]
                tokenized_k.append(self.end_token)
            elif len(tokenized_k) <= self.max_seq_len:
                tokenized_k.append(self.end_token)
                seq_pad = (self.max_seq_len - len(tokenized_k) + 1) * [self.unk_token]
                if len(seq_pad) > 0:
                    tokenized_k.extend(seq_pad)

            # reply
            if len(tokenized_r) > self.max_seq_len:
                tokenized_r = tokenized_r[:self.max_seq_len]
                tokenized_r.append(self.end_token)
            elif len(tokenized_r) <= self.max_seq_len:
                tokenized_r.append(self.end_token)
                seq_pad = (self.max_seq_len - len(tokenized_r) + 1) * [self.unk_token]
                if len(seq_pad) > 0:
                    tokenized_r.extend(seq_pad)

            # convert to indices - key
            for w in tokenized_k:
                idx = self.voc[w]
                input_phrase.append(idx)
            # add start/end sentence token - key
            input_phrase.insert(0, self.voc[self.start_token])
            # convert to indices - reply
            for w in tokenized_r:
                idx = self.voc[w]
                output_phrase.append(idx)
            # add start/end sentence token - reply
            output_phrase.insert(0, self.voc[self.start_token])
            # append to the inputs and outputs
            all_inputs.append(torch.tensor(input_phrase))
            all_outputs.append(torch.tensor(output_phrase))

        all_inputs = torch.stack(all_inputs)
        all_outputs = torch.stack(all_outputs)
        # return a tuple?
        output_tuple = (all_inputs, all_outputs)
        return output_tuple

    # number of dialogues, 83097
    def __len__(self):
        return len(self.all_dialogues)

    # x: input sequence
    # y: output sequence
    # idx:
    # output: tuple of two torch tensor stacks size (K-1)x max_seq_len
    def __getitem__(self, idx):
        self.dialogue = self.all_dialogues[idx]
        self.phrase = self.load_dialogue(self.dialogue)
        return idx, self.phrase
