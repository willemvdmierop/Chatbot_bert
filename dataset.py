import torch
import torch.utils
from torch.utils import data
import utils
import re
from transformers import BertTokenizer


class MoviePhrasesData(data.Dataset):

    # voc: vocabulary, word:idx
    # all dialogues:
    def __init__(self, all_dialogues, max_seq_len = 40, unk_token = '<UNK>', start_token = '<S>', end_token = '</S>', sep_token = '<SEP>'):
        # why init superclass? access methods from data.Dataset
        # no need to rewrite methods from the superclass
        # first argument: subclass, second: instance of a subclass
        # in Python3 this is equivalent to super()
        super(MoviePhrasesData, self).__init__()
        self.max_seq_len = max_seq_len
        #self.voc = voc
        self.all_dialogues = all_dialogues
        self.unk_token = unk_token
        self.end_token = end_token
        self.start_token = start_token
        self.sep_token = sep_token
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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

        tokenizer = self.tokenizer

        for k in keys:
            input_phrase = []
            output_phrase = []
            # tokenize here, both key and reply
            tokenized_k = re.sub('\W+', ' ', k).split()
            tokenized_r = re.sub('\W+', ' ', dial[k]).split() #dial is a dict so dial[k] returns the value associated with k
            
            tokenized_in = tokenized_k + [self.sep_token] + tokenized_r
            if len(tokenized_in) > self.max_seq_len:
                tokenized_in = tokenized_in[:self.max_seq_len]
                tokenized_in.append(self.end_token)
            elif len(tokenized_in) <= self.max_seq_len:
                tokenized_in.append(self.end_token)
                seq_pad = (self.max_seq_len - len(tokenized_in) + 1) * [self.unk_token]
                if len(seq_pad) > 0:
                    tokenized_in.extend(seq_pad)
            
            kwargs ={   'text': re.sub('\W+', ' ', k), 
                        'text_pair': re.sub('\W+', ' ', dial[k]),
                        'add_special_tokens': True,
                        'max_length': 40,
                        'pad_to_max_length': True,
                        'return_tensors': 'pt',
                        'return_token_type_ids': True,
                        'return_attention_mask': True,
                        'return_special_tokens_mask': True}
            input_phrase = tokenizer.encode_plus(**kwargs)
            masked_lm_labels = -100*(torch.ones(self.max_seq_len)- input_phrase['token_type_ids'])
            input_phrase['masked_lm_labels'] = masked_lm_labels

            all_inputs.append(input_phrase)

        #all_inputs = torch.stack(all_inputs)

        return all_inputs

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

