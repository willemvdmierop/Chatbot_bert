import torch
import torch.utils
from torch.utils import data
import utils
import re
from transformers import BertTokenizer
import copy


class MoviePhrasesData(data.Dataset):

    # voc: vocabulary, word:idx
    # all dialogues:
    def __init__(self, all_dialogues, scibert=False, max_seq_len=40, unk_token='<UNK>', start_token='<S>',
                 end_token='</S>',
                 sep_token='<SEP>'):
        # why init superclass? access methods from data.Dataset
        # no need to rewrite methods from the superclass
        # first argument: subclass, second: instance of a subclass
        # in Python3 this is equivalent to super()
        super(MoviePhrasesData, self).__init__()
        self.max_seq_len = max_seq_len
        # self.voc = voc
        self.all_dialogues = all_dialogues
        self.unk_token = unk_token
        self.end_token = end_token
        self.start_token = start_token
        self.sep_token = sep_token
        if scibert:
            self.tokenizer = BertTokenizer.from_pretrained("./scibert_scivocab_uncased")  # make sure this
            # is our scibert combined with our cornell tokenizer
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # loads one full dialogue (K phrases in a dialogue), an OrderedDict
    # If there are a total K phrases, the data point will be
    # ((K-1) x (MAX_SEQ + 2), (K-1) x (MAX_SEQ + 2))
    # zero-pad after EO

    def load_dialogue(self, dialogue):
        dial = dialogue
        all_inputs = []
        # k: phrase
        # get keys (first phrase) from the dialogues
        keys = dial.keys()
        tokenizer = self.tokenizer
        for k in keys:
            # tokenize here, both key and reply
            # tokenized_k = tokenizer.tokenize(k)
            # tokenized_r = tokenizer.tokenize(dial[k])  # dial is a dict so dial[k] returns the value associated with k
            kwargs = {'text': k,
                      'text_pair': dial[k],
                      'max_length': 40,
                      'pad_to_max_length': True,
                      'add_special_tokens': True,
                      'return_tensors': 'pt',
                      'return_token_type_ids': True,
                      'return_attention_mask': True,
                      'return_special_tokens_mask': True}
            input_phrase = tokenizer.encode_plus(**kwargs)
            masked_lm_labels = -100 * (
                        torch.ones(len(input_phrase['attention_mask'])) - input_phrase['token_type_ids'] == 1)
            input_phrase['attention_mask'] = copy.deepcopy(masked_lm_labels) / -100
            new_input_eval = copy.deepcopy(input_phrase['input_ids'])
            new_input_eval[(input_phrase['attention_mask'] - input_phrase[
                'token_type_ids']) == 0] = tokenizer.convert_tokens_to_ids('[MASK]')
            input_phrase['masked_lm_labels'] = masked_lm_labels
            input_phrase['new_input_eval'] = new_input_eval

            all_inputs.append(input_phrase)

        # all_inputs = torch.stack(all_inputs)

        return all_inputs

    # number of dialogues, 83097
    def __len__(self):
        return len(self.all_dialogues)

    # x: input sequence
    # y: output sequence
    # idx:
    # output: tuple of two torch tensor stacks size (K-1)x max_seq_len
    def __getitem__(self, idx):
        # Small test
        self.dialogue = self.all_dialogues[idx]
        self.phrase = self.load_dialogue(self.dialogue)
        return idx, self.phrase
