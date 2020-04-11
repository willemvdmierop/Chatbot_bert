import torch
import torch.utils
from torch.utils import data
from transformers import BertTokenizer
import copy
import utils


class MoviePhrasesData(data.Dataset):

    # voc: vocabulary, word:idx
    # all dialogues:
    def __init__(self, all_dialogues=None, questions_data=None, answers_data=None, max_seq_len=40, tokenizer = None):
        super(MoviePhrasesData, self).__init__()
        self.max_seq_len = max_seq_len
        # self.voc = voc
        self.all_dialogues = all_dialogues
        self.questions_data = questions_data
        self.answers_data = answers_data
        self.tokenizer = tokenizer
        
    def load_ques_answ(self, idx):
        question = self.questions_data[idx]
        answer = self.answers_data[idx]
        kwargs = {'text': question,
                  'text_pair': answer,
                  'max_length': 40,
                  'pad_to_max_length': True,
                  'add_special_tokens': True,
                  'return_tensors': 'pt',
                  'return_token_type_ids': True,
                  'return_attention_mask': True,
                  'return_special_tokens_mask': True}
        input_phrase = self.tokenizer.encode_plus(**kwargs)
        masked_lm_labels_temp = -100 * (
                    torch.ones(len(input_phrase['attention_mask'])) - input_phrase['token_type_ids'] == 1)
        masked_lm_labels = (input_phrase['token_type_ids']*input_phrase['input_ids']) + masked_lm_labels_temp
        input_phrase['attention_mask'] = copy.deepcopy(masked_lm_labels) / -100
        new_input_eval = copy.deepcopy(input_phrase['input_ids'])
        new_input_eval[
            (input_phrase['attention_mask'] - input_phrase['token_type_ids']) == 0] = self.tokenizer.convert_tokens_to_ids(
            '[MASK]')
        input_phrase['masked_lm_labels'] = masked_lm_labels
        input_phrase['new_input_eval'] = new_input_eval
        return input_phrase

    # number of dialogues, 83097
    def __len__(self):
        if self.all_dialogues is not None:
            return len(self.all_dialogues)
        else:
            return len(self.questions_data)

    def __getitem__(self, idx):
        phrase = self.load_ques_answ(idx)
        return phrase
