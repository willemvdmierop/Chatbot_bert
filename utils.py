import os, sys, re
from collections import OrderedDict
import time
import string
import torch
import numpy as np
import copy
import pandas as pd
import utils_generation as ugen
from nltk.translate import bleu_score as bleu
from bert_score import BERTScorer


def load_lines(file):
    lines = []
    with open(file, 'r', encoding="iso-8859-1") as f:
        for l in f:
            lines.append(l.strip().split(' +++$+++ '))  # takes these out  +++$+++
    return lines


def create_phrases_dict(l):
    phrases = {}  # for example {'L1045': 'they do not!', 'L1044': 'they do to!'} with L1045 an ID
    for idx, lines in enumerate(l):
        phrases[lines[0]] = lines[-1].lower()
    return phrases


loaded_lines = load_lines("movie_lines.txt")
all_lines = create_phrases_dict(loaded_lines)


# movie_conversations: u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197'] so we need the indexes
def load_dialogues(file):
    dialogues_all = []  # dialogues_all example: [['u0', 'u2', 'm0', "['L194', 'L195', 'L196', 'L197']"]]
    with open(file, 'r', encoding="iso-8859-1") as f:
        for l in f:
            dialogues_all.append(l.strip().split(' +++$+++ '))
    return dialogues_all


def question_answers_dataset():
    all_movie_lines = all_lines
    questions_dic = []
    answers_dic = []
    movie_dialogues = load_dialogues("movie_conversations.txt")
    for idx, dialogue in enumerate(movie_dialogues):
        phrases = dialogue[-1]
        phrases = re.split('\W+', phrases)[1:-1]
        # for i in range(0,len(phrases) -1,2):
        #    question = all_movie_lines[phrases[i]]
        #    answer = all_movie_lines[phrases[i+1]]
        #    question_answers[question] = answer
        for id, ph in enumerate(phrases[:-1]):
            question = all_movie_lines[phrases[id]]
            answer = all_movie_lines[phrases[id + 1]]
            if len(question.split()) <= 20 and len(answer.split()) <= 20:
                questions_dic.append(question)
                answers_dic.append(answer)

    return questions_dic, answers_dic


def create_dialogue_dataset():
    # first sample from each list
    # this file contains actual lines
    movie_lines = loaded_lines
    all_movie_lines = all_lines
    # this file contains indices of the phrases
    movie_dialogues = load_dialogues("movie_conversations.txt")
    # this list will contain all oredered dicts
    full_list = []
    for idx, dialogue in enumerate(movie_dialogues):
        phrases = dialogue[-1]
        phrases = re.split('\W+', phrases)[1:-1]
        # this_dialogue: ordered dict [(ph1,ph2), (ph2, ph3)...]
        # one dialgue: one data point
        this_dialogue = OrderedDict()
        # ph are like 'L19043' taken from the lines data
        for id, ph in enumerate(phrases[:-1]):
            this_dialogue[all_movie_lines[ph]] = all_movie_lines[phrases[id + 1]]  ##old code
            # this_dialogue[ph, phrases[id+1]] = all_movie_lines[phrases[id+2]]
        # if len(this_dialogue) != 0:
        full_list.append(this_dialogue)

    return full_list


def create_vocab():
    # movie_lines = loaded_lines
    all_movie_lines = all_lines
    vocab = []
    for k in all_movie_lines.keys():
        phrase = all_movie_lines[k]
        # get the list of trimmed tokens - get rid of non-letter chars
        phrase_trimmed = re.sub(r'[^a-zA-Z\s]+|(.)\1{3,}', ' ', phrase).lower().split()
        # phrase_trimmed = [word.strip(string.punctuation) for word in phrase.lower().split()]
        # print(phrase_trimmed)
        for w in phrase_trimmed:
            if not w in vocab:
                vocab.append(w)

    return sorted(vocab)


def print_dialogue_data_metrics(question_data, answer_data):
    max_length_questions = 0
    mean_length_q = 0
    for i in range(len(question_data)):
        max_length_questions = max(max_length_questions, len(question_data[i]))
        mean_length_q += len(question_data[i])
    mean_length_q /= len(question_data)

    mean_length_a = 0
    max_length_answers = 0
    for i in range(len(answer_data)):
        max_length_answers = max(max_length_answers, len(answer_data[i]))
        mean_length_a += len(answer_data[i])
    mean_length_a /= len(answer_data)

    print("\n" + 96 * '#')
    print('## Question data[0] : {} , \n## Answer data[0] :  {}'.format(question_data[0], answer_data[0]))
    print('The max lenght of the Questions is: {}, the max length of the answers is: {}'.format(max_length_questions,
                                                                                                max_length_answers))
    print(
        'The mean lenght of the Questions is: {0:.2f}, the mean length of the answers is: {1:.2f}'.format(mean_length_q,
                                                                                                          mean_length_a))
    print(96 * '#')


def save_data_csv(question_data, answer_data):
    array_q = []
    for _, value in enumerate(question_data):
        array_q.append([value])

    df = pd.DataFrame(array_q)
    df.to_csv("question_data.csv")

    array_a = []
    for _, value in enumerate(answer_data):
        array_a.append([value])

    df = pd.DataFrame(array_a)
    df.to_csv("answer_data.csv")


def return_metrics(scorer, refs, seed_text, n_samples, max_len =20, top_k=50, temperature=1.5, cuda=False, print_sent = False):
    untokenized, batch = ugen.sequential_generation(seed_text=seed_text, batch_size=n_samples, max_len=max_len,
                                                    top_k=top_k, temperature=temperature, cuda=cuda,
                                                    leed_out_len=len(seed_text))
    bleu_batch = []
    P_list = []
    R_list = []
    F1_list = []
    for b in batch:
        if print_sent:
            print(ugen.tokenizer.decode(b))
        bleu_batch.append(
            bleu.sentence_bleu(hypothesis=ugen.tokenizer.decode(b[len(seed_text) + 2:-1]), references=refs))
        P, R, F1 = scorer.score(cands=[ugen.tokenizer.decode(b[len(seed_text) + 2:-1])], refs=[refs])
        P_list.append(P.item())
        R_list.append(R.item())
        F1_list.append(F1.item())

    return np.mean(bleu_batch), np.mean(P_list), np.mean(R_list), np.mean(F1_list)


def make_input(question, answer, tokenizer):
    kwargs = {'text': question,
              'text_pair': answer,
              'max_length': 40,
              'pad_to_max_length': True,
              'add_special_tokens': True,
              'return_tensors': 'pt',
              'return_token_type_ids': True,
              'return_attention_mask': True,
              'return_special_tokens_mask': True}
    input_phrase = tokenizer.encode_plus(**kwargs)
    masked_lm_labels_temp = -100 * (
            torch.ones(len(input_phrase['attention_mask'])) - input_phrase['token_type_ids'] == 1)
    masked_lm_labels = (input_phrase['token_type_ids'] * input_phrase['input_ids']) + masked_lm_labels_temp
    new_input_eval = copy.deepcopy(input_phrase['input_ids'])
    new_input_eval[
        (input_phrase['attention_mask'] - input_phrase['token_type_ids']) == 0] = tokenizer.convert_tokens_to_ids(
        '[MASK]')
    input_phrase['masked_lm_labels'] = masked_lm_labels
    input_phrase['new_input_eval'] = new_input_eval
    return input_phrase
