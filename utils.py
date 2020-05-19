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

### Parse movie lines from txt file
def load_lines(file):
    lines = []
    with open(file, 'r', encoding="iso-8859-1") as f:
        for l in f:
            lines.append(l.strip().split(' +++$+++ '))  # takes these out  +++$+++
    return lines

### Create dictionary of line content in format : {Line_id: text}
def create_phrases_dict(l):
    phrases = {}  # for example {'L1045': 'they do not!', 'L1044': 'they do to!'}
    for idx, lines in enumerate(l):
        phrases[lines[0]] = lines[-1].lower()
    return phrases

### Load lines from movie dialogues
loaded_lines = load_lines("cornell_movie_dialogs_corpus/movie_lines.txt")
all_lines = create_phrases_dict(loaded_lines)


# Load sets of movie dialogue line IDs: e.g. ['L194', 'L195', 'L196', 'L197']
def load_dialogues(file):
    dialogues_all = []
    with open(file, 'r', encoding="iso-8859-1") as f:
        for l in f:
            dialogues_all.append(l.strip().split(' +++$+++ '))
    return dialogues_all

### Dataset of phrase pairs (question, answer)
def question_answers_dataset():
    all_movie_lines = all_lines
    questions_dic = []
    answers_dic = []
    movie_dialogues = load_dialogues("cornell_movie_dialogs_corpus/movie_conversations.txt")
    ### Iterate through all dialogues
    for idx, dialogue in enumerate(movie_dialogues):
        phrases = dialogue[-1]
        phrases = re.split('\W+', phrases)[1:-1]
        ### Iterate through all phrases in dialogue
        for id, ph in enumerate(phrases[:-1]):
            question = all_movie_lines[phrases[id]]
            answer = all_movie_lines[phrases[id + 1]]
            ### If either phrase is longer than 20 words, ignore pair
            if len(question.split()) <= 20 and len(answer.split()) <= 20:
                questions_dic.append(question)
                answers_dic.append(answer)

    return questions_dic, answers_dic

### Analyze dataset and print out metrics
def print_dialogue_data_metrics(question_data, answer_data):
    max_length_questions = 0
    ### Calculate maximum and mean length of phrases (questions)
    mean_length_q = 0
    for i in range(len(question_data)):
        max_length_questions = max(max_length_questions, len(question_data[i]))
        mean_length_q += len(question_data[i])
    mean_length_q /= len(question_data)
    ### Calculate maximum and mean length of responses (answers)
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

### Export dataset to csv format
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

### Calculate evaluation metrics given a question (seed_text) and known references (refs)
def return_metrics(scorer, refs, seed_text, n_samples, max_len =20, top_k=50, temperature=1.5, cuda=False, print_sent = False):
    bleu_batch = []
    P_list = []
    R_list = []
    F1_list = []

    ### Generate a batch of generated text based on given question (seed_text)
    untokenized, batch = ugen.sequential_generation(seed_text=seed_text, batch_size=n_samples, max_len=max_len,
                                                    top_k=top_k, temperature=temperature, cuda=cuda,
                                                    leed_out_len=len(seed_text))

    ### Iterate through batch of generated sequences and evaluate                                                
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

### Format input tensor ready to be inputted into model from two text phrases (question and answer)
def make_input(question, answer, tokenizer):
    ### Convert text to tokens ids
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

    ### Create masked lm label vector and add to input tensor
    masked_lm_labels_temp = -100 * (
            torch.ones(len(input_phrase['attention_mask'])) - input_phrase['token_type_ids'] == 1)
    masked_lm_labels = (input_phrase['token_type_ids'] * input_phrase['input_ids']) + masked_lm_labels_temp
    input_phrase['masked_lm_labels'] = masked_lm_labels
    return input_phrase
