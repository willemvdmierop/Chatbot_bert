import os, sys, re
from collections import OrderedDict
import time


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
    questions_dic = {}
    answers_dic = {}
    movie_dialogues = load_dialogues("movie_conversations.txt")
    count = 0
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
            questions_dic[count] = question
            answers_dic[count] = answer
            count += 1

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

import string
def create_vocab():
    #movie_lines = loaded_lines
    all_movie_lines = all_lines
    vocab = []
    for k in all_movie_lines.keys():
        phrase = all_movie_lines[k]
        # get the list of trimmed tokens - get rid of non-letter chars
        phrase_trimmed = re.sub(r'[^a-zA-Z\s]+|(.)\1{3,}', ' ', phrase).lower().split()
        #phrase_trimmed = [word.strip(string.punctuation) for word in phrase.lower().split()]
        # print(phrase_trimmed)
        for w in phrase_trimmed:
            if not w in vocab:
                vocab.append(w)

    return sorted(vocab)

def print_dialogue_data_metrics(self, question_data, answer_data):
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
    print('The max lenght of the Questions is: {}, the max length of the answers is: {}'.format(max_length_questions, max_length_answers))
    print('The mean lenght of the Questions is: {0:.2f}, the mean length of the answers is: {1:.2f}'.format(mean_length_q, mean_length_a))
    print(96 * '#')
