import os, sys, re
import tkinter as tk
import codecs
from collections import OrderedDict


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


loaded_lines = load_lines(os.path.join("cornell movie-dialogs corpus", "movie_lines.txt"))
all_lines = create_phrases_dict(loaded_lines)


# movie_conversations: u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197'] so we need the indexes
def load_dialogues(file):
    dialogues_all = []  # dialogues_all example: [['u0', 'u2', 'm0', "['L194', 'L195', 'L196', 'L197']"]]
    with open(file, 'r', encoding="iso-8859-1") as f:
        for l in f:
            dialogues_all.append(l.strip().split(' +++$+++ '))
    return dialogues_all


def create_dialogue_dataset():
    # first sample from each list
    # this file contains actual lines
    movie_lines = loaded_lines
    all_movie_lines = all_lines
    # this file contains indices of the phrases
    movie_dialogues = load_dialogues(os.path.join("cornell movie-dialogs corpus", "movie_conversations.txt"))
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
            this_dialogue[all_movie_lines[ph]] = all_movie_lines[phrases[id + 1]] ##old code
            #this_dialogue[ph, phrases[id+1]] = all_movie_lines[phrases[id+2]]
        #if len(this_dialogue) != 0:    
        full_list.append(this_dialogue)   

    return full_list


def create_vocab():
    movie_lines = loaded_lines
    all_movie_lines = all_lines
    vocab = []
    for k in all_movie_lines.keys():
        phrase = all_movie_lines[k]
        # get the list of trimmed tokens - get rid of non-letter chars
        phrase_trimmed = re.sub('\W+', ' ', phrase).split()
        # print(phrase_trimmed)
        for w in phrase_trimmed:
            if not w in vocab:
                vocab.append(w)

    return sorted(vocab)

