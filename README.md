# BERT Chatbot 
The report of this repo can be found here [[report](./DL_Prediction_BERT.pdf)].
(old report update to latest)

This repo introduces a BERT Chatbot that is fine-tuned with Cornell Movie-Quotes Corpus to generate answers with a basic conversational model, as well as understanding when to answer with a more intelligent (trained on scientific papers) model.
## Model
We fine-tuned the BertForMaskedLM model, this model has a language mod- eling head on top. The input phrase contains the question and the answer tokenized in one concatenated tensor. We create a custom method to create our masked language model labels because we only want to mask our answer. This means that our masked lm labels tensor has a -100 for the question and the padding and has the normal token ids for the answer from until a [SEP].

## Word Generation
BERT is bidirectional that means that when we want to generate text we will need the context to the right of the question. This isn’t available when we use chatbot so we implement a similar method as explained in ”BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model” (Wang and Cho 2019). With this method we add a [CLS] at the front of the input text, which is necessary as an input to BERT. Then we add a [SEP] showing the end of our question, after that we create mask tokens until the predefined maximum length of the phrase, a maximum length of 40 is chosen for our implementation. Finally at the end of the input phrase a [SEP] token is added to indicate the end of the question.
The BERT Chatbot allows the user to interact with a basic conversational model, as well as understanding when to answer with a more intelligent (trained on scientific papers) model.

## Dataset
The dataset used for fine-tuning is the Cornell Movie-Quotes Corpus that contains conversations form raw movie scripts. The movie lines.txt dataset contains the line-id, the charachter-id, the movie-id, the character-name and the text. The movie conversations.txt then contains the different line-ids of text from a particular converstation. This is used to generate our questions and answers dataset. The input to the BERT model is a input phrase with the question and answers concatenated, after that the next input is the previous answer and the next question.
## Required packages
- Python 3.6
- Pytorch 
- bert-score 0.3.1
- numpy
- nltk
- panda
- pickle
- Tensorflow

## Usage
To run the interface:
```
python3.6 Chatbot_GUI.py
```

To trian a BERT model on the Cornell Movie Dialogue dataset:
```
python3.6 FineTune_SCI_and_BERT.py
```
Training parameters can be changed by editing the corresponding section in beginning of the code

## Other files
utils.py :  utility methods used for training
utils_generation.py : utilty methods used for text generation
Dataset_Q_A.py : classes need for dataset creation
bert_word_generation.py : file used to generate text
movie_lines.txt : text file containing movie line content with line IDs
movie_conversations : text file containing the sets of line IDs that consist of each dialogue
why_refs.txt : text file containing all the answers provided to the question "Why?" from the cornell dataset
who_is_she_refs.txt : text file containing all the answers provided to the question "Who is she?" from the cornell dataset
are_you_okay_refs.txt : text file containing all the answers provided to the question "Are you akoy?" from the cornell dataset
