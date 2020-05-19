This code was initial developed for the purposes of completing a MSc Coursework assignment

# BERT Chatbot 
The BERT Chatbot allows the user to interact with a basic conversational model, as well as understanding when to answer with a more intelligent (trained on scientific papers) model.

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
python3.6 FineTune_questions_answers.py
```
Training parameters can be changed by editing the corresponding section in beginning of the code

## Other files
utils.py :  utility methods used for training
utils_generation.py : utilty methods used for text generation
dataset_Q_A.py : classes need for dataset creation
bert_word_generation.py : file used to generate text
movie_lines.txt : text file containing movie line content with line IDs
movie_conversations : text file containing the sets of line IDs that consist of each dialogue
why_refs.txt : text file containing all the answers provided to the question "Why?" from the cornell dataset
who_is_she_refs.txt : text file containing all the answers provided to the question "Who is she?" from the cornell dataset
are_you_okay_refs.txt : text file containing all the answers provided to the question "Are you akoy?" from the cornell dataset
