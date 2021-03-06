import torch
from bert_score import BERTScorer
import pickle
import os
from nltk.translate import bleu_score as bleu
import utils
import utils_generation as ugen
from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer
device = 'cpu'
cuda = False
if (torch.cuda.is_available()):
    device = torch.device('cuda')
    cuda = True
# Load the BERT tokenizer.
print('Loading BERT model...')

model_path = 'bert-base-uncased'
tokenizer_path = 'bert-base-uncased'
#model_path = '/Users/willemvandemierop/Google Drive/DL Prediction (706)/BERT_100/model_bert_lr0001_wd001_batch200_ep100_mPlenght40_tmp'
#tokenizer_path = '/Users/willemvandemierop/Google Drive/DL Prediction (706)/BERT_100/model_bert_lr0001_wd001_batch200_ep100_mPlenght40_tmp'
#model_path = '/Users/willemvandemierop/Documents/Master AI/Pycharm/Chatbot_bert/saved_directories/model_scibert_lr0001_wd001_batch200_ep1_final'
#tokenizer_path = '/Users/willemvandemierop/Documents/Master AI/Pycharm/Chatbot_bert/saved_directories/model_scibert_lr0001_wd001_batch200_ep1_final'
ugen.load_model_tokenizer(model_path = model_path, tokenizer_path = tokenizer_path)

"""We're finally going to use the generation function: 

1.   max_len(40): length of sequence to generate
2.   top_k(100): at each step, sample from the top_k most likely words
3.   temperature(1): link provided with explanation = smoothing parameter for next word distribution. Higher values means more like uniform, lower means more peaky
4.   seed_text(["CLS"]): prefix to generate for. we start with this as a test later text needs to be added for bot.
"""
#===================================== Hyperparameters for generation ==================================================
n_samples = 1
batch_size = 2
max_len = 20
top_k = 50
temperature = 1.5
generation_mode = "parallel-sequential"
leed_out_len = 5 # max_len
burnin = 250
sample = True
max_iter = 500
question = 2
ModelForQ_A_on = True
Metrics_calculation = True

#========================================== BERTScorer initialisation ==================================================
Q_metrics = [[],[],[]]
scorer = BERTScorer(model_type='bert-base-uncased')
q_refs = pickle.load(open('Metrics_files/Q_refs.pkl', 'rb'))
q3_refs = q_refs['q3_refs']
q2_refs = q_refs['q2_refs']
q1_refs = q_refs['q1_refs']
all_q_refs = [q1_refs,q2_refs,q3_refs]
all_q_cands = ['Who is she?', 'Are you okay?', 'Why?']


#==================================================== Word generation ==================================================
# Choose the prefix context
#seed_text = ugen.tokenizer.tokenize("who is she?".lower())
if Metrics_calculation:
    print('Metrics (BLEU, P, R, F1)')
    for i in range(len(all_q_cands)):
        seed_text = ugen.tokenizer.tokenize(all_q_cands[i].lower())
        refs = all_q_refs[i]
        bleu, P, R, F1 = utils.return_metrics(scorer=scorer, refs=refs, seed_text=seed_text,
                                                n_samples=n_samples, top_k=top_k,
                                                temperature=temperature, max_len=max_len, cuda=cuda, print_sent = True)
        Q_metrics[i].append([bleu, P, R, F1])
        print('Q'+str(i+1),' Metrics: ', Q_metrics[i][-1])


    metrics = {'q_metrics': Q_metrics} #{'q1_metrics': Q1_metrics 'q2_metrics': Q2_metrics, 'q3_metrics': Q3_metrics}
    torch.save(metrics, os.path.join(os.getcwd(),'metrics_with_modelforQA.pkl'))

#sci_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
##vocab_sci = sci_tokenizer.get_vocab()
#bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
#vocab_bert = bert_tokenizer.get_vocab()

import readability
# readability_score = readability.getmeasures(answers[0], lang = 'en')
# print("readability score:", readability_score['readability grades']['FleschReadingEase'])
# https://en.wikipedia.org/wiki/Flesch–Kincaid_readability_tests
# https://readable.com/blog/how-can-lix-and-rix-help-score-readability-for-non-english-content/
# the flesh readability score represents the school reading level. So a lower score means scibert?
for i in range(len(all_q_cands)):
    print("\nQuestion: ", all_q_cands[i])
    seed_text = ugen.tokenizer.tokenize(all_q_cands[i].lower())
    print("readability score of question",readability.getmeasures(seed_text, lang='en')['readability grades']['FleschReadingEase'])
    print("Lix score of question",
          readability.getmeasures(seed_text, lang='en')['readability grades']['LIX'])

difficult_question = "The inoculation of Wagner, was it gravitational indexing and macroscopic?"
print("\nDifficult Question: ", difficult_question)
test_ques = ugen.tokenizer.tokenize(difficult_question.lower())
print("readability score of test question",readability.getmeasures(test_ques, lang='en')['readability grades']['FleschReadingEase'])
print("Lix score of test question",
          readability.getmeasures(test_ques, lang='en')['readability grades']['LIX'])

if readability.getmeasures(test_ques, lang='en')['readability grades']['FleschReadingEase'] < 80:
    print('\nHold on let me get an expert to help you with your question!')

answers = []
#======================================= Bert for question and answering ===============================================
if ModelForQ_A_on:
    modelForQuestionAnswering = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    for i in range(len(all_q_cands)):
        done = False
        while not done:
            seed_text = ugen.tokenizer.tokenize(all_q_cands[i].lower())
            untokenized, batch = ugen.sequential_generation(seed_text=seed_text, batch_size=n_samples, max_len=max_len,
                                                            top_k=top_k, temperature=temperature, cuda=cuda,
                                                            leed_out_len=len(seed_text))

            question = all_q_cands[i].lower()
            text = ugen.tokenizer.decode(batch[0][len(seed_text) + 2:-1])
            encoding = ugen.tokenizer.encode_plus(question, text)
            input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
            start_scores, end_scores = modelForQuestionAnswering(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
            print("\nQuestion: ", question, "\nAnswer:", text)
            all_tokens = ugen.tokenizer.convert_ids_to_tokens(input_ids)
            print(answer)
            answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1])
            print(f'The answer according to modelForQuestionAnswering is: "{answer}"')
            if len(answer) > 0:
                answers.append(answer)
                print(question + ' ' + answer)
                done = True
            else:
                print('Empty answer, try again!')

print("\nThese are our answers", answers)

