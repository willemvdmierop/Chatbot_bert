import torch
from bert_score import BERTScorer
import pickle
import os
from nltk.translate import bleu_score as bleu
import utils
import utils_generation as ugen
from transformers import BertForQuestionAnswering

device = 'cpu'
cuda = False
if (torch.cuda.is_available()):
    device = torch.device('cuda')
    cuda = True
# Load the BERT tokenizer.
print('Loading BERT model...')

model_path = 'bert-base-uncased'
tokenizer_path = 'bert-base-uncased'
#model_path = '/Users/willemvandemierop/Google Drive/DL Prediction (706)/BERT_60_Baseline/model_bert_lr0001_wd001_batch200_ep60_mPlenght40_tmp'
#tokenizer_path = '/Users/willemvandemierop/Google Drive/DL Prediction (706)/BERT_60_Baseline/model_bert_lr0001_wd001_batch200_ep60_mPlenght40_tmp'
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
Metrics_calculation = False

#========================================== BERTScorer initialisation ==================================================
Q_metrics = [[],[],[]]
scorer = BERTScorer(model_type='bert-base-uncased')
q_refs = pickle.load(open('Q_refs.pkl', 'rb'))
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
    torch.save(metrics, os.path.join(os.getcwd(),'metrics_sci_benchmark.pkl'))

#======================================= Bert for question and answering ===============================================
if ModelForQ_A_on:
    modelForQuestionAnswering = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    for i in range(len(all_q_cands)):
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
        answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1])
        print(f'The answer according to modelForQuestionAnswering is: {answer}')





'''
scorer = BERTScorer(model_type='bert-base-uncased')
q_refs = pickle.load(open('Q_refs.pkl', 'rb'))
q3_refs = q_refs['q3_refs']
q2_refs = q_refs['q2_refs']
q1_refs = q_refs['q1_refs']
if question == 1:
    seed_text = ugen.tokenizer.tokenize("who is she?".lower())
    refs = q1_refs
elif question == 2:
    seed_text = ugen.tokenizer.tokenize("are you okay?".lower())
    refs = q2_refs
elif question == 3:
    seed_text = ugen.tokenizer.tokenize("why?".lower())
    refs = q3_refs
'''
'''
len_seed = len(seed_text)
bert_sents = ugen.generate(n_samples, seed_text=seed_text, batch_size=batch_size, max_len=max_len,
                      generation_mode=generation_mode,
                      sample=sample, top_k=top_k, temperature=temperature, burnin=burnin, max_iter=max_iter,
                      cuda=cuda)

for sent in bert_sents:
    ugen.printer(sent, should_detokenize=True)

max_n = 4

# this produces BLEU of 0 (no 4-grams)
#for i in range(len(bert_sents)):
#  bert_sents[i] = bert_sents[i][len_seed:]

#print(bert_sents)
print("BERT %s self-BLEU: %.2f" % (model_path, 100 * ugen.self_bleu(bert_sents)))
'''
'''
untokenized, batch = ugen.sequential_generation(seed_text=seed_text, batch_size=n_samples,max_len=max_len, top_k=top_k,temperature=temperature, cuda=cuda, leed_out_len=len(seed_text))
gen_batch = []
print(untokenized)
for b in batch:
    gen_batch.append(ugen.tokenizer.decode(b[len(seed_text)+2:-1]))
    print(ugen.tokenizer.decode(b))
print('BLEU Score : ', bleu.sentence_bleu(hypothesis=gen_batch[0],references=refs))
print('BERTScore: ', scorer.score(cands=[gen_batch[0]],refs=[refs]))

max_n = 4
pct_uniques = ugen.self_unique_ngrams(untokenized, max_n)
for i in range(1, max_n + 1):
    print("BERT %s unique %d-grams relative to self: %.2f" % (model_path, i, 100 * pct_uniques[i]))
'''