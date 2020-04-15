from transformers import AutoTokenizer, AutoModel, BertForMaskedLM, AutoModelWithLMHead, BertTokenizer
import torch
from torch.utils.tensorboard import SummaryWriter
from bert_score import BERTScorer
import pickle
from nltk.translate import bleu_score as bleu

import utils_generation as ugen

device = 'cpu'
cuda = False
if (torch.cuda.is_available()):
    device = torch.device('cuda')
    cuda = True

# %matplotlib inline
# %load_ext tensorboard
# %tensorboard --logdir=runs

# Load the BERT tokenizer.
print('Loading BERT model...')
model_path = 'bert-base-uncased'
tokenizer_path = 'bert-base-uncased'
#model_path = '/Users/willemvandemierop/Documents/Master AI/Pycharm/Chatbot_bert/saved_directories/model_scibert_lr0001_wd001_batch200_ep1_final'
#tokenizer_path = '/Users/willemvandemierop/Documents/Master AI/Pycharm/Chatbot_bert/saved_directories/model_scibert_lr0001_wd001_batch200_ep1_final'
ugen.load_model_tokenizer(model_path = model_path, tokenizer_path = tokenizer_path)



"""We're finally going to use the generation function: 

1.   max_len(40): length of sequence to generate
2.   top_k(100): at each step, sample from the top_k most likely words
3.   temperature(1): link provided with explanation = smoothing parameter for next word distribution. Higher values means more like uniform, lower means more peaky
4.   seed_text(["CLS"]): prefix to generate for. we start with this as a test later text needs to be added for bot.
"""

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

# Choose the prefix context
#seed_text = ugen.tokenizer.tokenize("who is she?".lower())

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
untokenized, batch = ugen.sequential_generation(seed_text=seed_text, batch_size=n_samples,max_len=max_len, top_k=top_k,temperature=temperature, cuda=cuda, leed_out_len=len(seed_text))
gen_batch = []

for b in batch:
    gen_batch.append(ugen.tokenizer.decode(b[len(seed_text)+2:-1]))
    print(ugen.tokenizer.decode(b))
print('BLEU Score : ', bleu.sentence_bleu(hypothesis=gen_batch[0],references=refs))
print('BERTScore: ', scorer.score(cands=[gen_batch[0]],refs=[refs]))

max_n = 4
pct_uniques = ugen.self_unique_ngrams(untokenized, max_n)
for i in range(1, max_n + 1):
    print("BERT %s unique %d-grams relative to self: %.2f" % (model_path, i, 100 * pct_uniques[i]))
