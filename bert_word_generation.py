from transformers import AutoTokenizer, AutoModel, BertForMaskedLM, AutoModelWithLMHead, BertTokenizer
import torch
from torch.utils.tensorboard import SummaryWriter
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
#model_path = 'allenai/scibert_scivocab_uncased'
#tokenizer_path = 'allenai/scibert_scivocab_uncased'
ugen.load_model_tokenizer(model_path = model_path, tokenizer_path = tokenizer_path)



"""We're finally going to use the generation function: 

1.   max_len(40): length of sequence to generate
2.   top_k(100): at each step, sample from the top_k most likely words
3.   temperature(1): link provided with explanation = smoothing parameter for next word distribution. Higher values means more like uniform, lower means more peaky
4.   seed_text(["CLS"]): prefix to generate for. we start with this as a test later text needs to be added for bot.
"""

n_samples = 4
batch_size = 2
max_len = 15
top_k = 100
temperature = 1.0
generation_mode = "parallel-sequential"
leed_out_len = 5 # max_len
burnin = 250
sample = True
max_iter = 500

# Choose the prefix context
seed_text = ugen.tokenizer.tokenize("Wagner was a German composer, his inoculation was ambiguous.".lower())
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

pct_uniques = ugen.self_unique_ngrams(bert_sents, max_n)
for i in range(1, max_n + 1):
    print("BERT %s unique %d-grams relative to self: %.2f" % (model_path, i, 100 * pct_uniques[i]))
