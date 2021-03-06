import os
import time
import torch
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModel, BertForMaskedLM, AutoModelWithLMHead, BertTokenizer
import math
from nltk.util import ngrams
from nltk.translate import bleu_score as bleu

device = 'cpu'
cuda = False
if (torch.cuda.is_available()):
    device = torch.device('cuda')
    cuda = True

PAD, MASK, CLS, SEP = '[PAD]', '[MASK]', '[CLS]', '[SEP]'

### Loads the tokenizer and it's parameters into the scope of the generation utilities
def load_model_tokenizer(model_path = 'bert-base-uncased', tokenizer_path = 'bert-base-uncased', is_path=True):
    global model, tokenizer, mask_id, sep_id, cls_id

    ### The tokenizer can be passed to this method either by file path or as an object
    if is_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
        model = BertForMaskedLM.from_pretrained(model_path)
    else:
        tokenizer = tokenizer_path
        model = model_path
    model.eval()
    model = model.to(device)
    params = list(model.named_parameters())
    mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
    sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
    cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]

### Convert batch of text sentences to token ids
def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]

### Convert batch of token ids to tokens
def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]

### Convert tokens to readable text
def detokenize(sent):
    new_sent = []
    for i, token in enumerate(sent):
        if token.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + token[2:]
        else:
            new_sent.append(token)
    return new_sent

### Generate single word at position gen_idx+1
def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
    """ Generate a word from from out[gen_idx]
    
    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k

    - explanation of temperature:
    https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277

  """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k,
                                        dim=1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx

### Format given seed text with special characters and masked padding up to max length
def get_init_text(seed_text, max_len, batch_size=1, rand_init=False):
    batch = [[CLS] + seed_text + [SEP] +  [MASK] * max_len + [SEP] for _ in range(batch_size)]
    return tokenize_batch(batch)

### Output text in a readable format
def printer(sent, should_detokenize=True):
    if should_detokenize:
        sent = detokenize(sent)
    print(" ".join(sent))

### Generate text in a sequential (right to left) context one word at a time
def sequential_generation(seed_text, batch_size=10, max_len=15, leed_out_len=15,
                          top_k=0, temperature=None, sample=False, cuda=True):
    seed_len = len(seed_text)+2
    batch = get_init_text(seed_text, max_len, batch_size)

    for i in range(max_len):
        input_text = [sent[:seed_len + i + leed_out_len] + [sep_id] for sent in batch]
        input_text = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        output_model = model(input_text)
        output_model = output_model[0].detach()
        idxs = generate_step(output_model, gen_idx=seed_len + i, top_k=top_k, temperature=temperature, sample=sample)
        for j in range(batch_size):
            batch[j][seed_len + i] = idxs[j]

    return untokenize_batch(batch), batch

### Generate text for a random position of the masked sequence until all words are generated
def parallel_sequential_generation(seed_text, batch_size=10, max_len=15, top_k=0, temperature=None, max_iter=300,
                                   burnin=200,
                                   cuda=False, print_every=10, verbose=True):
    """ Generate for one random position at a timestep
    args:
        - burnin: during burn-in period, sample from full distribution; afterwards take argmax
    """
    seed_len = len(seed_text)+2
    batch = get_init_text(seed_text, max_len, batch_size)

    for ii in range(max_iter):
        kk = np.random.randint(0, max_len)
        for jj in range(batch_size):
            batch[jj][seed_len + kk] = mask_id
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        out = out[0].detach()
        topk = top_k if (ii >= burnin) else 0
        idxs = generate_step(out, gen_idx=seed_len + kk, top_k=topk, temperature=temperature, sample=(ii < burnin))
        for jj in range(batch_size):
            batch[jj][seed_len + kk] = idxs[jj]

        if verbose and np.mod(ii + 1, print_every) == 0:
            for_print = tokenizer.convert_ids_to_tokens(batch[0])
            for_print = for_print[:seed_len + kk + 1] + ['(*)'] + for_print[seed_len + kk + 1:]
            print("iter", ii + 1, " ".join(for_print))

    return untokenize_batch(batch), batch

### Wrapper method to generate text given a phrase (seed_text), and generation parameters
def generate(n_samples, seed_text="[CLS]", batch_size=10, max_len=25,
             generation_mode="parallel-sequential",
             sample=True, top_k=100, temperature=1.0, burnin=200, max_iter=500,
             cuda=False, print_every=1):
    sentences = []
    n_batches = math.ceil(n_samples / batch_size)
    start_time = time.time()
    for batch_n in range(n_batches):
        if generation_mode == "parallel-sequential":
            batch = parallel_sequential_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k,
                                                   temperature=temperature, burnin=burnin, max_iter=max_iter,
                                                   cuda=cuda, verbose=False)
        elif generation_mode == "sequential":
            batch = sequential_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k,
                                          temperature=temperature, sample=sample,
                                          cuda=cuda)
        
        if (batch_n + 1) % print_every == 0:
            print("Finished batch %d in %.3f minutes" % (batch_n + 1, (time.time() - start_time)/60))
            start_time = time.time()
        sentences += batch
    return sentences

### Calculate BLEU score of a batch of sentences using the batch as references
def self_bleu(sents):
    return bleu.corpus_bleu([[s for (j, s) in enumerate(sents) if j != i] for i in range(len(sents))], sents)

### Caluclate the number of n-grams (n: 1-4) in the batch of sentences
def get_ngram_counts(sents, max_n=4):
    size2count = {}
    for i in range(1, max_n + 1):
        size2count[i] = Counter([n for sent in sents for n in ngrams(sent, i)])
    return size2count

### Calculate unique n-grams (n: 1-4) in batch of sentences
def self_unique_ngrams(preds, max_n=4):
    # get # of pred ngrams with count 1
    pct_unique = {}
    pred_ngrams = get_ngram_counts(preds, max_n)
    for i in range(1, max_n + 1):
        n_unique = len([k for k, v in pred_ngrams[i].items() if v == 1])
        total = sum(pred_ngrams[i].values())
        pct_unique[i] = n_unique / total
    return pct_unique
