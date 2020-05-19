import utils_generation as ugen
import utils
from transformers import BertTokenizer
import pandas as pd
import torch
from bert_score import BERTScorer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel, BertForMaskedLM
import pickle
import numpy as np


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model_Q_A = BertForMaskedLM.from_pretrained('bert-base-uncased')

n_samples = 10
max_len = 20
#question = 1

scorer = BERTScorer(model_type='bert-base-uncased')
q_refs = pickle.load(open('Q_refs.pkl', 'rb'))
q3_refs = q_refs['q3_refs']
q2_refs = q_refs['q2_refs']
q1_refs = q_refs['q1_refs']
all_q_refs = [q1_refs,q2_refs,q3_refs]
all_q_cands = ['Who is she?', 'Are you okay?', 'Why?']

topK = [20,40,60,80,100,120,140]
temp = [1, 1.5, 2, 2.5, 3, 3.5, 4]

ugen.load_model_tokenizer(model_path=model_Q_A, tokenizer_path=tokenizer, is_path=False)
print('Metrics (temp, topK, BLEU, P, R, F1)')

cuda  = False

metrics = [[],[],[]]
for temperature in temp:
    for top_k in topK:
        for i in range(len(all_q_cands)):
            seed_text = tokenizer.tokenize(all_q_cands[i].lower())
            refs = all_q_refs[i]
            bleu, P, R, F1 = utils.return_metrics(scorer=scorer, refs=refs, seed_text=seed_text,
                                                  n_samples=n_samples, top_k=top_k,
                                                  temperature=temperature, max_len=max_len, cuda=cuda)
            metrics[i].append([temperature, top_k, bleu, P, R, F1])


final_metrics = {'gen_metrics': metrics}
torch.save(final_metrics, "word_gen_metrics.pkl")

df = pd.DataFrame(metrics)
df.to_csv("metrics_final_word_gen.csv")

print("finished saving the metrics")