# -*- coding: utf-8 -*-
"""GPT2 test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XRvbR9hkd0kVq-lLEn5hxMDnnMFc2Eak
"""

from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
import warnings

warnings.filterwarnings("ignore")

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelWithLMHead.from_pretrained('gpt2')

input_ids = torch.tensor(tokenizer.encode("what is the meaning of life ?", return_tensors='pt'))
mask_id = tokenizer.convert_tokens_to_ids('[MASK]')
print(input_ids)
print(mask_id)
outputs = model.generate(input_ids=input_ids, num_return_sequences=1,
                         temperature=2)
for i in range(1):
    print(outputs[i])
    print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))
