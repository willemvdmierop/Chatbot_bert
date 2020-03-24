from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

from transformers import AutoModelWithLMHead, AutoTokenizer
'''
tokenizer = AutoTokenizer.from_pretrained("lysandre/arxiv-nlp")
model = AutoModelWithLMHead.from_pretrained("lysandre/arxiv-nlp")


sequence = f"Hugging Face is based in DUMBO, New York City, and is"

input = tokenizer.encode(sequence, return_tensors="pt")
generated = model.generate(input, max_length=50)

resulting_string = tokenizer.decode(generated.tolist()[0])
print(resulting_string)
'''

#tokenizer_scibert = AutoTokenizer.from_pretrained("./scibert_scivocab_uncased")
#model_scibert = AutoModelForQuestionAnswering.from_pretrained("./scibert_scivocab_uncased")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')

text = r"""
🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.
"""

questions = [
    "How many pretrained models are available in Transformers?",
    "What does Transformers provide?",
    "Transformers provides interoperability between which frameworks?",
]

for question in questions:
    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs)

    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    print(f"Question: {question}")
    print(f"Answer: {answer}\n")

