from tkinter import *
from PIL import Image, ImageTk
import torch
from bert_score import BERTScorer
import pickle
import os
import utils
import utils_generation as ugen
import readability
from transformers import BertForQuestionAnswering
root = Tk()
root.title("Chatbot BERT")
HEIGHT = 700
WIDTH = 800
canvas = Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()
# =================================================== Model Loading ====================================================

model_path_BERT = 'bert-base-uncased'
tokenizer_path_BERT = 'bert-base-uncased'

model_path_SCIBERT = 'allenai/scibert_scivocab_uncased'
tokenizer_path_SCIBERT = 'allenai/scibert_scivocab_uncased'

ugen.load_model_tokenizer(model_path=model_path_BERT, tokenizer_path=tokenizer_path_BERT)
modelForQuestionAnswering = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
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
ModelForQ_A_on = True
Metrics_calculation = False
cuda = False

sci_global = False
def sci_on():
    global sci_global
    sci_global = True
    label = Label(root, text="SciBERT = ON", font=("Arial Bold", 20), bg='red')
    label.place(x=320, y=280)
# ============================================== Answer generation =====================================================
# this input needs to be passed to the chatbot
def Enter():
    label = None
    if not sci_global:
        sci_answer = False
    else:
        sci_answer = True
    input_text = entry.get()
    # ===================================== BERT or SCIBERT ============================================
    seed_text = ugen.tokenizer.tokenize(input_text.lower())
    print("\nreadability score of question",
          readability.getmeasures(seed_text, lang='en')['readability grades']['FleschReadingEase'])
    print("\nLix score of question",
          readability.getmeasures(seed_text, lang='en')['readability grades']['LIX'])

    if readability.getmeasures(seed_text, lang='en')['readability grades']['FleschReadingEase'] < 80:
        print('\nHold on let me get an expert to help you with your question!')
        ugen.load_model_tokenizer(model_path=model_path_SCIBERT, tokenizer_path=tokenizer_path_SCIBERT)
        sci_answer = True

    # ===================================== Word generation ============================================
    done = False
    while not done:
        untokenized, batch = ugen.sequential_generation(seed_text=seed_text, batch_size=n_samples, max_len=max_len,
                                                        top_k=top_k, temperature=temperature, cuda=cuda,
                                                        leed_out_len=len(seed_text))
        text = ugen.tokenizer.decode(batch[0][len(seed_text) + 2:-1])
        encoding = ugen.tokenizer.encode_plus(input_text, text)
        input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
        start_scores, end_scores = modelForQuestionAnswering(torch.tensor([input_ids]),
                                                             token_type_ids=torch.tensor([token_type_ids]))
        print("\nQuestion: ", input_text, "\nAnswer:", text)
        all_tokens = ugen.tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1])
        print(f'The answer according to modelForQuestionAnswering is: "{answer}"')
        if len(answer) > 0:
            print(input_text + ' ' + answer)
            done = True
        else:
            print('Empty answer, try again!')

    if sci_answer:
        label = Label(root, text="SciBERT = ON", font=("Arial Bold", 20), bg='red')
        label.place(x=320, y=280)
        label = Label(root, text="This answer was generated by an expert", font=("Arial Bold", 15), bg='yellow')
        label.place(x=250, y=250)
    if label is not None:
        label.destroy()
    answer_str = StringVar()
    answer_str.set(answer)
    label = Label(output_frame, textvariable=answer_str, font=40, bg="#00d2ff")
    label.place(relx=0.2, rely=0.35)
    answer = str()

#================================================== Tkinter ============================================================
img = ImageTk.PhotoImage(Image.open("/Users/willemvandemierop/Documents/Master AI/Pycharm/Chatbot_bert/images/City_logo.jpg"))
root.iconphoto(False, img)
background_label = Label(root, image=img)
background_label.place(x=0, y=0, relheight=1, relwidth=1)

label = Label(root, text = "This is a BERT chatbot, pls enter a question below", font=("Arial Bold", 25))
label.place(x = 95, y = 25)

scibert_button = Button(root, text = "SciBERT answer", font = 40, bg = 'red', command = sci_on)
scibert_button.place(x=600, y=250)
# this creates the box for where we input text
input_frame = Frame(root, bg='gray', bd=5)
input_frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.1, anchor="n")
# entry of the text
entry = Entry(input_frame, font=40)
entry.place(relwidth=0.65, relheight=1)
# Enter button
button = Button(input_frame, text="Enter", font=40, command= Enter)
button.place(relx=0.68, relheight=1, relwidth=0.3)
# this creates the output frame for the answer of the chatbot
output_frame = Frame(root, bg="#98f5ff")
output_frame.place(relx=0.5, rely=0.22, relwidth=0.75, relheight=0.1, anchor="n")

label = Label(output_frame, text="Answer Bot: ", font=40, bg="#00d2ff")
label.place(relx=0.05, rely=0.35)


# button to quit the program
button_quit = Button(root, text = "Stop talking to chatbot", command = root.quit, font = ("Arial Bold", 15), padx = 30, pady = 30)
button_quit.place(x = 285, y = 620)
root.mainloop()