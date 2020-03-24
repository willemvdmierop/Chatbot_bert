from tkinter import *
from PIL import Image, ImageTk

root = Tk()
root.title("Chatbot BERT")
HEIGHT = 700
WIDTH = 800
canvas = Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()


img = ImageTk.PhotoImage(Image.open("/Users/willemvandemierop/Documents/Master AI/Pycharm/Chatbot_bert/City_logo.jpg"))
root.iconphoto(False, img)
background_label = Label(root, image=img)
background_label.place(x=0, y=0, relheight=1, relwidth=1)

label = Label(root, text = "This is a chatbot made from bert, pls enter a question below", font=("Arial Bold", 25))
label.place(x = 50, y = 25)

# this input needs to be passed to the chatbot
def Enter():
    input = entry.get()
    output_txt = "Hello " + str(input) + " you sound like a smart man that knows a lot about AI"
    label = Label(output_frame, text=output_txt, font=40, bg="#00d2ff")
    label.place(relx=0.2, rely=0.35)

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
