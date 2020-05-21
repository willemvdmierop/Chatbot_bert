import pickle
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
metrics = torch.load("Metrics/metrics_SCIBERT_lr001.pkl")
Q_metrics = metrics['q_metrics']
stop = 100
single_plots = False
Q1_metrics = Q_metrics[0]
Q2_metrics = Q_metrics[1]
Q3_metrics = Q_metrics[2]

df_q1 = pd.DataFrame(Q1_metrics)
df_q2 = pd.DataFrame(Q2_metrics)
df_q3 = pd.DataFrame(Q3_metrics)
#df_q1.to_csv("Q1-test.csv")
print(df_q1[0])
window = 20
x = np.linspace(start = 0, stop = 81, num= 3402)
print(x)
print(x.shape)

if single_plots:
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.plot(x,df_q1[0].rolling(window).mean(), label = 'BLEU_score')
    ax.plot(x,df_q1[1].rolling(window).mean(), label = 'Precision')
    ax.plot(x,df_q1[2].rolling(window).mean(), label = 'Recal')
    ax.plot(x,df_q1[3].rolling(window).mean(), label = 'F1')
    ax.set_title("Who is she?")
    ax.set_xlabel("count")
    ax.set_ylabel("Score")
    ax.legend()
    plt.show()
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.plot(x, df_q2[0].rolling(window).mean(), label='BLEU_score')
    ax.plot(x, df_q2[1].rolling(window).mean(), label='Precision')
    ax.plot(x, df_q2[2].rolling(window).mean(), label='Recal')
    ax.plot(x, df_q2[3].rolling(window).mean(), label='F1')
    ax.set_title("Are you okay?")
    ax.set_xlabel("count")
    ax.set_ylabel("Score")
    ax.legend()
    plt.show()
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.plot(x, df_q3[0].rolling(window).mean(), label='BLEU_score')
    ax.plot(x, df_q3[1].rolling(window).mean(), label='Precision')
    ax.plot(x, df_q3[2].rolling(window).mean(), label='Recal')
    ax.plot(x, df_q3[3].rolling(window).mean(), label='F1')
    ax.set_title("why?")
    ax.set_xlabel("count")
    ax.set_ylabel("Score")
    ax.legend()
    plt.show()
else:
    fig, ax = plt.subplots(3,1,figsize = (12,10))
    ax[0].plot(x,df_q1[0].rolling(window).mean(), label = 'BLEU_score')
    ax[0].plot(x,df_q1[1].rolling(window).mean(), label = 'Precision')
    ax[0].plot(x,df_q1[2].rolling(window).mean(), label = 'Recal')
    ax[0].plot(x,df_q1[3].rolling(window).mean(), label = 'F1')
    ax[0].set_title("Who is she?")
    ax[0].set_xlabel("count")
    ax[0].set_ylabel("Score")
    ax[0].legend()

    ax[1].plot(x,df_q2[0].rolling(window).mean(), label = 'BLEU_score')
    ax[1].plot(x,df_q2[1].rolling(window).mean(), label = 'Precision')
    ax[1].plot(x,df_q2[2].rolling(window).mean(), label = 'Recal')
    ax[1].plot(x,df_q2[3].rolling(window).mean(), label = 'F1')
    ax[1].set_title("Are you okay?")
    ax[1].set_xlabel("count")
    ax[1].set_ylabel("Score")
    ax[1].legend()

    ax[2].plot(x,df_q3[0].rolling(window).mean(), label = 'BLEU_score')
    ax[2].plot(x,df_q3[1].rolling(window).mean(), label = 'Precision')
    ax[2].plot(x,df_q3[2].rolling(window).mean(), label = 'Recal')
    ax[2].plot(x,df_q3[3].rolling(window).mean(), label = 'F1')
    ax[2].set_title("why?")
    ax[2].set_xlabel("count")
    ax[2].set_ylabel("Score")
    ax[2].legend()
    fig.subplots_adjust(hspace=0.5)
    plt.show()