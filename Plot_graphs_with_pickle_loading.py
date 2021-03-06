import pickle
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
metrics = torch.load("Metrics/metrics_SCIBERT_lr001_wd0001_FINAL.pkl")
Q_metrics = metrics['q_metrics']
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
stop = 100
x = np.linspace(start = 0, stop = stop, num= stop*42)
print(x)
print(x.shape)

if single_plots:
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(x,df_q1[0].rolling(window).mean(), label = 'BLEU score')
    ax.plot(x,df_q1[1].rolling(window).mean(), label = 'Precision')
    ax.plot(x,df_q1[2].rolling(window).mean(), label = 'Recal')
    ax.plot(x,df_q1[3].rolling(window).mean(), label = 'F1')
    ax.set_title("Who is she?")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Score")
    ax.legend(bbox_to_anchor=(1.145, 1.03))
    plt.tight_layout()
    plt.show()
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(x, df_q2[0].rolling(window).mean(), label='BLEU score')
    ax.plot(x, df_q2[1].rolling(window).mean(), label='Precision')
    ax.plot(x, df_q2[2].rolling(window).mean(), label='Recal')
    ax.plot(x, df_q2[3].rolling(window).mean(), label='F1')
    ax.set_title("Are you okay?")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Score")
    ax.legend(bbox_to_anchor=(1.145, 1.03))
    plt.tight_layout()
    plt.show()
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(x, df_q3[0].rolling(window).mean(), label='BLEU score')
    ax.plot(x, df_q3[1].rolling(window).mean(), label='Precision')
    ax.plot(x, df_q3[2].rolling(window).mean(), label='Recal')
    ax.plot(x, df_q3[3].rolling(window).mean(), label='F1')
    ax.set_title("why?")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Score")
    ax.legend(bbox_to_anchor=(1.145, 1.03))
    plt.tight_layout()
    plt.show()
else:
    fig, ax = plt.subplots(3,1,figsize = (12,10))
    ax[0].plot(x,df_q1[0].rolling(window).mean(), label = 'BLEU score')
    ax[0].plot(x,df_q1[1].rolling(window).mean(), label = 'Precision')
    ax[0].plot(x,df_q1[2].rolling(window).mean(), label = 'Recal')
    ax[0].plot(x,df_q1[3].rolling(window).mean(), label = 'F1')
    ax[0].set_title("Who is she?")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Score")
    ax[0].legend(bbox_to_anchor=(1.145, 1.03))

    ax[1].plot(x,df_q2[0].rolling(window).mean(), label = 'BLEU score')
    ax[1].plot(x,df_q2[1].rolling(window).mean(), label = 'Precision')
    ax[1].plot(x,df_q2[2].rolling(window).mean(), label = 'Recal')
    ax[1].plot(x,df_q2[3].rolling(window).mean(), label = 'F1')
    ax[1].set_title("Are you okay?")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Score")
    ax[1].legend(bbox_to_anchor=(1.145, 1.03))

    ax[2].plot(x,df_q3[0].rolling(window).mean(), label = 'BLEU score')
    ax[2].plot(x,df_q3[1].rolling(window).mean(), label = 'Precision')
    ax[2].plot(x,df_q3[2].rolling(window).mean(), label = 'Recal')
    ax[2].plot(x,df_q3[3].rolling(window).mean(), label = 'F1')
    ax[2].set_title("why?")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Score")
    ax[2].legend(bbox_to_anchor=(1.145, 1.03))
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    plt.show()