import pickle
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
metrics = torch.load("metrics.pkl")
Q_metrics = metrics['q_metrics']

print(Q_metrics)
Q1_metrics = Q_metrics[0]
Q2_metrics = Q_metrics[1]
Q3_metrics = Q_metrics[2]

df_q1 = pd.DataFrame(Q1_metrics)
df_q2 = pd.DataFrame(Q2_metrics)
df_q3 = pd.DataFrame(Q3_metrics)
#df_q1.to_csv("Q1-test.csv")
print(df_q1[0])

x = np.linspace(start = 0, stop = 39, num= 1680)
print(x)
print(x.shape)

fig, ax = plt.subplots(3,1,figsize = (12,12))

ax[0].plot(x,df_q1[0], label = 'BLEU_score')
ax[0].plot(x,df_q1[1], label = 'Precision')
ax[0].plot(x,df_q1[2], label = 'Recal')
ax[0].plot(x,df_q1[3], label = 'F1')
ax[0].set_title("Scores")
ax[0].set_xlabel("count")
ax[0].set_ylabel("Score")
ax[0].legend()

ax[1].plot(x,df_q2[0], label = 'BLEU_score')
ax[1].plot(x,df_q2[1], label = 'Precision')
ax[1].plot(x,df_q2[2], label = 'Recal')
ax[1].plot(x,df_q2[3], label = 'F1')
ax[1].set_title("Scores")
ax[1].set_xlabel("count")
ax[1].set_ylabel("Score")
ax[1].legend()

ax[2].plot(x,df_q2[0], label = 'BLEU_score')
ax[2].plot(x,df_q2[1], label = 'Precision')
ax[2].plot(x,df_q2[2], label = 'Recal')
ax[2].plot(x,df_q2[3], label = 'F1')
ax[2].set_title("Scores")
ax[2].set_xlabel("count")
ax[2].set_ylabel("Score")
ax[2].legend()
fig.subplots_adjust(hspace=0.5)
plt.show()