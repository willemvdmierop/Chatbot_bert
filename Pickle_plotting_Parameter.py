import pickle
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
metrics1 = torch.load("Metrics/metrics_65epochs_gradientclipping_WD001_lr0001.pkl")
metrics2 = torch.load("Metrics/metrics_WD_001_lr0001_Sched_True_60_epochs.pkl")
metrics3 = torch.load("Metrics/metrics_WD_0001_lr0001_Sched_True_60_epochs.pkl")
Q_metrics1 = metrics1['q_metrics']
Q_metrics2 = metrics2['q_metrics']
Q_metrics3 = metrics3['q_metrics']
stop = 100
print(Q_metrics1)
single_plots = False
Q1_metrics1 = Q_metrics1[0]
Q2_metrics1 = Q_metrics1[1]
Q3_metrics1 = Q_metrics1[2]

Q1_metrics2 = Q_metrics2[0]
Q2_metrics2 = Q_metrics2[1]
Q3_metrics2 = Q_metrics2[2]

Q1_metrics3 = Q_metrics3[0]
Q2_metrics3 = Q_metrics3[1]
Q3_metrics3 = Q_metrics3[2]

df_q11 = pd.DataFrame(Q1_metrics1).loc[:2519]
df_q21 = pd.DataFrame(Q2_metrics1).loc[:2519]
df_q31 = pd.DataFrame(Q3_metrics1).loc[:2519]

df_q12 = pd.DataFrame(Q1_metrics2)
df_q22 = pd.DataFrame(Q2_metrics2)
df_q32 = pd.DataFrame(Q3_metrics2)

df_q13 = pd.DataFrame(Q1_metrics3)
df_q23 = pd.DataFrame(Q2_metrics3)
df_q33 = pd.DataFrame(Q3_metrics3)
#df_q1.to_csv("Q1-test.csv")
#print(df_q1[0])
window = 20
x = np.linspace(start = 0, stop = 60, num= 2520)
print(x)
print(x.shape)


fig, ax = plt.subplots(3,1,figsize = (12,12))

ax[0].plot(x,df_q11[0].rolling(window).mean(), label = 'BLEU_score WD 0.001 no schedule')
ax[0].plot(x,df_q12[0].rolling(window).mean(), label = 'BLEU_score WD 0.001 & cosine with warmup')
ax[0].plot(x,df_q13[0].rolling(window).mean(), label = 'BLEU_score WD 0.0001 & cosine with warmup')
#ax[0].plot(x,df_q1[1].rolling(window).mean(), label = 'Precision')
#ax[0].plot(x,df_q1[2].rolling(window).mean(), label = 'Recal')
#ax[0].plot(x,df_q1[3].rolling(window).mean(), label = 'F1')
ax[0].set_title("Who is she?")
ax[0].set_xlabel("count")
ax[0].set_ylabel("Score")
ax[0].legend()

ax[1].plot(x,df_q21[0].rolling(window).mean(), label = 'BLEU_score WD 0.001')
ax[1].plot(x,df_q22[0].rolling(window).mean(), label = 'BLEU_score WD 0.001 & cosine with warmup')
ax[1].plot(x,df_q23[0].rolling(window).mean(), label = 'BLEU_score WD 0.0001 & cosine with warmup')
#ax[1].plot(x,df_q2[1].rolling(window).mean(), label = 'Precision')
#ax[1].plot(x,df_q2[2].rolling(window).mean(), label = 'Recal')
#ax[1].plot(x,df_q2[3].rolling(window).mean(), label = 'F1')
ax[1].set_title("Are you okay?")
ax[1].set_xlabel("count")
ax[1].set_ylabel("Score")
ax[1].legend()

ax[2].plot(x,df_q31[0].rolling(window).mean(), label = 'BLEU_score WD 0.001')
ax[2].plot(x,df_q32[0].rolling(window).mean(), label = 'BLEU_score WD 0.001 & cosine with warmup')
ax[2].plot(x,df_q33[0].rolling(window).mean(), label = 'BLEU_score WD 0.0001 & cosine with warmup')
#ax[2].plot(x,df_q2[1].rolling(window).mean(), label = 'Precision')
#ax[2].plot(x,df_q2[2].rolling(window).mean(), label = 'Recal')
#ax[2].plot(x,df_q2[3].rolling(window).mean(), label = 'F1')
ax[2].set_title("why?")
ax[2].set_xlabel("count")
ax[2].set_ylabel("Score")
ax[2].legend()
fig.subplots_adjust(hspace=0.5)
plt.show()