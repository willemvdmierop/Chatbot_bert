import torch
import numpy as np
import pandas as pd

#metrics = pd.read_csv("metrics_final_word_gen.csv")
metrics = torch.load("word_gen_metrics.pkl")
Gen_metrics = metrics['gen_metrics']

Q1_metrics = Gen_metrics[0]
Q2_metrics = Gen_metrics[1]
Q3_metrics = Gen_metrics[2]
table = []
import xlsxwriter
workbook = xlsxwriter.Workbook('WordGeneration_parameters_fineTuned.xlsx')
worksheet = workbook.add_worksheet()
row = 0
col = 0


for i in range(len(Q1_metrics)):
    table.append([Q1_metrics[i][0],Q1_metrics[i][1],Q1_metrics[i][2],Q2_metrics[i][2],Q3_metrics[i][2]])
    worksheet.write(row, col, Q1_metrics[i][0])
    worksheet.write(row, col + 1, Q1_metrics[i][1])
    worksheet.write(row, col + 2, Q1_metrics[i][2])
    worksheet.write(row, col + 3, Q2_metrics[i][2])
    worksheet.write(row, col + 4, Q3_metrics[i][2])
    row += 1

workbook.close()

df = pd.DataFrame(table)
df.to_csv("table_word_gen_fineTuned.csv")
print(table)


