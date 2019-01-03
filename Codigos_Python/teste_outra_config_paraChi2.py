# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 15:09:35 2018

@author: F004197
"""


teste1 = pd.crosstab(df_aveg_sem_perdas['ID_MOTAVERIG_Averiguacao'],df_aveg_sem_perdas['fraude']) 
chi2, p, dof, expected = stats.chi2_contingency(teste1)

list_of_proved = df_aveg_sem_perdas['fraude'].unique()
list_of_averigs = df_aveg_sem_perdas['ID_MOTAVERIG_Averiguacao'].astype(int).sort_values(ascending= True).unique()


confusion = []
for mot_averig in list_of_averigs:
    for result_fraud in list_of_proved:
        cond = (df_aveg_sem_perdas['ID_MOTAVERIG_Averiguacao'].astype('str') == mot_averig) & (df_aveg_sem_perdas['fraude'] == result_fraud )      
        confusion.append(cond.sum())



confusion_matrix = np.array(confusion).reshape(len(list_of_proved),len(list_of_averigs))


def cramers_stat(confusion_matrix):
    chi2 = ss.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum()
    return np.sqrt(chi2 / (n*(min(confusion_matrix.shape)-1)))

def chi2(confusion_matrix):
    chi2 = ss.chi2_contingency(confusion_matrix, correction=False)[0]
    print(chi2)    return chi2


result_chi2 = chi2(confusion_matrix) # 600
print(result_chi2)
result = cramers_stat(confusion_matrix) # 0.308
print(result)

