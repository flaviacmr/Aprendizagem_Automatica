# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:35:15 2018

@author: f004197
"""

"""
Created on Tue Nov  6 09:45:30 2018

@author: f004197
"""


# Importação Bibliotecas

# Workframes

import pandas as pd
import numpy as np

# Output consla
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)

# Plots

from pandas.plotting import scatter_matrix
import seaborn  as sns


from matplotlib.pyplot import *

import matplotlib.pyplot as plt
from matplotlib import cm as cm


matplotlib.style.use('ggplot')

plt.style.available
plt.style.use('seaborn-paper')

# para cramer's V
from   scipy.stats               import ttest_ind
from   scipy.stats.distributions import chi2
import scipy.stats               as ss
import scipy.stats               as stats

#Keras

from keras.models import Sequential
import tensorflow as tf
# Importação Dados

# Dados para calculo correlaçãoi Cramer e Chi2

df_modelo = pd.read_excel('D:\\Users\\f004197\\Desktop\\TRABALHO FIDLIDADE\\Averiguacoes_Final\\Amostra_3_(64363).xlsx')

#casa

df_modelo = pd.read_excel('C:\\Users\\atvfcmr\\Desktop\\Fraude\\Amostra_3_(64363).xlsx')

hm = ['COUNT_DISTINCT_of_SUBSINIS',  'ObjetRegul1', 'CoberturaNaoUsada1', 'Cobertura1', 'indexclu', 'CAPITAL_APOL', 'CD_COND_GERAIS',
'COBRTIP', 'TIPFRACC', 'TIPCONT', 'TP_CRC_SINIST', 'CORPORAC', 'produto', 'Sinistro_semana_Ano', 'Sinistro_dia_semana', 'Sinistro_quarter',
'Sinistro_Ano', 'ACCAUSA', 'TIPOSIN', 'DANOSMC', 'DEFRESP', 'TPRCASO', 'dias_ate_abrir_proc', 'APCONC', 'Diff_CustoProv', 'FORMENTP',
'ORIGPART' ,'CONDSEXO' ,'idade_condutor', 'dias_com_carta' ,'VIATANO','ViatsAno_1', 'flag_autoridade', 'flag_oficina_nossa',
'flag_testemunhas', 'Fraude_Final']

df_m1 = df_modelo[hm]

##



df_m1 = df_modelo[['COUNT_DISTINCT_of_SUBSINIS', 'ObjetRegul1', 'Dano_viatura1', 'Object_Type1', 'ID_Objeto1', 'Tempo_Apolice_Ate_Sinistro', 
                   'Duracao_apolice', 
'sin_mes_apolice', 'cos_mes_apolice', 'CoberturaNaoUsada1', 'Cobertura1', 'indexclu', 'PREMTTAN', 
'CAPITAL_APOL', 'CD_COND_GERAIS', 'COBRTIP', 'TIPFRACC', 'TIPCONT', 'TP_CRC_SINIST', 'CORPORAC', 'produto', 'Sin_Hr_Sinistro', 'Cos_Hr_Sinistro', 
 'Sin_SemanaAno', 'Cos_SemanaAno', 'Sin_DiaSemana', 'Cos_DiaSemana',  'Cos_Quarter', 
'Sin_Quarter', 'Cos_Mes', 'Sin_Mes', 'ACCAUSA', 'TIPOSIN', 'DANOSMC', 'DEFRESP', 'TPRCASO',
 'dias_ate_abrir_proc', 'APCONC', 'CTOTPRCS', 'PINIPRCS', 'Diff_CustoProv', 'FORMENTP', 'ORIGPART', 'CONDSEXO', 'idade_condutor',
 'dias_com_carta', 'VIATANO', 'ViatsAno_1', 'VIATCIL', 'VIATMARC', 'flag_autoridade', 'flag_oficina_nossa', 'flag_testemunhas', 'Fraude_Final']]

# Define which columns should be encoded vs scaled
 df_m1['ViatsAno_1'] =  df_m1['ViatsAno_1'].astype('float64')
 df_m1['Diff_Anos_Viat'] = df_m1['VIATANO'] -  df_m1['ViatsAno_1']

#columns_to_scale2  = df_m1.select_dtypes(include=[np.number]).columns
columns_to_scale  = ['COUNT_DISTINCT_of_SUBSINIS',  'Tempo_Apolice_Ate_Sinistro', 'Duracao_apolice', 'sin_mes_apolice', 
                     'cos_mes_apolice',  'Sin_Hr_Sinistro', 
                     'Cos_Hr_Sinistro', 'Sin_SemanaAno', 'Cos_SemanaAno', 'Sin_DiaSemana', 'Cos_DiaSemana', 'Cos_Quarter', 'Sin_Quarter', 
                     'Cos_Mes', 'Sin_Mes',  'dias_ate_abrir_proc', 'CTOTPRCS', 'PINIPRCS', 'Diff_CustoProv', 'PREMTTAN', 'CAPITAL_APOL', 
                     'idade_condutor', 'dias_com_carta', 'Diff_Anos_Viat', 'VIATCIL']

columns_to_encode = df_m1[df_m1.columns.difference(['COUNT_DISTINCT_of_SUBSINIS', 'Object_Type1', 'ID_Objeto1', 'Tempo_Apolice_Ate_Sinistro', 
                                                    'Duracao_apolice', 'sin_mes_apolice', 
                     'cos_mes_apolice','PREMTTAN', 'CAPITAL_APOL',  'Sin_Hr_Sinistro', 
                     'Cos_Hr_Sinistro', 'Sin_SemanaAno', 'Cos_SemanaAno', 'Sin_DiaSemana', 'Cos_DiaSemana', 'Cos_Quarter', 'Sin_Quarter', 
                     'Cos_Mes', 'Sin_Mes',  'dias_ate_abrir_proc', 'CTOTPRCS', 'PINIPRCS', 'Diff_CustoProv',  
                     'idade_condutor', 'dias_com_carta', 'Diff_Anos_Viat', 'VIATCIL' ])].columns


columns_to_encode = columns_to_encode.drop(['Fraude_Final','flag_autoridade', 'flag_oficina_nossa', 'flag_testemunhas'])



#### Teste a Menos Variaveis

columns_to_scale2 = ['COUNT_DISTINCT_of_SUBSINIS','Tempo_Apolice_Ate_Sinistro','dias_com_carta','Duracao_apolice','idade_condutor', 
                     'Diff_CustoProv','Sin_Hr_Sinistro',  'Cos_Hr_Sinistro' ,'CAPITAL_APOL','Diff_CustoProv']


columns_to_encode2 = ['ACCAUSA']


 # PARA Ccolumns_to_encode2
# Instantiate encoder/scaler
 from sklearn.preprocessing import StandardScaler, OneHotEncoder
scaler = StandardScaler()
ohe    = OneHotEncoder(sparse=False)

#encoded_columns =  ohe.fit_transform(df_m1[columns_to_encode])


# Scale and Encode Separate Columns
df_m1[columns_to_scale] = df_m1[columns_to_scale].fillna(0)
scaled_array = scaler.fit_transform(df_m1[columns_to_scale])

scaled_columns  = pd.DataFrame(scaled_array, index = df_m1[columns_to_scale].index, columns=columns_to_scale) 

df_codificar = df_m1[columns_to_encode].astype('category')

encoded_columns = pd.get_dummies(df_codificar)


encoded_columnsF = pd.concat([df_m1[['Fraude_Final','flag_autoridade', 'flag_oficina_nossa', 'flag_testemunhas']], encoded_columns], axis=1 )


#convert to pandas dataframe
df_processed = pd.concat([scaled_columns,encoded_columnsF], axis=1)


df_proc2 = df_processed[columns_to_scale + ['Fraude_Final','flag_autoridade', 'flag_oficina_nossa', 'flag_testemunhas']]


