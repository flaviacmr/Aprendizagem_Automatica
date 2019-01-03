# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 16:53:14 2018

@author: f004197


Modelo - validar mais feautures para medir o sucesso das averiguações

       -verdadeiros positivos são dados pelo Resultado no GIF, para o pior caso possível ( quando INCIDENTE_RESULT_DESC = 'Provada' e ignorando outros casos 
       como or exemplo, Fortes Indicios mas prova)
"""


#Bibliotecas e configs

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn  as sns



from matplotlib import cm as cm
import matplotlib
matplotlib.style.use('ggplot')

##configs 

## graficos
plt.style.available
plt.style.use('seaborn-paper')



#consola output
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

#Modelos do scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

# Importação Dados

df_aveg = pd.read_excel('D:\\Users\\f004197\\Desktop\\docs python averiguacoes\Averigs_Sem_PerdaTotal_MaisFeatures.xlsx') 

# vaiavel Fraude e Not Fraude - pior cenário possível

df_aveg['fraude'] = np.where(df_aveg['INCIDENTRESULT_DESC'] == 'Provada', 'Fraude', 'Not_fraude')
#Tratamento dos dados

col_names = df_aveg.columns
col_numeric = ['COUNT_DISTINCT_of_ID_AVERIGUACAO','NUM_VIATURAS','NUM_SINISTRADOS','SIN_CUSTOMEDIO']




col_dates = [
'RECEIVEDON_GIF',    'CHANGEDON_GIF',    'DATA_SINISTRO',    'DATA_PARTICIPACAO',    'DATA_ABERTURA',    'DATA_ENCERRAMENTO',        'DATE_UPDATE', 
'IDS_DATA_ENVIO',    'DATE_RESP_ANALISADA',    'DATA_PRE_ENCERRAMENTO',    
'DATA_REEMBOLSO',    'DATA_ENTRADA_PLATAFORMA',    'DATA_ENTRADA_COMPANHIA', 'DATA_PEDIDO_PARTICIPACAO',    'DATA_RECLAMACAO',    'DATA_DECISAO_FINAL',    
'DATA_ASSUNCAO_RESP',    'DATA_INICIO_SUSPENDE',    'DATA_FIM_SUSPENDE',    
'DATA_PROPOSTA_ARB',    'DATA_RECUSA_ARB',    'DATA_DECISAO_ARB', 'DATA_COMUNIC_SEG','DATA_MIGRACAO_SINAUTO']



col_categ = [
'ID_MOTAVERIG_Averiguacao','INCIDENTTYPE_DESC','ID_AVERIGUADOR_GIF','INCIDENTSOURCE_DESC','PROCESSSTATUS_DESC','CONTENCIOSO',
'REGUL_GERAL_SIN','MOTOCICLOS','SEXO_CONDUTOR','COND_ASSIST_FERIDOS','TESTEMUNHAS','RESP_SEGURADO','ID_NOTIFICADOR','ID_UTILIZADOR',
'ID_FORMATOPART','ID_MODOABERTURA','ID_CAUSASIN','ID_GABINETECVERDE','HOUVE_ACIDENTE','SISTEMA_SUGERIU','CREDORLOCADOR','CHOQUE_CADEIA',
'SINISTRO_VALIDO','SEGURADO_ESPECIAL','DAAA_SEGURADO','PAGAMENTOS_MIGRADOS','RESPONSABILIDADE_ANALISADA','PREMIUM_RECEIPT_STATUS',
'DESC_REEMB','BONUS_MALUS_SUGERIDO','VALORPAGO_SIN_MIGRADO','RECLAMACAO_SEG','ENVIADO_FGA','PROTOCOLO','SYSTEM_REFERENCE',
'VALORRECEBIDO_SIN_MIGRADO','INSURED_PROTECTION','CONTROL_QUAL','ID_NATUREZAAV_Averiguacao','INCIDENTRESULT_DESC','ID_ESTADO_GIF',
'ID_CODPOSTAL_GIF','INCIDENTSUBTYPE_DESC','PROCESSTYPE_DESC','HORA_SINISTRO','DESC_ACIDENTE','EMBATE_VIATURAS','INTER_AUTORIDADE',
'PROV_INICIAL','RECURSOCASA','ID_LOCALNOTIFICACAO','ID_EQUIPAUTIL','ID_TIPOSINISTRO','ID_CONFAPOLICE','ID_TPR','ID_ENTIDADE',
'ID_CONDUTOR','DECISAO_AVERIGUACAO','ID_GARANTIA_INDEMN','ID_MOTREABERTURA','FLAGGESTOR','ID_ESTADO','N_SINISTRO_MIGRADO',
'SINISTRO_ICI','ID_RESP_ANALISADA','ID_UTILIZADOR_ANTERIOR','N_SIN_CONG','QUARTA_DIRECTIVA','ENQUADRAMENTO','FACTORES',
'MARCA','MIGRADO_SINAUTO','REF_REPRESENTADA','RECONDICIONAMENTO','GPS_COORD','Sub_sin','flag_Gep','Retratação','PROCESSSTAGE_DESC',
'LINEOFBUSINESS_DESC','FORMAENTRADA','SITUACAO_SINISTRO','MATRICULA','CONSEQUENCIA','SEG_VIATURA_ACIDENTE','AUTORIDADE','ENQUAD_TPR',
'DATACASA','ID_FRONTOFFICE','ID_PARTFORMAL','ID_SEGURADORA','ID_CONCELHO','ID_MOTIVOATRIB','ID_UTILABRIU','CONDUTOR_HABITUAL',
'RESP_1CONTACTO','TPR_VIATURA','RESSEGURO_PLATAFORMA','AUTO_ALLOC_OFF','REEMBOLSO','USER_UPDATE','IDS_REEM_CRIADO','MOTIVO_SUGESTAO',
'REPOS_CAPITAL','REEMB_ANALISADO','SIN_IRT_ESPECIAL','ID_MOT_ENQUADRAMENTO','ALLOC_DL','SOPRE','ID_TIPO_SIN','SITUACAO_COGEN','FHS',
'VALOR_RECOND','GIF','COUNTER_DAAA','PROC_ESPECIAL','DESPORTIVO']



col_categ_reduced = [
 'ID_MOTAVERIG_Averiguacao','INCIDENTTYPE_DESC','ID_AVERIGUADOR_GIF','INCIDENTSOURCE_DESC', 'COND_ASSIST_FERIDOS',
 'TESTEMUNHAS','RESP_SEGURADO', 
 ]

df_aveg[col_categ]=df_aveg[col_categ].replace(['?','-'], ' ')


# tudo a variaveis categóricas
for col in col_names:
    df_aveg[col] = df_aveg[col].astype('str',copy=True)
    df_aveg[col] = df_aveg[col].replace(['.'], np.nan)

# variaveis numéricas
for col in col_numeric:
    df_aveg[col] = df_aveg[col].replace(np.nan, 0 )
    df_aveg[col] = df_aveg[col].astype('int',copy=True)

#datas 
for col in col_dates:    
    df_aveg[col] = df_aveg[col].astype('str',copy=True).str[:9]


for col in col_dates:
        df_aveg[col]=pd.to_datetime(df_aveg[col], format='%d%b%Y', utc=False) 
        print(col)



        
# Representação dos dados -- é o pedaço de código que está a correr

for col in col_names:
    sns.boxplot(x=col, y='COUNT_DISTINCT_of_ID_AVERIGUACAO', hue="fraude", data=df_aveg)
           
#Tratamento dos dados
    

#df_bin = pd.get_dummies(df_aveg[col_categ])



# Para os Modelos - classificação binária fraude\not_fraude ----> tentar depois acrescentar masi categorias 
 

#df_fin = pd.concat([df_aveg[col_numeric] ,df_bin], axis=1)

#df_fin.corr()

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
for i in col_names:
    df_aveg[i] = le.fit_transform(df_aveg[i])
   
var_predict = df_aveg['fraude']

df_aveg['fraude'].replace({'Not_fraude':0,'Fraude':1},inplace=True)

     



col_analise =  [c for c in col_names if c not in col_dates]


le = preprocessing.LabelEncoder()

for col in col_analise:
    if df_aveg[col_analise].dtypes == object:
        df_aveg[col_analise] = le.fit_transform(df_aveg[col_analise].astype(str))
    else:
        pass
print(col_analise)        
   


from sklearn import tree

clf = tree.DecisionTreeClassifier()


clf = clf.fit(df_aveg, var_predict)