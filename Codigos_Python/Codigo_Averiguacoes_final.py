# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 12:17:49 2018

@author: f004197
"""

# Importação Bibliotecas

# Workframes

import pandas as pd
import numpy as np

# Output consla
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Plots

from pandas.plotting import scatter_matrix
import seaborn  as sns

import matplotlib.pyplot as plt
from matplotlib import cm as cm
matplotlib.style.use('ggplot')

plt.style.available
plt.style.use('seaborn-paper')

#grafico para arvores decisão
import graphviz as gv


# para cramer's V
from   scipy.stats               import ttest_ind
from   scipy.stats.distributions import chi2
import scipy.stats               as ss
import scipy.stats               as stats




# cross validation
from sklearn.model_selection import cross_val_score

# Modelos: decisionTreeclassifier

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix

from sklearn.tree.export import export_graphviz
from sklearn.feature_selection import mutual_info_classif

# Para Clusters

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from scipy import cluster

# Importação Dados

# Dados para calculo correlaçãoi Cramer e Chi2

df_aveg_sem_perdas = pd.read_excel('D:\\Users\\f004197\\Desktop\\docs python averiguacoes\Averiguacoes_em_Linha.xlsx')

# Dados para Modelo

df_aveg = pd.read_excel('D:\\Users\\f004197\\Desktop\\docs python averiguacoes\Averigs_Sem_PerdaTotal_MaisFeatures.xlsx')

# Dados para Modelo igual ao anterior mas com data Averiguação

df_aveg = pd.read_excel('D:\\Users\\f004197\\Desktop\\Averiguacoes_em_Linha_com_data_Aveg.xlsx')


#########################################################################################################################################


#
#df_man = df_aveg[['DATA_ALTERACAO_data_Averiguacao','ID_MOTAVERIG_Averiguacao','ID_NATUREZAAV_Averiguacao','Process_ID']]
#df_man['ID_MOTAVERIG_Averiguacao'] = df_man['ID_MOTAVERIG_Averiguacao'].astype(str)
#df_man['ID_NATUREZAAV_Averiguacao'] = df_man['ID_NATUREZAAV_Averiguacao'].astype(str)
#
#

df_aveg = pd.merge(df_aveg, df_man , 
         how='inner', on=None, left_on=['ID_MOTAVERIG_Averiguacao','ID_NATUREZAAV_Averiguacao','Process_ID'], 
         right_on=['ID_MOTAVERIG_Averiguacao','ID_NATUREZAAV_Averiguacao','Process_ID'],
         sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)


df_aveg['DATA_ALTERACAO_data_Averiguacao'] = df_aveg['DATA_ALTERACAO_data_Averiguacao'].astype('str',copy=True).str[:10]
df_aveg['Hora_ALTERACAO_data_Averiguacao'] = df_aveg['DATA_ALTERACAO_data_Averiguacao'].astype('str',copy=True).str[11:-3]


df_aveg['DATA_ALTERACAO_data_Averiguacao']=pd.to_datetime(df_aveg['DATA_ALTERACAO_data_Averiguacao'], utc=False)





######  Análise a  Ficheiro df_aveg_sem_perdas

# Colunas para df_aveg_sem_perdas



testColumns = ['ID_MOTAVERIG_Averiguacao', 'ID_NATUREZAAV_Averiguacao',
       'INCIDENTTYPE_DESC',
       'COUNT_DISTINCT_of_Sub_sin', 'COUNT_DISTINCT_of_flag_Gep',
       'ID_COORDENADOR_GIF', 'ID_AVERIGUADOR_GIF', 'ID_CODPOSTAL_GIF',
       'Retratação',  'datedif_gif','mes_recebido','mes_modificado','diaSemana_recebido', 'diaSemana_modificado',
       'quarter_recebido','quarter_modificado','semanaAno_recebido','semanaAno_modificado','dummyCat' ]
#
#
#convert_columns = ['ID_MOTAVERIG_Averiguacao', 'ID_NATUREZAAV_Averiguacao',
#       'INCIDENTTYPE_DESC','ID_COORDENADOR_GIF', 'ID_AVERIGUADOR_GIF', 'ID_CODPOSTAL_GIF','datedif_gif']

df_aveg_sem_perdas['fraude'] = np.where(df_aveg_sem_perdas['INCIDENTRESULT_DESC'] == 'Provada', 'Fraude', 'Not_fraude')

df_aveg_sem_perdas['mes_recebido']  = df_aveg_sem_perdas['Recebida_GIF'].dt.month
df_aveg_sem_perdas['mes_modificado']  = df_aveg_sem_perdas['Modificada_GIF'].dt.month

df_aveg_sem_perdas['diaSemana_recebido']  = df_aveg_sem_perdas['Recebida_GIF'].dt.dayofweek
df_aveg_sem_perdas['diaSemana_modificado']  = df_aveg_sem_perdas['Modificada_GIF'].dt.dayofweek

df_aveg_sem_perdas['quarter_recebido']  = df_aveg_sem_perdas['Recebida_GIF'].dt.quarter
df_aveg_sem_perdas['quarter_modificado']  = df_aveg_sem_perdas['Modificada_GIF'].dt.quarter

df_aveg_sem_perdas['semanaAno_recebido']  = df_aveg_sem_perdas['Recebida_GIF'].dt.week
df_aveg_sem_perdas['semanaAno_modificado']  = df_aveg_sem_perdas['Modificada_GIF'].dt.week


df_aveg_sem_perdas['dummyCat'] = np.random.choice([0, 1], size=(len(df_aveg_sem_perdas),), p=[0.5, 0.5]) #para testar a função


#Função que cálcula Chi2

class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        self.dfObserved = None
        self.dfExpected = None

    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        print(result)

    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        self.dfObserved = pd.crosstab(Y,X)
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        self._print_chisquare_result(colX,alpha)
        print(chi2)

#Initialize ChiSquare Class

cT = ChiSquare(df_aveg_sem_perdas)

#Feature Selection

for var in testColumns:
    cT.TestIndependence(colX=var,colY="fraude")


teste1 = pd.crosstab(df_aveg_sem_perdas['ID_MOTAVERIG_Averiguacao'],df_aveg_sem_perdas['fraude'])

chi2, p, dof, expected = stats.chi2_contingency(teste1)

teste1.reset_index(level=0, inplace=True)



# chi2 600.0299986052639
# dof  39
#  p 1.3090797042510827e-101
#

# Calculo Cramer V

def cramers_corrected_stat(confusion_matrix_p):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix_p, correction=False)[0]
    n = confusion_matrix_p.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix_p.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


v_cramer_corrected = cramers_corrected_stat(teste1) #0.29784802849576003



############ Fim Análise Estatística
################################################################################################################################



#########################################################################################################################################


###### Modelos ML: 

###### Análise a Ficheiro df_aveg: ficheiro que contem as features de Sinistros


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
'DATA_PROPOSTA_ARB',    'DATA_RECUSA_ARB',    'DATA_DECISAO_ARB', 'DATA_COMUNIC_SEG','DATA_MIGRACAO_SINAUTO','DATA_ALTERACAO_data_Averiguacao']



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
'VALOR_RECOND','GIF','COUNTER_DAAA','PROC_ESPECIAL','DESPORTIVO','ID_COORDENADOR_GIF', 'ID_AVERIGUADOR_GIF','fraude']

categ_final = [
'ID_MOTAVERIG_Averiguacao','INCIDENTTYPE_DESC','ID_AVERIGUADOR_GIF','INCIDENTSOURCE_DESC','PROCESSSTATUS_DESC','CONTENCIOSO',
'REGUL_GERAL_SIN','MOTOCICLOS','SEXO_CONDUTOR','COND_ASSIST_FERIDOS','TESTEMUNHAS','RESP_SEGURADO','ID_NOTIFICADOR','ID_UTILIZADOR',
'ID_FORMATOPART','ID_CAUSASIN','ID_GABINETECVERDE','HOUVE_ACIDENTE','SISTEMA_SUGERIU','CREDORLOCADOR','CHOQUE_CADEIA',
'PREMIUM_RECEIPT_STATUS',
'BONUS_MALUS_SUGERIDO',
'VALORRECEBIDO_SIN_MIGRADO','INSURED_PROTECTION','CONTROL_QUAL','ID_NATUREZAAV_Averiguacao','INCIDENTRESULT_DESC','ID_ESTADO_GIF',
'ID_CODPOSTAL_GIF','INCIDENTSUBTYPE_DESC','PROCESSTYPE_DESC','HORA_SINISTRO','DESC_ACIDENTE','EMBATE_VIATURAS','INTER_AUTORIDADE',
'PROV_INICIAL','RECURSOCASA','ID_LOCALNOTIFICACAO','ID_EQUIPAUTIL','ID_TIPOSINISTRO','ID_TPR','N_SINISTRO_MIGRADO',
'SINISTRO_ICI','N_SIN_CONG','QUARTA_DIRECTIVA','ENQUADRAMENTO','FACTORES',
'MARCA','MIGRADO_SINAUTO','Sub_sin','flag_Gep','Retratação','PROCESSSTAGE_DESC',
'LINEOFBUSINESS_DESC','SITUACAO_SINISTRO','MATRICULA','CONSEQUENCIA','SEG_VIATURA_ACIDENTE','AUTORIDADE','ENQUAD_TPR',
'DATACASA','ID_FRONTOFFICE','ID_PARTFORMAL','ID_SEGURADORA','ID_CONCELHO','ID_MOTIVOATRIB','ID_UTILABRIU','CONDUTOR_HABITUAL',
'TPR_VIATURA','RESSEGURO_PLATAFORMA','AUTO_ALLOC_OFF','REEMBOLSO','USER_UPDATE','IDS_REEM_CRIADO','MOTIVO_SUGESTAO',
 'ID_MOT_ENQUADRAMENTO','ID_TIPO_SIN',
'VALOR_RECOND','GIF','COUNTER_DAAA','PROC_ESPECIAL','DESPORTIVO','ID_COORDENADOR_GIF', 'mes_DATA_SINISTRO',
'diaSemana_DATA_SINISTRO','quarter_DATA_SINISTRO','semanaAno_DATA_SINISTRO','fraude']







col_categ_reduced = [
 'ID_MOTAVERIG_Averiguacao','INCIDENTTYPE_DESC','ID_AVERIGUADOR_GIF','INCIDENTSOURCE_DESC', 'COND_ASSIST_FERIDOS',
 'TESTEMUNHAS','RESP_SEGURADO',
 ]


col_sum =['hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos','quarter_sin', 'quarter_cos', 'semanaAno_sin', 'semanaAno_cos',
        'cluster','CodigoPostal_GIF_2', 'dias_ate_participacao', 'dias_ate_respo_analisada', 'dias_entrada_companhia',
        'dt_tratamento',  
        'ID_CAUSASIN','ID_AVERIGUADOR_GIF', 'INCIDENTSOURCE_DESC', 
 'MOTOCICLOS',  'COND_ASSIST_FERIDOS', 'TESTEMUNHAS', 'RESP_SEGURADO',
 'ID_NOTIFICADOR', 'ID_FORMATOPART', 'ID_CAUSASIN', 'HOUVE_ACIDENTE', 'SISTEMA_SUGERIU',
 'CHOQUE_CADEIA', 'PREMIUM_RECEIPT_STATUS', 'BONUS_MALUS_SUGERIDO', 
  'CONTROL_QUAL', 'ID_NATUREZAAV_Averiguacao', 'INCIDENTRESULT_DESC', 'ID_ESTADO_GIF',
 'ID_CODPOSTAL_GIF',  'PROCESSTYPE_DESC', 'EMBATE_VIATURAS',
 'INTER_AUTORIDADE','ID_LOCALNOTIFICACAO', 'ID_EQUIPAUTIL',
 'ID_TIPOSINISTRO', 'ID_TPR',   'QUARTA_DIRECTIVA', 'ENQUADRAMENTO',
 'FACTORES', 'MARCA', 'MIGRADO_SINAUTO',  'flag_Gep','Retratação',
 'PROCESSSTAGE_DESC', 'LINEOFBUSINESS_DESC','SITUACAO_SINISTRO', 'CONSEQUENCIA',
 'SEG_VIATURA_ACIDENTE', 'AUTORIDADE', 'ENQUAD_TPR', 'ID_PARTFORMAL', 'ID_SEGURADORA', 'ID_MOTIVOATRIB',  
 'CONDUTOR_HABITUAL', 'TPR_VIATURA', 'RESSEGURO_PLATAFORMA',
 'AUTO_ALLOC_OFF', 'REEMBOLSO',  'IDS_REEM_CRIADO',  'ID_MOT_ENQUADRAMENTO', 'ID_TIPO_SIN',
 'VALOR_RECOND',
 'COUNTER_DAAA','DESPORTIVO','ID_COORDENADOR_GIF', 'fraude']


df_aveg[col_categ]=df_aveg[col_categ].replace(['?','-'], ' ')


# tudo a variaveis categóricas
for col in col_names:
    df_aveg[col] = df_aveg[col].astype('str',copy=True)
    df_aveg[col] = df_aveg[col].replace(['.'], np.nan)


#datas
for col in col_dates:
    df_aveg[col] = df_aveg[col].astype('str',copy=True).str[:10]
    print(col)

for col in col_dates:
        df_aveg[col]=pd.to_datetime(df_aveg[col], format='%Y-%m-%d', utc=False)
        print(col)



df_aveg['mes_DATA_SINISTRO']        = df_aveg['DATA_SINISTRO'].dt.month
df_aveg['diaSemana_DATA_SINISTRO']  = df_aveg['DATA_SINISTRO'].dt.dayofweek
df_aveg['quarter_DATA_SINISTRO']    = df_aveg['DATA_SINISTRO'].dt.quarter
df_aveg['semanaAno_DATA_SINISTRO']  = df_aveg['DATA_SINISTRO'].dt.week

df_aveg['mes_DATA_averig']        = df_aveg['DATA_ALTERACAO_data_Averiguacao'].dt.month
df_aveg['diaSemana_DATA_averig']  = df_aveg['DATA_ALTERACAO_data_Averiguacao'].dt.dayofweek
df_aveg['quarter_DATA_averig']    = df_aveg['DATA_ALTERACAO_data_Averiguacao'].dt.quarter
df_aveg['semanaAno_DATA_averig']  = df_aveg['DATA_ALTERACAO_data_Averiguacao'].dt.week


df_aveg['dias_ate_participacao']            = (df_aveg['DATA_SINISTRO'] - df_aveg['DATA_PARTICIPACAO']).dt.days*(-1)
df_aveg['dias_ate_respo_analisada']         = (df_aveg['DATA_SINISTRO'] - df_aveg['DATE_RESP_ANALISADA']).dt.days*(-1)
df_aveg['dias_entrada_companhia']           = (df_aveg['DATA_SINISTRO'] - df_aveg['DATA_ENTRADA_COMPANHIA']).dt.days*(-1)
df_aveg['dt_tratamento']                    = (df_aveg['DATA_ABERTURA'] - df_aveg['DATA_ENCERRAMENTO']).dt.days*(-1)
df_aveg['dias_ate_reclamacao']              = (df_aveg['DATA_SINISTRO'] - df_aveg['DATA_RECLAMACAO']).dt.days*(-1)
df_aveg['dias_ate_decisao_arb']             = (df_aveg['DATA_SINISTRO'] - df_aveg['DATA_DECISAO_ARB']).dt.days*(-1)
df_aveg['dias_ate_IDS']                     = (df_aveg['DATA_SINISTRO'] - df_aveg['IDS_DATA_ENVIO']).dt.days*(-1)
df_aveg['dias_ate_Averig']                  = (df_aveg['DATA_SINISTRO'] - df_aveg['DATA_ALTERACAO_data_Averiguacao']).dt.days*(-1)



###   Modelo que avalia sucesso do pedido de averiguações
##    Tratamento Dados para decision tree e avergiaguações


df_aveg['fraude'].replace({'Not_fraude':0,'Fraude':1},inplace=True)

a = df_aveg[['ID_MOTAVERIG_Averiguacao','fraude']]

b = pd.get_dummies(df_aveg['ID_MOTAVERIG_Averiguacao'])

c = pd.concat([a ,b], axis=1)
c.drop('ID_MOTAVERIG_Averiguacao', axis=1, inplace = True)

c['fraude'].value_counts()

# dataframe teste e treino

train, test = train_test_split(c, train_size = 0.8)

clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit( train.iloc[ :,1:], train['fraude'])


# Cross Validation


scores = cross_val_score(clf1, c.iloc[:,1:], c['fraude'] , cv=5)
print(scores)

y_pred1 = clf1.predict(test.iloc[ :,1:])




###
test['pred1']=y_pred1
labels1 = ['real1', 'pred1']

cm1 = confusion_matrix(test['fraude'], test['pred1'])
sns.heatmap(cm1,  annot=True, fmt="g", cmap='viridis',xticklabels=True,yticklabels=True)
plt.xlabel('real')
plt.ylabel('predicted')
plt.show()

clf1.feature_importances_

for name, importance in zip(test.iloc[:,1:].columns, clf1.feature_importances_):
    plt.bar(name, importance)



# ROC Curve
#
#probs = clf1.predict_proba(test.iloc[ :,1:])
#preds = probs[:,1]
#fpr, tpr, threshold = metrics.roc_curve(test['fraude'], preds)
#roc_auc = metrics.auc(fpr, tpr)

#plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()

from sklearn.externals.six import StringIO
import pydot

#from igraph import Graph


dotfile = open("D:\\Users\\f004197\\mytree.dot", 'w')
dot_data = StringIO()
tree.export_graphviz(clf1, out_file = dot_data, feature_names = c.iloc[:,1:].columns)


##    USEI ESTE SITE PARA GERAR GRÁFICO! http://webgraphviz.com/


#
clf1.feature_importances_

for name, importance in zip(test.iloc[:,1:].columns, clf1.feature_importances_):
    plt.bar(name, importance)




#
#
#
####
## K -Means
#
#


from sklearn import metrics
from scipy.spatial.distance import cdist, pdist


K = range(1,21)
KM = [KMeans(n_clusters=k).fit(c) for k in K]
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(c, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/c.shape[0] for d in dist]

# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(c)**2)/c.shape[0]
bss = tss-wcss


fig = plt.figure()
ax = fig.add_subplot(111)
axes = plt.gca()
axes.set_xlim([0,20])
axes.set_ylim([0,100])
ax.plot(K, bss/tss*100, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')



####### Retirando Fraude 
##'1003', '1004', '1007', '1008', '1009', '701', '702', '703', '704', '705', '707', '708', '709', '710', '711', '712', 
## '713', '714', '715', '716', '717', '718', '719', '720', '722', '723', '724', '727', '728', '730', '731', '732', '733', 
## '734', '735', '736', '738', '741', '742'

df_clusters_averigs = c[['fraude','1003', '1004', '1007', '1008', '1009', '701', '702', '703', '704', '705', '707', '708', '709', '710', '711', 
'712', '713', '714', '715', '716', '717', '718', '719', '720', '722', '723', '724', '727', '728', '730', '731', '732', '733', '734', 
'735', '736', '738', '741', '742']]



K = range(1,21)
KM = [KMeans(n_clusters=k).fit(df_clusters_averigs.iloc[:,1:]) for k in K]
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(df_clusters_averigs.iloc[:,1:], cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/df_clusters_averigs.iloc[:,1:].shape[0] for d in dist]



# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(df_clusters_averigs.iloc[:,1:])**2)/df_clusters_averigs.iloc[:,1:].shape[0]
bss = tss-wcss

fig = plt.figure()
ax = fig.add_subplot(111)
axes = plt.gca()
axes.set_xlim([0,20])
axes.set_ylim([0,100])
ax.plot(K, bss/tss*100, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')
plt.show()



from sklearn.metrics import silhouette_score

s = []
for n_clusters in range(2,21):
    kmeansS = KMeans(n_clusters=n_clusters)
    kmeansS.fit(df_clusters_averigs.iloc[:,1:])
    labelsS = kmeansS.labels_
    centroidsS = kmeansS.cluster_centers_
    s.append(silhouette_score(df_clusters_averigs.iloc[:,1:], labelsS, metric='euclidean'))

plt.plot(s)
plt.ylabel("Silhouette")
plt.xlabel("Number of Clusters")
plt.title("Silhouette for K-means cell's behaviour")
sns.despine()




## Melhor aproximação às averiguações: 7 clusters

num_clusters = 7

kmeans = KMeans(n_clusters=num_clusters, max_iter=1000).fit(df_clusters_averigs.iloc[:,1:])

df_clusters_averigs["clusters"] = kmeans.labels_
centers = [kmeans.cluster_centers_]


series = df_clusters_averigs['clusters'].value_counts()
colors = cm.viridis(np.linspace(0, 1, 9))
series.plot(kind='bar', title='Numero Averiguações Por Cluster', color = colors)

################################################### Para representar clusters

from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit(df_clusters_averigs.iloc[:,1:])
ver_pca = pca.components_


pca_d = pd.DataFrame(pca.transform(df_clusters_averigs.iloc[:,1:]), columns=['PCA%i' % i for i in range(2)], index=d.index)
scat= plt.scatter(pca_d['PCA0'],pca_d['PCA1'], c =kmeans.labels_ )
bounds = np.linspace(0,9,9+1)
cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
cb.set_label('Custom cbar')
plt.show()

############################################################################


#REPRESENTAçÃO GRÀFICA 3 PCA do SET de Dados com base nos clusters identifcados

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

pca = PCA(n_components=3).fit(df_clusters_averigs.iloc[:,1:])
ver_pca = pca.components_

fig = plt.figure()
ax = Axes3D(fig)


pca_d = pd.DataFrame(pca.transform(df_clusters_averigs.iloc[:,1:]), columns=['PCA%i' % i for i in range(3)], index=df_clusters_averigs.iloc[:,1:].index)
ax.scatter(pca_d['PCA0'],pca_d['PCA1'],pca_d['PCA2'], c =kmeans.labels_ )
bounds = np.linspace(-10,100,110+1)
cb = plt.colorbar(scat, spacing='proportional',ticks=bounds,shrink=0.8 )
cb.set_label('Clusters')
plt.show()

# Dump components relations with features:PENDENTE - AINDA NÃO PERCEBI

pc_matriz_components = pd.DataFrame(pca.components_,columns=d.columns,index = ['PCA0','PCA1', 'PCA2'])



from pandas.plotting import scatter_matrix
scatter_matrix(pc_matriz_components, alpha=0.2, figsize=(6, 6), diagonal='kde')



###############################################################################


## Refazer Arvores de decisão mas agora só com clusters

# dataframe teste e treino



e =pd.get_dummies(df_clusters_averigs['clusters'])

e = pd.concat([e ,df_clusters_averigs], axis=1)

c.drop('clusters', axis=1, inplace = True)

train, test = train_test_split(c[['fraude',0, 1, 2, 3, 4, 5, 6, 7, 8]], train_size = 0.8)


freq_clusters_aveg = df_clusters_averigs.groupby(['clusters']).sum()

freq_clusters_aveg.reset_index(inplace= True)

plt.scatter(freq_clusters_aveg[['clusters']],freq_clusters_aveg[['1003', '1004', '1007', '1008', '1009', '701', '702', '703', '704', '705', '707', '708', '709', '710', '711', 
'712', '713', '714', '715', '716', '717', '718', '719', '720', '722', '723', '724', '727', '728', '730', '731', '732', '733', '734', 
'735', '736', '738', '741', '742']])

    
  
teste1 = pd.crosstab(df_clusters_averigs['clusters'],df_clusters_averigs['fraude'])     




#### Averiguação Por Cluster

   

df_clusters_averigs = df_clusters_averigs.set_index(['fraude','clusters'])
df_clusters_averigs.columns = df_clusters_averigs.columns.str.extract('(\d+)', expand=False)
df_clusters_averigs = df_clusters_averigs.stack().reset_index(name='count_averigs').rename(columns={'level_2':'Motivos_Averiguacao'})


cluster_de_cada_averiguacao = df_clusters_averigs.groupby(['clusters','Motivos_Averiguacao'])['count_averigs'].sum()
cluster_de_cada_averiguacao = cluster_de_cada_averiguacao.reset_index()


cluster_de_cada_averiguacao = cluster_de_cada_averiguacao.loc[cluster_de_cada_averiguacao['count_averigs']!=0.0]   




####

 
clfC = tree.DecisionTreeClassifier()
clfC = clfC.fit( train.iloc[:,1:], train['fraude'])


# Cross Validation


scores = cross_val_score(clfC, c.iloc[:,1:], c['fraude'] , cv=5)
print(scores)

y_pred1 = clfC.predict(test.iloc[ :,1:])





#####################################Quando ainda tinha 9 clusters
test['pred1']=y_pred1
labels1 = ['real1', 'pred1']


fig = plt.figure()
cmC = confusion_matrix(test['fraude'], test['pred1'])
sns.heatmap(cmC,  annot=True, fmt="g", cmap='viridis',xticklabels=True,yticklabels=True)
plt.xlabel('real')
plt.ylabel('predicted')
plt.show()




clfC.feature_importances_

for name, importance in zip(test.iloc[:,1:].columns, clfC.feature_importances_):
   print(name, importance)
    


 # ROC Curve
clfP  = clfC.fit( train[[0, 1, 2, 3, 4, 5, 6, 7, 8]], train['fraude'])
probs = clfP.predict_proba(test[[0, 1, 2, 3, 4, 5, 6, 7, 8]])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(test['fraude'], preds)
roc_auc = metrics.auc(fpr, tpr)



plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



#############################

# df_clusters_averigs  -> tabela com os clusters de cada averiguação


ver = df_aveg[col_dates]

df_feat = df_aveg[[]]

c = pd.concat([a ,b], axis=1)



df_aveg['dias_ate_participacao'] = (df_aveg['DATA_SINISTRO'] - df_aveg['DATA_PARTICIPACAO']).dt.days
df_aveg['dias_ate_respo_analisada'] = (df_aveg['DATA_SINISTRO'] - df_aveg['DATE_RESP_ANALISADA']).dt.days
df_aveg['dias_entrada_companhia'] = (df_aveg['DATA_SINISTRO'] - df_aveg['DATA_ENTRADA_COMPANHIA']).dt.days


col_dates_diff = ['dias_ate_participacao','dias_ate_respo_analisada','dias_entrada_companhia'] 






   

col =  df_aveg.columns.values 
df_aveg_describe = df_aveg[col].value_counts().reset_index()
df_aveg_describe.columns = df_aveg.columns


   
df[column].value_counts()

# get indexes
colunas = df[column].value_counts().index.tolist()

# get values
valores = df[column].value_counts().values.tolist()




df_aveg['mes_DATA_SINISTRO']        = df_aveg['DATA_SINISTRO'].dt.month
df_aveg['diaSemana_DATA_SINISTRO']  = df_aveg['DATA_SINISTRO'].dt.dayofweek
df_aveg['quarter_DATA_SINISTRO']    = df_aveg['DATA_SINISTRO'].dt.quarter
df_aveg['semanaAno_DATA_SINISTRO']  = df_aveg['DATA_SINISTRO'].dt.week


df_aveg['HORA_SINISTRO_2']=df_aveg['HORA_SINISTRO'].astype(str).str[0:5].apply(lambda x: float(x.replace(":",".")))



df_aveg['hr_sin']   = np.sin(df_aveg['HORA_SINISTRO_2']*(2.*np.pi/24))
df_aveg['hr_cos']   = np.cos(df_aveg['HORA_SINISTRO_2']*(2.*np.pi/24))
df_aveg['mnth_sin'] = np.sin((df_aveg['mes_DATA_SINISTRO']-1)*(2.*np.pi/12))
df_aveg['mnth_cos'] = np.cos((df_aveg['mes_DATA_SINISTRO']-1)*(2.*np.pi/12))

df_aveg['quarter_sin']   = np.sin(df_aveg['quarter_DATA_SINISTRO']*(2.*np.pi/4))
df_aveg['quarter_cos']   = np.cos(df_aveg['quarter_DATA_SINISTRO']*(2.*np.pi/4))
df_aveg['semanaAno_sin'] = np.sin(df_aveg['semanaAno_DATA_SINISTRO']*(2.*np.pi/187))
df_aveg['semanaAno_cos'] = np.cos(df_aveg['semanaAno_DATA_SINISTRO']*(2.*np.pi/187))



'hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos','quarter_sin', 'quarter_cos', 'semanaAno_sin', 'semanaAno_cos'

#regiao -1 digito codigo postal

df_aveg['CodigoPostal_GIF_2'] = df_aveg['ID_CODPOSTAL_GIF'].str[0:2]

cols = [
 'ID_AVERIGUADOR_GIF', 'INCIDENTSOURCE_DESC', 'REGUL_GERAL_SIN',
 'MOTOCICLOS',  'COND_ASSIST_FERIDOS', 'TESTEMUNHAS', 'RESP_SEGURADO',
 'ID_NOTIFICADOR', 'ID_FORMATOPART', 'ID_CAUSASIN', 'HOUVE_ACIDENTE', 'SISTEMA_SUGERIU',
 'CHOQUE_CADEIA', 'PREMIUM_RECEIPT_STATUS', 'BONUS_MALUS_SUGERIDO', 
 'INSURED_PROTECTION', 'CONTROL_QUAL', 'ID_NATUREZAAV_Averiguacao',
 'ID_CODPOSTAL_GIF',  'PROCESSTYPE_DESC', 'EMBATE_VIATURAS',
 'INTER_AUTORIDADE','ID_LOCALNOTIFICACAO', 'ID_EQUIPAUTIL',
 'ID_TIPOSINISTRO', 'ID_TPR',   'QUARTA_DIRECTIVA', 'ENQUADRAMENTO',
 'FACTORES', 'MARCA', 'MIGRADO_SINAUTO',  'flag_Gep','Retratação',
 'PROCESSSTAGE_DESC', 'LINEOFBUSINESS_DESC','SITUACAO_SINISTRO', 'CONSEQUENCIA',
 'SEG_VIATURA_ACIDENTE', 'AUTORIDADE', 'ENQUAD_TPR', 'ID_PARTFORMAL', 'ID_SEGURADORA',
 'ID_CONCELHO', 'ID_MOTIVOATRIB',  'CONDUTOR_HABITUAL', 'TPR_VIATURA', 'RESSEGURO_PLATAFORMA',
 'AUTO_ALLOC_OFF', 'REEMBOLSO',  'IDS_REEM_CRIADO',  'ID_MOT_ENQUADRAMENTO', 'ID_TIPO_SIN',
 'VALOR_RECOND','GIF',
 'COUNTER_DAAA','PROC_ESPECIAL','DESPORTIVO','ID_COORDENADOR_GIF', 'HORA_SINISTRO',
 'mes_DATA_SINISTRO','diaSemana_DATA_SINISTRO','quarter_DATA_SINISTRO',
 'semanaAno_DATA_SINISTRO','dias_ate_participacao','dias_ate_respo_analisada','dias_entrada_companhia',
 'cluster',
 'fraude'
 ]


cols = [
 'ID_AVERIGUADOR_GIF', 'ID_CAUSASIN','ID_LOCALNOTIFICACAO', 'ID_EQUIPAUTIL',
 'COUNTER_DAAA', 'HORA_SINISTRO','ID_CODPOSTAL_GIF',
 'mes_DATA_SINISTRO','diaSemana_DATA_SINISTRO','quarter_DATA_SINISTRO',
 'semanaAno_DATA_SINISTRO','dias_ate_participacao','dias_ate_respo_analisada','dias_entrada_companhia',
 'cluster',
 'fraude'
 ]


# 'DESC_ACIDENTE',
# 'INCIDENTSUBTYPE_DESC',

def func(row):
    if   row['ID_MOTAVERIG_Averiguacao'] =='704':
        return 1
    elif row['ID_MOTAVERIG_Averiguacao'] =='711':
        return 2
    elif row['ID_MOTAVERIG_Averiguacao'] =='720':
        return 3 
    elif row['ID_MOTAVERIG_Averiguacao'] =='703':
        return 4 
    elif row['ID_MOTAVERIG_Averiguacao'] =='702':
        return 5 
    elif row['ID_MOTAVERIG_Averiguacao'] =='736':
        return 6
    else:
        return 0


df_aveg['cluster'] = df_aveg.apply(func, axis=1)


df_corte = df_aveg[cols]


#df_corte.drop('CONTENCIOSO', axis=1, inplace=True)

df_corte_bin = pd.get_dummies(df_corte.drop(['ID_CODPOSTAL_GIF','HORA_SINISTRO',
'mes_DATA_SINISTRO','diaSemana_DATA_SINISTRO','quarter_DATA_SINISTRO',
'semanaAno_DATA_SINISTRO','dias_ate_participacao','dias_ate_respo_analisada','dias_entrada_companhia'], axis= 1))


df_corte_bin.fillna(0, inplace=True)


############## DATAFRAME QUE VAI CLASSIFICAR SINISTROS COM FRAUDE COM 91 FEAUTURES 


train, test = train_test_split(df_corte_bin, train_size = 0.8)


# Random Tree Classifier

clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit(train.drop('fraude', axis=1), train['fraude'])

y_pred1 = clf1.predict(test.drop('fraude', axis=1))


# Cross Validation


scores = cross_val_score(clf1, c.iloc[:,1:], c['fraude'] , cv=5)
print(scores)

###
test['pred1']=y_pred1
labels1 = ['real1', 'pred1']

cm1 = confusion_matrix(test['fraude'], test['pred1'])
sns.heatmap(cm1,  annot=True, fmt="g", cmap='viridis',xticklabels=True,yticklabels=True)
plt.xlabel('real')
plt.ylabel('predicted')
plt.show()

clf1.feature_importances_

# zip itera tuples na lista
for name, importance in zip(test.drop('fraude', axis=1), clf1.feature_importances_)[::-1][:15]:
    plt.bar(name, importance)
#    scatter_matrix(name,importance)
#    plt.show()


def selectKImportance(model, X, k=10):
     return X.iloc[:,model.feature_importances_.argsort()[::-1][:k]]





Feat_Importantes = selectKImportance(clf1, test,k=15)


scatter_matrix(Feat_Importantes)
plt.show()


import seaborn as sns

def plot_corr(df):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
   # fig, ax = plt.subplots(figsize=(size, size))
   # ax.matshow(corr)
    sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values, cmap=sns.diverging_palette(220, 10, as_cmap=True)) 



plot_corr(Feat_Importantes)


Feat_Importantes.columns

#######
#
# Fazer um teste com clusters aos mesmos dados classificados no RandomTreeClassifier
#
#######

K = range(1,21)
KM = [KMeans(n_clusters=k).fit(df_corte_bin) for k in K]
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(df_corte_bin, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/df_corte_bin.shape[0] for d in dist]



# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(df_corte_bin)**2)/df_corte_bin.shape[0]
bss = tss-wcss

fig = plt.figure()
ax = fig.add_subplot(111)
axes = plt.gca()
axes.set_xlim([0,20])
axes.set_ylim([0,100])
ax.plot(K, bss/tss*100, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
plt.title('Elbow for KMeans clustering')
plt.show()



from sklearn.metrics import silhouette_score

s = []
for n_clusters in range(2,21):
    kmeansS = KMeans(n_clusters=n_clusters)
    kmeansS.fit(df_corte_bin.iloc[:,1:])
    labelsS = kmeansS.labels_
    centroidsS = kmeansS.cluster_centers_
    s.append(silhouette_score(df_corte_bin.iloc[:,1:], labelsS, metric='euclidean'))

plt.plot(s)
plt.ylabel("Silhouette")
plt.xlabel("Number of Clusters")
plt.title("Silhouette for K-means cell's behaviour")
sns.despine()




