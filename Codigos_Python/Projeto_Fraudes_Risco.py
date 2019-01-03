# -*- coding: utf-8 -*-
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

df_modelo = pd.read_excel('D:\\Users\\f004197\\Desktop\\Averiguacoes_Final\\Modelo_Fraudes.xlsx')

df_m1 = df_modelo.drop(['PROCESSO', 'APOLICE', 'DTSINIST','DTINICIO', 'DTTERMO', 'DTANUL','hrsinist', 'CONDANON', 
                        'CONDANON', 'CONDANOC', 'ACCONC', 'medprod'], axis=1)

# esta variavel tem muitos valores unicos 'medprod',

import seaborn as sns
corr = df_m1.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()



# Import libraries and download example data
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define which columns should be encoded vs scaled
columns_to_encode =['tipoclub', 'indexclu', 'CD_COND_GERAIS', 'COBRTIP', 'TIPFRACC', 'TIPCONT', 'CORPORAC', 
 'produto', 'ACCAUSA', 'TIPOSIN', 'DANOSMC', 'TPRCASO', 'ACEGS', 'PAIS', 'CONDSEXO', 'VIATMARC', 'VIATTIPO', 'ORIGPART']


columns_to_scale  = ['Tempo_Apolice_Ate_Sinistro', 'Duracao_apolice', 'sin_mes_apolice', 'cos_mes_apolice', 
 'sin_ano_apolice', 'cos_ano_apolice', 'COBRADOR', 'PREMTTAN', 'CAPITAL_APOL', 'TP_CRC_SINIST', 'Sin_Hr_Sinistro', 'Cos_Hr_Sinistro', 
 'Sinistro_semana_Ano', 'Sin_SemanaAno', 'Cos_SemanaAno', 'Sinistro_dia_semana', 'Sin_DiaSemana', 'Cos_DiaSemana', 'Sinistro_quarter', 
 'Cos_Quarter', 'Sin_Quarter', 'Sinistro_Mes', 'Cos_Mes', 'Sin_Mes', 'Sinistro_Ano', 'Cos_Ano', 'Sin_Ano', 'DEFRESP', 'dias_ate_abrir_proc', 
 'APCONC', 'CTOTPRCS', 'PINIPRCS', 'Diff_CustoProv', 
  'EQUIPA', 'FORMENTP', 'ORIGPART', 'idade_condutor', 'dias_com_carta', 'VIATANO', 'VIATCIL']



# Instantiate encoder/scaler
scaler = StandardScaler()
ohe    = OneHotEncoder(sparse=False)

# Scale and Encode Separate Columns
df_m1[columns_to_scale] = df_m1[columns_to_scale].fillna(0)
scaled_array = scaler.fit_transform(df_m1[columns_to_scale])
scaled_columns  = pd.DataFrame(scaled_array, index = df_m1[columns_to_scale].index, columns=columns_to_scale) 


encoded_columns = pd.get_dummies(df_m1[columns_to_encode])

#encoded_columns =  ohe.fit_transform(df_m1[columns_to_encode])

encoded_columnsF = pd.concat([df_m1['Fraude_Final'], encoded_columns], axis=1 )


# Concatenate (Column-Bind) Processed Columns Back Together

#processed_data = np.concatenate((scaled_columns,encoded_columnsF), axis=1)


#convert to pandas dataframe
df_processed = pd.concat([scaled_columns,encoded_columnsF], axis=1)



# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split

# Specify the data 


X = df_processed.loc[:, df_processed.columns != 'Fraude_Final']

# Specify the target labels and flatten the array 
Y=df_processed['Fraude_Final']

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)



from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model,Sequential

inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)



# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(1301, activation='relu', input_shape=(1301,)))

# Add one hidden layer 
model.add(Dense(2, activation='relu'))

# Add one hidden layer 
#model.add(Dense(50, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))

# Model output shape
model.output_shape

# Model summary
model.summary()

# Model config
model.get_config()

# List all weight tensors 
model.get_weights()

#Compile and Fit

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   

model.fit(X_train, y_train,epochs=20, batch_size=1, verbose=1)



y_pred = model.predict(X_test)


score = model.evaluate(X_test, y_test,verbose=1)

print(score)


# Import the modules from `sklearn.metrics`
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

# Confusion matrix
confusion_matrix(y_test, y_pred)



# Precision 
precision_score(y_test, y_pred)



### Análise a Variáveis Que Vão Alimentar Modelo

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

#Feature Selection

testColumns = df_m1[columns_to_encode].columns
testColumns2 = ['ACCAUSA', 'APCONC', 'CD_COND_GERAIS', 'COBRTIP', 'CONDSEXO', 'CORPORAC', 'Cobertura1', 'CoberturaNaoUsada1', 
'DANOSMC', 'DEFRESP', 'Dano_viatura1', 'FORMENTP', 'ORIGPART', 'ObjetRegul1', 'TIPCONT', 
'TIPFRACC', 'TIPOSIN', 'TPRCASO', 'VIATMARC', 'flag_autoridade', 'flag_oficina_nossa', 
'flag_testemunhas', 'indexclu', 'produto','Fraude_Final']
testColumns3 = ['ACCAUSA', 'APCONC', 'CD_COND_GERAIS', 'COBRTIP', 'CONDSEXO', 'CORPORAC', 'Cobertura1', 'CoberturaNaoUsada1', 
'DANOSMC', 'DEFRESP', 'Dano_viatura1', 'FORMENTP', 'ORIGPART', 'ObjetRegul1', 'TIPCONT', 
'TIPFRACC', 'TIPOSIN', 'TPRCASO', 'VIATMARC', 'flag_autoridade', 'flag_oficina_nossa', 
'flag_testemunhas', 'indexclu', 'produto']

#Initialize ChiSquare Class

cT = ChiSquare(df_m1[testColumns2])



for var in testColumns3:
    cT.TestIndependence(colX=var,colY="Fraude_Final")



chi2, p, dof, expected = stats.chi2_contingency(df_m1)



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



v_cramer_corrected = cramers_corrected_stat(df_m1[columns_to_encode]) #0.29784802849576003



def plot_corr(df,size=250):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
    
plot_corr(df_m1[columns_to_encode],250)    
    
# Detecção de Outliers

def Quartiles(df):
    """ 
        Determine the Quartiles
    """
   
    for v in testColumns:
            plt.boxplot(df[v].balance)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)
    print(df[v] < (Q1 - 1.5 * IQR)) |(df[v] > (Q3 + 1.5 * IQR))
 




#Get relevant percentiles and see their distribution
df['balance'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])




#def FilterDataBasedStats(method, df):
#    
#    if method == 'Quartile':
#    df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
#    elif method == 'ChiSquare':
#    df_out = df[~df[p<alpha]]











