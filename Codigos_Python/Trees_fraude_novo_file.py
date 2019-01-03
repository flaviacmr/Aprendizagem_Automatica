# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:48:31 2018

@author: f004197
"""

# Diminuicao sem grande preocupacao do volume de dados

df_tree =df_processed.sample(frac=0.02)

pca = PCA(n_components=30).fit(df_tree)
ver_pca = pca.components_


# cross validation
from sklearn.model_selection import cross_val_score

# Modelos: decisionTreeclassifier

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix

from sklearn.tree.export import export_graphviz
from sklearn.feature_selection import mutual_info_classif


columns_to_encode = [
'ACCAUSA', 'APCONC', 'CD_COND_GERAIS', 'COBRTIP', 'CONDSEXO', 'CORPORAC', 'Cobertura1', 'CoberturaNaoUsada1', 'DANOSMC', 'DEFRESP', 
'Dano_viatura1', 'FORMENTP', 'ORIGPART', 'ObjetRegul1', 'TIPCONT', 'TIPFRACC', 'TIPOSIN', 'TPRCASO', 'VIATMARC', 'indexclu', 'produto']


encoded_columnsF.hist()

columns_to_encode2 = encoded_columnsF.columns

for i in columns_to_encode2:
    sns.countplot(x=i, hue="Fraude_Final",data=encoded_columnsF, palette="coolwarm")



encoded_columns = pd.get_dummies(df_m1[columns_to_encode])

#encoded_columns =  ohe.fit_transform(df_m1[columns_to_encode])

encoded_columnsF = pd.concat([df_m1['Fraude_Final'], encoded_columns], axis=1 )



# dataframe teste e treino


train, test = train_test_split(encoded_columnsF, train_size = 0.8)

clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit( train.iloc[ :,1:], train['Fraude_Final'])


# Cross Validation


scores = cross_val_score(clf1, train.loc[:, train.columns != 'Fraude_Final'], train['Fraude_Final'] , cv=5)
print(scores)

# print(scores) = [0.98952013 0.9812362  0.98730684 0.98565121 0.98565121]

y_pred1 = clf1.predict(test.loc[:, test.columns != 'Fraude_Final'])


from matplotlib import pyplot as plt
plt.style.use('seaborn-ticks')
###
test['pred1']=y_pred1
labels1 = ['real1', 'pred1']

cm1 = confusion_matrix(test['Fraude_Final'], test['pred1'])
sns.heatmap(cm1,  annot=True, fmt="g", cmap='viridis',xticklabels=True,yticklabels=True)
plt.xlabel('real')
plt.ylabel('predicted')
plt.show()


importantes  = clf1.feature_importances_


for name, importance in zip(test.loc[:, test.columns != 'Fraude_Final'].columns, np.argsort(importantes[::-1])[:20]):
    #plt.bar(name, importance)
     a = print(name, importance)
     

plot_importance(clf1)
pyplot.show()   
    
train, test = train_test_split(df_m1, train_size = 0.8)    
 ## Import the random forest model.
from sklearn.ensemble import RandomForestClassifier 
## This line instantiates the model. 
rf = RandomForestClassifier() 
## Fit the model on your training data.
rf.fit(train.loc[:, train.columns != 'Fraude_Final'], train['Fraude_Final'] ) 
## And score it on your testing data.
rf.score(test.loc[:, test.columns != 'Fraude_Final'], test['Fraude_Final'])   