# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:33:33 2018

@author: f004197
"""

# -*- coding: utf-8 -*-


import seaborn as sns
corr = df_proc2.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, cmap="RdGy")
plt.show()





# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split

# Specify the data 


train, test = train_test_split(df_processed, train_size = 0.8)



from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix



clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit( train.loc[:, train.columns != 'Fraude_Final'], train['Fraude_Final'])


# Cross Validation
from sklearn.model_selection import cross_val_score

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


clf1.feature_importances_

# zip itera tuples na lista
for name, importance in zip(test.drop('Fraude_Final', axis=1), clf1.feature_importances_):
#    plt.bar(name, importance)
#    scatter_matrix(name,importance)
     print(name, importance)


def selectKImportance(model, X, k=20):
     return X.iloc[:,model.feature_importances_.argsort()[::-1][:k]]


selectKImportance(clf1,test)

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(test['Fraude_Final'], y_pred1)


plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()


############

train, test = train_test_split(encoded_columnsF, train_size = 0.8)


clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit( train.loc[:, train.columns != 'Fraude_Final'], train['Fraude_Final'])


# Variaveis Categóricas - Cross Validation
from sklearn.model_selection import cross_val_score

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



import seaborn as sns

####

sns.pairplot(df_m1)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '5pt'})\
    .set_caption("Hover to magify")\
    .set_precision(2)\
    .set_table_styles(magnify())


corr = df_m1.corr()
sns.heatmap(corr, annot=True,fmt=".2f",
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, cmap="RdBu_r")
plt.show()

# GRAFICOS INTERESSANTES - VER DEPOIS
g = sns.FacetGrid(encoded_columnsF, row='Fraude_Final', col='flag_oficina_nossa')
g.map(sns.distplot, "flag_autoridade")
plt.show()



clf1.feature_importances_

# zip itera tuples na lista
for name, importance in zip(test.drop('Fraude_Final', axis=1), clf1.feature_importances_.argsort()[::-1][:50]):
#    plt.bar(name, importance)
#    scatter_matrix(name,importance)
    print(name, importance)