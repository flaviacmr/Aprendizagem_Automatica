# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 16:08:36 2018

@author: f004197
"""
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

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

from sklearn.model_selection import train_test_split

#cluster Averiguações

df_aveg['fraude'].replace({'Not_fraude':0,'Fraude':1},inplace=True)
a= df_aveg[['ID_MOTAVERIG_Averiguacao','fraude']] 
b = pd.get_dummies(df_aveg['ID_MOTAVERIG_Averiguacao'])
c = pd.concat([a ,b], axis=1)
c.drop('ID_MOTAVERIG_Averiguacao', axis=1, inplace = True)
# fazer isto
train, test = train_test_split(c, train_size = 0.8)

clf1 = tree.DecisionTreeClassifier(  )
clf1 = clf1.fit( train.iloc[ :,1:], train['fraude']) 

# CV
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf1, c.iloc[:,1:], c['fraude'] , cv=5)


y_pred1 = clf1.predict(test.iloc[ :,1:])




###
test['pred1']=y_pred1
labels1 = ['real1', 'pred1']

cm1 = confusion_matrix(test['fraude'], test['pred1'])
sns.heatmap(cm1,  annot=True, fmt="g", cmap='viridis',xticklabels=True,yticklabels=True)
plt.xlabel('real')
plt.ylabel('predicted')
plt.show()

# ROC Curve

probs = clf1.predict_proba(test.iloc[ :,1:])
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(test['fraude'], preds)
roc_auc = metrics.auc(fpr, tpr)


#
#print(tree.DecisionTreeClassifier().get_params())
#
#{'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None,
# 'min_impurity_decrease': 0.0, 'min_impurity_split': None,
# 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'presort': False, 'random_state': None, 'splitter': 'best'}

# method I: plt

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


clf1.feature_importances_

from sklearn.tree.export import export_graphviz
from sklearn.feature_selection import mutual_info_classif
#import graphviz


export_graphviz(tree, out_file="mytree.dot")
with open("mytree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
#Clusters


for name, importance in zip(test.iloc[:,1:].columns, clf1.feature_importances_):
    print(name, importance)


import graphviz as gv
# uncommenting the row above produces an error
clf = tree.DecisionTreeClassifier()
iris = load_iris()
clf = clf.fit(iris.data, iris.target)
with open('graph.dot', 'w') as file:
    tree.export_graphviz(clf, out_file = file)
file.close()


