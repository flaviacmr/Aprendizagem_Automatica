# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:35:22 2018

@author: f004197


Variaveis finais - 

"""


col_dates = [
'RECEIVEDON_GIF',    
'CHANGEDON_GIF',    
'DATA_SINISTRO',    
'DATA_PARTICIPACAO',    'DATA_ABERTURA',    'DATA_ENCERRAMENTO',        'DATE_UPDATE',
'IDS_DATA_ENVIO',    'DATE_RESP_ANALISADA',    'DATA_PRE_ENCERRAMENTO',
'DATA_REEMBOLSO',    'DATA_ENTRADA_PLATAFORMA',    'DATA_ENTRADA_COMPANHIA', 'DATA_PEDIDO_PARTICIPACAO',    
'DATA_RECLAMACAO',    'DATA_DECISAO_FINAL',
'DATA_ASSUNCAO_RESP',    'DATA_INICIO_SUSPENDE',    'DATA_FIM_SUSPENDE',
'DATA_PROPOSTA_ARB',    'DATA_RECUSA_ARB',    'DATA_DECISAO_ARB', 'DATA_COMUNIC_SEG',
'DATA_MIGRACAO_SINAUTO','DATA_ALTERACAO_data_Averiguacao']





#datas
for col in col_dates:
    df_aveg[col] = df_aveg[col].astype('str',copy=True).str[:9]


for col in col_dates:
        df_aveg[col]=pd.to_datetime(df_aveg[col], format='%Y-%m-%d', utc=False)
        print(col)




df_aveg['dias_ate_participacao']            = (df_aveg['DATA_SINISTRO'] - df_aveg['DATA_PARTICIPACAO']).dt.days*(-1)
df_aveg['dias_ate_respo_analisada']         = (df_aveg['DATA_SINISTRO'] - df_aveg['DATE_RESP_ANALISADA']).dt.days*(-1)
df_aveg['dias_entrada_companhia']           = (df_aveg['DATA_SINISTRO'] - df_aveg['DATA_ENTRADA_COMPANHIA']).dt.days*(-1)
df_aveg['dt_tratamento']                    = (df_aveg['DATA_ABERTURA'] - df_aveg['DATA_ENCERRAMENTO']).dt.days*(-1)
df_aveg['dias_ate_reclamacao']              = (df_aveg['DATA_SINISTRO'] - df_aveg['DATA_RECLAMACAO']).dt.days*(-1)
df_aveg['dias_ate_decisao_arb']             = (df_aveg['DATA_SINISTRO'] - df_aveg['DATA_DECISAO_ARB']).dt.days*(-1)
df_aveg['dias_ate_IDS']                     = (df_aveg['DATA_SINISTRO'] - df_aveg['IDS_DATA_ENVIO']).dt.days*(-1)
df_aveg['dias_ate_Averig']                  = (df_aveg['DATA_SINISTRO'] - df_aveg['DATA_ALTERACAO_data_Averiguacao']).dt.days*(-1)




dt_plots = df_aveg[['dias_ate_participacao', 'dias_ate_respo_analisada', 'dias_entrada_companhia',  'dt_tratamento',
                    'dias_ate_reclamacao', 'dias_ate_decisao_arb', 'dias_ate_IDS','dias_ate_Averig']]



from datetime import date


dt_plots.hist(bins=100, range=(-10, 300))


df_aveg['HORA_SINISTRO_2']=df_aveg['HORA_SINISTRO'].astype(str).str[0:5].apply(lambda x: float(x.replace(":",".")))

df_aveg['DATA_ALTERACAO_data_Averiguacao'] = df_aveg['DATA_ALTERACAO_data_Averiguacao'].astype('str',copy=True).str[:10]
#df_aveg['Hora_ALTERACAO_data_Averiguacao'] = df_aveg['DATA_ALTERACAO_data_Averiguacao'].astype('str',copy=True).str[11:-3]





df_aveg['SINISTRO_hr_sin']   = np.sin(df_aveg['HORA_SINISTRO_2']*(2.*np.pi/24))
df_aveg['SINISTRO_hr_cos']   = np.cos(df_aveg['HORA_SINISTRO_2']*(2.*np.pi/24))
df_aveg['SINISTRO_mnth_sin'] = np.sin((df_aveg['mes_DATA_SINISTRO']-1)*(2.*np.pi/12))
df_aveg['SINISTRO_mnth_cos'] = np.cos((df_aveg['mes_DATA_SINISTRO']-1)*(2.*np.pi/12))

df_aveg['SINISTRO_quarter_sin']   = np.sin(df_aveg['quarter_DATA_SINISTRO']*(2.*np.pi/4))
df_aveg['SINISTRO_quarter_cos']   = np.cos(df_aveg['quarter_DATA_SINISTRO']*(2.*np.pi/4))
df_aveg['SINISTRO_semanaAno_sin'] = np.sin(df_aveg['semanaAno_DATA_SINISTRO']*(2.*np.pi/187))
df_aveg['SINISTRO_semanaAno_cos'] = np.cos(df_aveg['semanaAno_DATA_SINISTRO']*(2.*np.pi/187))




#df_aveg['AVERIG_hr_sin']   = np.sin(df_aveg['Hora_ALTERACAO_data_Averiguacao']*(2.*np.pi/24))
#df_aveg['AVERIG_hr_cos']   = np.cos(df_aveg['Hora_ALTERACAO_data_Averiguacao']*(2.*np.pi/24))
#

df_aveg['AVERIG_mnth_sin'] = np.sin((df_aveg['mes_DATA_averig']-1)*(2.*np.pi/12))
df_aveg['AVERIG_mnth_cos'] = np.cos((df_aveg['mes_DATA_averig']-1)*(2.*np.pi/12))
df_aveg['AVERIG_quarter_sin']   = np.sin(df_aveg['quarter_DATA_averig']*(2.*np.pi/4))
df_aveg['AVERIG_quarter_cos']   = np.cos(df_aveg['quarter_DATA_averig']*(2.*np.pi/4))
df_aveg['AVERIG_semanaAno_sin'] = np.sin(df_aveg['semanaAno_DATA_averig']*(2.*np.pi/187))
df_aveg['AVERIG_semanaAno_cos'] = np.cos(df_aveg['semanaAno_DATA_averig']*(2.*np.pi/187))


df_aveg['CodigoPostal_GIF_1'] = df_aveg['CodigoPostal_GIF_2'].str[0:1]


df_aveg['Retratação'] = df_aveg['Retratação'].replace('nan','0.0')

df_aveg['ID_PARTFORMAL'] = df_aveg['ID_PARTFORMAL'].replace('nan','0.0')

df_aveg['ID_MOTIVOATRIB'] = df_aveg['ID_MOTIVOATRIB'].replace('nan','0.0')

df_aveg['CONDUTOR_HABITUAL'] = df_aveg['CONDUTOR_HABITUAL'].replace('nan','2.0')

df_aveg['TPR_VIATURA'] = df_aveg['TPR_VIATURA'].replace('nan','Ninguem')


df_aveg['REEMBOLSO'] = df_aveg['REEMBOLSO'].replace('nan','0.0')

df_aveg['ID_MOT_ENQUADRAMENTO'] = df_aveg['ID_MOT_ENQUADRAMENTO'].replace('nan','0.0')

df_aveg['ID_TIPO_SIN'] = df_aveg['ID_TIPO_SIN'].replace('nan','0.0')

df_aveg['COUNTER_DAAA'] = df_aveg['COUNTER_DAAA'].replace('nan','0.0')

df_aveg['PREMIUM_RECEIPT_STATUS'] = df_aveg['PREMIUM_RECEIPT_STATUS'].replace('nan','-1.0')

df_aveg['CONTROL_QUAL'] = df_aveg['CONTROL_QUAL'].replace('nan','0.0')


df_aveg['ID_TPR'] = df_aveg['ID_TPR'].replace('nan','0.0')


df_aveg['ID_FORMATOPART'] = df_aveg['ID_FORMATOPART'].replace('nan','0.0')

df_aveg['QUARTA_DIRECTIVA'] = df_aveg['QUARTA_DIRECTIVA'].replace('nan','0.0')
#
#df_aveg[['hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos','quarter_sin', 'quarter_cos', 'semanaAno_sin', 'semanaAno_cos']].hist()

#pendente - binarizar id_codigo_postal
# pendente - Autoridade -> mais frequentes 



df_aveg_sum = df_aveg[['SINISTRO_hr_sin', 'SINISTRO_hr_cos', 'SINISTRO_mnth_sin', 'SINISTRO_mnth_cos','SINISTRO_quarter_sin', 
 'SINISTRO_quarter_cos', 'SINISTRO_semanaAno_sin', 'SINISTRO_semanaAno_cos',
 'AVERIG_mnth_sin', 'AVERIG_mnth_cos','AVERIG_quarter_sin','AVERIG_quarter_cos', 'AVERIG_semanaAno_sin','AVERIG_semanaAno_cos',
'CodigoPostal_GIF_1', 'dias_ate_participacao', 'dias_ate_respo_analisada', 'dias_entrada_companhia','dt_tratamento', 
'ID_CAUSASIN','ID_AVERIGUADOR_GIF', 'INCIDENTSOURCE_DESC', 
 'MOTOCICLOS',  'COND_ASSIST_FERIDOS', 'TESTEMUNHAS', 'RESP_SEGURADO',
 'ID_NOTIFICADOR', 'ID_FORMATOPART',  'HOUVE_ACIDENTE', 'SISTEMA_SUGERIU',
 'CHOQUE_CADEIA', 'PREMIUM_RECEIPT_STATUS', 'BONUS_MALUS_SUGERIDO', 
 'CONTROL_QUAL', 'ID_NATUREZAAV_Averiguacao', 
  'PROCESSTYPE_DESC', 'EMBATE_VIATURAS',
 'INTER_AUTORIDADE','ID_LOCALNOTIFICACAO', 'ID_EQUIPAUTIL',
 'ID_TIPOSINISTRO', 'ID_TPR', 'ENQUADRAMENTO',
 'MARCA',   'flag_Gep','Retratação',
 'PROCESSSTAGE_DESC', 'SITUACAO_SINISTRO', 'CONSEQUENCIA',
 'SEG_VIATURA_ACIDENTE',  'ENQUAD_TPR', 'ID_PARTFORMAL', 'ID_SEGURADORA', 'ID_MOTIVOATRIB',  
 'CONDUTOR_HABITUAL', 'TPR_VIATURA', 
 'AUTO_ALLOC_OFF', 'REEMBOLSO',  'IDS_REEM_CRIADO',  'ID_MOT_ENQUADRAMENTO', 'ID_TIPO_SIN',
 'COUNTER_DAAA','ID_COORDENADOR_GIF','QUARTA_DIRECTIVA', 'fraude']]


df_aveg_sum['RESP_SEGURADO'] = df_aveg_sum['RESP_SEGURADO'].astype(float)
df_aveg_sum['MOTOCICLOS'] = df_aveg_sum['MOTOCICLOS'].astype(float)
df_aveg_sum['COND_ASSIST_FERIDOS'] = df_aveg_sum['COND_ASSIST_FERIDOS'].astype(float)
df_aveg_sum['SISTEMA_SUGERIU'] = df_aveg_sum['SISTEMA_SUGERIU'].astype(float)
df_aveg_sum['CONTROL_QUAL'] = df_aveg_sum['CONTROL_QUAL'].astype(float)
df_aveg_sum['EMBATE_VIATURAS'] = df_aveg_sum['EMBATE_VIATURAS'].astype(float)
df_aveg_sum['CONTROL_QUAL'] = df_aveg_sum['CONTROL_QUAL'].astype(float)
df_aveg_sum['REEMBOLSO'] = df_aveg_sum['REEMBOLSO'].astype(float)
df_aveg_sum['IDS_REEM_CRIADO'] = df_aveg_sum['IDS_REEM_CRIADO'].astype(float)
df_aveg_sum['HOUVE_ACIDENTE'] = df_aveg_sum['HOUVE_ACIDENTE'].astype(float)
df_aveg_sum['QUARTA_DIRECTIVA'] = df_aveg_sum['QUARTA_DIRECTIVA'].astype(float)
df_aveg_sum['flag_Gep'] = df_aveg_sum['flag_Gep'].astype(float)
df_aveg_sum['Retratação'] = df_aveg_sum['Retratação'].astype(float)




df_aveg_sum.fillna(0, inplace=True)


df_aveg_sum.dtypes
df_aveg_sum[]


 df_aveg_sum = pd.get_dummies(df_aveg_sum, columns =['ID_CAUSASIN',
  'CHOQUE_CADEIA', 'PREMIUM_RECEIPT_STATUS', 'BONUS_MALUS_SUGERIDO', 
  'ID_NATUREZAAV_Averiguacao', 'PROCESSTYPE_DESC', 
 'INTER_AUTORIDADE','ID_LOCALNOTIFICACAO', 'ID_EQUIPAUTIL',
 'ID_TIPOSINISTRO', 'ID_TPR',  'ENQUADRAMENTO',
 'PROCESSSTAGE_DESC','SITUACAO_SINISTRO', 'CONSEQUENCIA',
 'SEG_VIATURA_ACIDENTE',  'ENQUAD_TPR', 'ID_PARTFORMAL', 'ID_SEGURADORA', 'ID_MOTIVOATRIB',  
 'CONDUTOR_HABITUAL', 'TPR_VIATURA', 
 'AUTO_ALLOC_OFF',   'ID_MOT_ENQUADRAMENTO', 'ID_TIPO_SIN',
 'COUNTER_DAAA','CodigoPostal_GIF_1','ID_AVERIGUADOR_GIF', 'INCIDENTSOURCE_DESC', 
 'TESTEMUNHAS', 'ID_NOTIFICADOR', 'ID_FORMATOPART', 'MARCA', 'ID_COORDENADOR_GIF'])



df_dummies = df_aveg[['ID_CAUSASIN',
  'CHOQUE_CADEIA', 'PREMIUM_RECEIPT_STATUS', 'BONUS_MALUS_SUGERIDO', 
  'ID_NATUREZAAV_Averiguacao', 'PROCESSTYPE_DESC', 
 'INTER_AUTORIDADE','ID_LOCALNOTIFICACAO', 'ID_EQUIPAUTIL',
 'ID_TIPOSINISTRO', 'ID_TPR',  'ENQUADRAMENTO',
 'PROCESSSTAGE_DESC','SITUACAO_SINISTRO', 'CONSEQUENCIA',
 'SEG_VIATURA_ACIDENTE',  'ENQUAD_TPR', 'ID_PARTFORMAL', 'ID_SEGURADORA', 'ID_MOTIVOATRIB',  
 'CONDUTOR_HABITUAL', 'TPR_VIATURA', 
 'AUTO_ALLOC_OFF',   'ID_MOT_ENQUADRAMENTO', 'ID_TIPO_SIN',
 'COUNTER_DAAA','CodigoPostal_GIF_1','ID_AVERIGUADOR_GIF', 'INCIDENTSOURCE_DESC', 
 'TESTEMUNHAS', 'ID_NOTIFICADOR', 'ID_FORMATOPART', 'MARCA', 'ID_COORDENADOR_GIF', 'fraude']]



 df_dummies = pd.get_dummies(df_dummies, columns =['ID_CAUSASIN',
  'CHOQUE_CADEIA', 'PREMIUM_RECEIPT_STATUS', 'BONUS_MALUS_SUGERIDO', 
  'ID_NATUREZAAV_Averiguacao', 'PROCESSTYPE_DESC', 
 'INTER_AUTORIDADE','ID_LOCALNOTIFICACAO', 'ID_EQUIPAUTIL',
 'ID_TIPOSINISTRO', 'ID_TPR',  'ENQUADRAMENTO',
 'PROCESSSTAGE_DESC','SITUACAO_SINISTRO', 'CONSEQUENCIA',
 'SEG_VIATURA_ACIDENTE',  'ENQUAD_TPR', 'ID_PARTFORMAL', 'ID_SEGURADORA', 'ID_MOTIVOATRIB',  
 'CONDUTOR_HABITUAL', 'TPR_VIATURA', 
 'AUTO_ALLOC_OFF',   'ID_MOT_ENQUADRAMENTO', 'ID_TIPO_SIN',
 'COUNTER_DAAA','CodigoPostal_GIF_1','ID_AVERIGUADOR_GIF', 'INCIDENTSOURCE_DESC', 
 'TESTEMUNHAS', 'ID_NOTIFICADOR', 'ID_FORMATOPART', 'MARCA', 'ID_COORDENADOR_GIF'])
# dataframe teste e treino

train, test = train_test_split(df_modelo, train_size = 0.8)

clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit( train.iloc[ :,:-1], train['fraude'])


scores = cross_val_score(clf1, test.iloc[:,:-1], test['fraude'] , cv=5)
print(scores)

y_pred1 = clf1.predict(test.iloc[ :,:-1])


test['pred1']=y_pred1
labels1 = ['real1', 'pred1']

cm1 = confusion_matrix(test['fraude'], test['pred1'])
sns.heatmap(cm1,  annot=True, fmt="g", cmap='viridis',xticklabels=True,yticklabels=True)
plt.xlabel('real')
plt.ylabel('predicted')
plt.show()

clf1.feature_importances_

for name, importance in zip(test.iloc[:,:-1].columns, clf1.feature_importances_.argsort()):
    plt.bar(name, importance)

def selectKImportance(model, X, k=20):
     return X.iloc[:,model.feature_importances_.argsort()[::-1][:k]]
 
 
df_Feat_Import = selectKImportance(clf1, train.iloc[ :,:-1], k=30)


###########
    


corr = df_aveg_sum2.corr()
sns.heatmap(corr, vmax=1., square=False).xaxis.tick_top()

df_aveg_sum2 = df_aveg[['SINISTRO_hr_sin', 'SINISTRO_hr_cos', 'SINISTRO_mnth_sin', 'SINISTRO_mnth_cos','SINISTRO_quarter_sin', 
 'SINISTRO_quarter_cos', 'SINISTRO_semanaAno_sin', 'SINISTRO_semanaAno_cos',
 'AVERIG_mnth_sin', 'AVERIG_mnth_cos','AVERIG_quarter_sin','AVERIG_quarter_cos', 'AVERIG_semanaAno_sin','AVERIG_semanaAno_cos',
'CodigoPostal_GIF_1', 'dias_ate_participacao', 'dias_ate_respo_analisada', 'dias_entrada_companhia','dt_tratamento', 
'ID_CAUSASIN','ID_AVERIGUADOR_GIF', 'INCIDENTSOURCE_DESC', 
 'MOTOCICLOS',  'COND_ASSIST_FERIDOS', 'TESTEMUNHAS', 'RESP_SEGURADO',
 'ID_NOTIFICADOR', 'ID_FORMATOPART',  'HOUVE_ACIDENTE', 'SISTEMA_SUGERIU',
 'CHOQUE_CADEIA', 'PREMIUM_RECEIPT_STATUS', 'BONUS_MALUS_SUGERIDO', 
 'CONTROL_QUAL', 'ID_NATUREZAAV_Averiguacao', 
  'PROCESSTYPE_DESC', 'EMBATE_VIATURAS',
 'INTER_AUTORIDADE','ID_LOCALNOTIFICACAO', 'ID_EQUIPAUTIL',
 'ID_TIPOSINISTRO', 'ID_TPR', 'ENQUADRAMENTO',
 'MARCA',   'flag_Gep','Retratação',
 'PROCESSSTAGE_DESC', 'SITUACAO_SINISTRO', 'CONSEQUENCIA',
 'SEG_VIATURA_ACIDENTE',  'ENQUAD_TPR', 'ID_PARTFORMAL', 'ID_SEGURADORA', 'ID_MOTIVOATRIB',  
 'CONDUTOR_HABITUAL', 'TPR_VIATURA', 
 'AUTO_ALLOC_OFF', 'REEMBOLSO',  'IDS_REEM_CRIADO',  'ID_MOT_ENQUADRAMENTO', 'ID_TIPO_SIN',
 'COUNTER_DAAA','ID_COORDENADOR_GIF','QUARTA_DIRECTIVA', 'fraude']]



df_modelo = df_aveg_sum[['SINISTRO_hr_sin', 'SINISTRO_hr_cos', 'SINISTRO_mnth_sin', 'SINISTRO_mnth_cos',
'AVERIG_mnth_sin', 'AVERIG_mnth_cos', 'AVERIG_quarter_sin', 'AVERIG_quarter_cos',
'AVERIG_semanaAno_sin', 'AVERIG_semanaAno_cos','CodigoPostal_GIF_1_4','CodigoPostal_GIF_1_3',  
'CodigoPostal_GIF_1_7','CodigoPostal_GIF_1_1','CodigoPostal_GIF_1_8',
'fraude']]


corr = df_modelo.corr()
sns.heatmap(corr)



'INCIDENTSOURCE_DESC_Matrícula', 
'PROCESSSTAGE_DESC_Saneamento / Seleção de Carteira', 'CodigoPostal_GIF_1_4', 
'CodigoPostal_GIF_1_3', 'CONSEQUENCIA_Material', 'CONDUTOR_HABITUAL_0.0', 
'CodigoPostal_GIF_1_2',  'CodigoPostal_GIF_1_7','AUTO_ALLOC_OFF_2', 
'SITUACAO_SINISTRO_Fechado', 'ID_NATUREZAAV_Averiguacao_20', 'CodigoPostal_GIF_1_1',  
'ID_CAUSASIN_05C', 'ID_CAUSASIN_013', 'SEG_VIATURA_ACIDENTE_0',
'ID_PARTFORMAL_1.0', 'ID_TPR_37.0', 'ID_NATUREZAAV_Averiguacao_1', 
'CONDUTOR_HABITUAL_1.0', 'ID_CAUSASIN_999','SITUACAO_SINISTRO_Reembolso',
 'ID_TPR_26.0', 'ID_CAUSASIN_05B', 'ID_NATUREZAAV_Averiguacao_40', 
'AUTO_ALLOC_OFF_0', 'ID_EQUIPAUTIL_257', 'CodigoPostal_GIF_1_8',
 'ID_EQUIPAUTIL_256', 'ID_TPR_13.0', 'ID_EQUIPAUTIL_262',