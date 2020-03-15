#%%
import os
import numpy as np
import pandas as pd
import glob
from collections import Counter

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler #Normalizing
from sklearn.model_selection import train_test_split


from imblearn.over_sampling import SMOTE #oversampling
from imblearn.over_sampling import SMOTENC 
from imblearn.under_sampling import RandomUnderSampler #undersampling
from imblearn.over_sampling import RandomOverSampler

from sklearn.linear_model import LogisticRegression 
import statsmodels.discrete.discrete_model as sm
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt 
 

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit





#abspath = os.path.abspath("__file__")
#dname = os.path.dirname(abspath)
#print(dname)
os.chdir("C:/Users/Yashu Dhatrika/Desktop/My files/RAship/Diabetes/numom_data")

#%%
 #Joining all tables(Screening,Visit 1 ,Vist2 and Physical Activity)
 dfs=[]
 keys=[]
 csv_files=glob.glob('*csv')
 for filename in csv_files:
     print(filename)
     data=pd.read_csv(filename,encoding='cp1252')
     columns = ['STUDYID']
     temp=pd.DataFrame(data, columns=columns)
     keys.append(temp)
 #    print(keys)
 #    print(type(data))
     dfs.append(data)

 keys=pd.concat(keys,ignore_index=True)
 unique1=pd.DataFrame(keys.STUDYID.unique(),columns=['STUDYID'])
 print(len(unique1))

#%%
 consol=pd.merge(unique1, dfs[7], how='left', on='STUDYID')
 for i in range(2,len(dfs)):
     if i>7:
         consol=pd.merge(consol, dfs[i], how='left', on='STUDYID')

 phys_act=dfs[6][['STUDYID','METs']]
 phys_act=phys_act.groupby('STUDYID', as_index=False).agg({"METs": "mean"})
 
 #%%
 consol=pd.merge(consol, phys_act, how='left', on='STUDYID')


#%%
pre_process=consol
fooddata=dfs[5]
sel_fooddata=fooddata.iloc[:,np.r_[0,21:74,699:736]]


#%%

#%%
subset=pre_process[['STUDYID','oDM','S01B01','S01B02','S01B03a','S01B03b','S01B03c','S01B03d','S01B03e','S01B03e_SP','S01B03f','S01B04','V1AD02f','V1BA01_LB','V1BA01a','V1BA02a','V1BA02b','V1BA02c','V1BA03a','V1BA03b','V1BA03c','V1BA04a','V1BA04b','V1BA04c','V1BA05a','V1BA05b','V1BA05c','V1BA06a1','V1BA06a2','V1BA06b1','V1BA06b2','V1BA07a','V1BA07b','V1BA07c','V1EA02a','V2AE04','V2AE04a1a','V2AE04a1b','V2AE04a2a','V2AE04a2b','V2AE04a3a','V2AE04a3b','V2AE04a4a','V2AE04a4b','V2AE04a5a','V2AE04a5b','V2AE04a6a','V2AE04a6b','V2BA01_LB','V2BA01a','V2BA02a1','V2BA02a2','V2BA02b1','V2BA02b2','METs']]

#subset=pre_process[['oDM','S01B01','S01B02','S01B03a','S01B03b','S01B03c','S01B03d','S01B03e','S01B03e_SP','S01B03f','S01B04','V1AD02f','V1BA01_LB','V1BA01a','V1BA02a','V1BA02b','V1BA02c','V1BA03a','V1BA03b','V1BA03c','V1BA04a','V1BA04b','V1BA04c','V1BA05a','V1BA05b','V1BA05c','V1BA06a1','V1BA06a2','V1BA06b1','V1BA06b2','V1BA07a','V1BA07b','V1BA07c','V1EA02a','V2AE04','V2AE04a1a','V2AE04a1b','V2AE04a2a','V2AE04a2b','V2AE04a3a','V2AE04a3b','V2AE04a4a','V2AE04a4b','V2AE04a5a','V2AE04a5b','V2AE04a6a','V2AE04a6b','V2BA01_LB','V2BA01a','V2BA02a1','V2BA02a2','V2BA02b1','V2BA02b2','METs']]

#target variable
subset = subset.drop(subset[(subset.oDM ==1)].index)
subset=subset[pd.notnull(subset['oDM'])]
subset.oDM[subset.oDM == 3] = 0
subset.oDM[subset.oDM == 2] = 1
subset=subset.rename(columns = {'oDM':'Target_gb'})

#Age variable
subset=subset.rename(columns = {'S01B01':'Age'})
subset['Age']=subset[["Age"]].astype(str).replace('D',np.nan)
subset['Age'] = subset['Age'].astype(float)
mean_age=subset['Age'].mean()
subset['Age'].fillna(mean_age, inplace=True)

#Ethnicity

subset["Race"] = np.nan

subset.loc[ subset.S01B02=='1', 'Race' ] = 'Hispanic/Native'
subset.loc[ subset.S01B03a== '1', 'Race' ] = 'White'
subset.loc[ subset.S01B03b== '1', 'Race' ] = 'Black/African'
subset.loc[ subset.S01B03c== '1', 'Race' ] = 'American Indian/Alaska Native'
subset.loc[ subset.S01B03d== '1', 'Race' ] = 'Asian'

subset['Race'].fillna('Other', inplace=True)
subset=subset.drop(['S01B02', 'S01B03a', 'S01B03b','S01B03c','S01B03d','S01B03e','S01B03e_SP','S01B03f'], axis=1)


#Education
subset=subset.rename(columns = {'S01B04':'Education'})
subset['Education']=subset['Education'].astype(str).replace('D',np.nan)
subset['Education']=subset['Education'].astype(str).replace('nan',np.nan)
subset['Education'].fillna('Not Avail', inplace=True)

#preparedness
subset=subset.rename(columns = {'V1AD02f':'Preparedness'})
subset.Preparedness[subset.Preparedness == 1] = 'Yes'
subset.Preparedness[subset.Preparedness == 2] = 'No'
subset['Preparedness'].fillna('Not Avail', inplace=True)


#V1_weight
subset=subset.rename(columns = {'V1BA01_LB':'V1_Weight'})
subset['V1_Weight']=subset[["V1_Weight"]].astype(str).replace('D',np.nan)
subset['V1_Weight'] = subset['V1_Weight'].astype(float)
mean_V1_Weight=subset['V1_Weight'].mean()
subset['V1_Weight'].fillna(mean_V1_Weight, inplace=True)

#'Self_reported_weight'
subset=subset.rename(columns = {'V1BA01a':'Self_reported_weight'})
subset.Self_reported_weight[subset.Self_reported_weight == 1] = 'Yes'
subset.Self_reported_weight[subset.Self_reported_weight == 0] = 'No'
subset['Self_reported_weight'].fillna('Not Avail', inplace=True)

#Height
subset["Height"] = subset['V1BA02c']
subset.loc[subset.V1BA02c.isnull(), "Height"] =subset['V1BA02b']
subset.loc[subset.V1BA02c.isnull() & subset.V1BA02b.isnull() , "Height"] =subset['V1BA02a']
subset['Height']=subset[["Height"]].astype(str).replace('D',np.nan)
subset['Height']=subset[["Height"]].astype(str).replace('S',np.nan)
subset['Height'] = subset['Height'].astype(float)
mean_Height=subset['Height'].mean()
subset['Height'].fillna(mean_Height, inplace=True)

subset=subset.drop(['V1BA02c','V1BA02b','V1BA02a'],axis=1)

#natural_waist_circum
subset["Nat_waist_circum"] = subset['V1BA03c']
subset.loc[subset.V1BA03c.isnull(), "Nat_waist_circum"] =subset['V1BA03b']
subset.loc[subset.V1BA03c.isnull() & subset.V1BA03b.isnull() , "Nat_waist_circum"] =subset['V1BA03a']
subset['Nat_waist_circum']=subset[["Nat_waist_circum"]].astype(str).replace('D',np.nan)
subset['Nat_waist_circum']=subset[["Nat_waist_circum"]].astype(str).replace('S',np.nan)
subset['Nat_waist_circum'] = subset['Nat_waist_circum'].astype(float)
mean_Nat_waist_circum=subset['Nat_waist_circum'].mean()
subset['Nat_waist_circum'].fillna(mean_Nat_waist_circum, inplace=True)
subset=subset.drop(['V1BA03c','V1BA03b','V1BA03a'],axis=1)

#iliac_waist_circum
subset["iliac_waist_circum"] = subset['V1BA04c']
subset.loc[subset.V1BA04c.isnull(), "iliac_waist_circum"] =subset['V1BA04b']
subset.loc[subset.V1BA04c.isnull() & subset.V1BA04b.isnull() , "iliac_waist_circum"] =subset['V1BA04a']
subset['iliac_waist_circum']=subset[["iliac_waist_circum"]].astype(str).replace('D',np.nan)
subset['iliac_waist_circum']=subset[["iliac_waist_circum"]].astype(str).replace('S',np.nan)
subset['iliac_waist_circum'] = subset['iliac_waist_circum'].astype(float)
mean_iliac_waist_circum=subset['iliac_waist_circum'].mean()
subset['iliac_waist_circum'].fillna(mean_iliac_waist_circum, inplace=True)
subset=subset.drop(['V1BA04c','V1BA04b','V1BA04a'],axis=1)

#Hip_circum
subset["Hip_circum"] = subset['V1BA05c']
subset.loc[subset.V1BA05c.isnull(), "Hip_circum"] =subset['V1BA05b']
subset.loc[subset.V1BA05c.isnull() & subset.V1BA05b.isnull() , "Hip_circum"] =subset['V1BA05a']
subset['Hip_circum']=subset[["Hip_circum"]].astype(str).replace('D',np.nan)
subset['Hip_circum']=subset[["Hip_circum"]].astype(str).replace('S',np.nan)
subset['Hip_circum'] = subset['Hip_circum'].astype(float)
mean_Hip_circum=subset['Hip_circum'].mean()
subset['Hip_circum'].fillna(mean_Hip_circum, inplace=True)
subset=subset.drop(['V1BA05c','V1BA05b','V1BA05a'],axis=1)


#BP_systolic
subset["BP_systolic"] = subset['V1BA06a2']
subset.loc[subset.V1BA06a2.isnull(), "BP_systolic"] =subset['V1BA06a1']
subset['BP_systolic']=subset[["BP_systolic"]].astype(str).replace('D',np.nan)
subset['BP_systolic']=subset[["BP_systolic"]].astype(str).replace('S',np.nan)
subset['BP_systolic'] = subset['BP_systolic'].astype(float)
mean_BP_systolic=subset['BP_systolic'].mean()
subset['BP_systolic'].fillna(mean_BP_systolic, inplace=True)
subset=subset.drop(['V1BA06a1','V1BA06a2'],axis=1)


#BP_diastolic
subset["BP_diastolic"] = subset['V1BA06b2']
subset.loc[subset.V1BA06b2.isnull(), "BP_diastolic"] =subset['V1BA06b1']
subset['BP_diastolic']=subset[["BP_diastolic"]].astype(str).replace('D',np.nan)
subset['BP_diastolic']=subset[["BP_diastolic"]].astype(str).replace('S',np.nan)
subset['BP_diastolic'] = subset['BP_diastolic'].astype(float)
mean_BP_diastolic=subset['BP_diastolic'].mean()
subset['BP_diastolic'].fillna(mean_BP_diastolic, inplace=True)
subset=subset.drop(['V1BA06b1','V1BA06b2'],axis=1)


#Neck_circum
subset["Neck_circum"] = subset['V1BA07c']
subset.loc[subset.V1BA07c.isnull(), "Neck_circum"] =subset['V1BA07b']
subset.loc[subset.V1BA07c.isnull() & subset.V1BA07b.isnull() , "Neck_circum"] =subset['V1BA07a']
subset['Neck_circum']=subset[["Neck_circum"]].astype(str).replace('D',np.nan)
subset['Neck_circum']=subset[["Neck_circum"]].astype(str).replace('S',np.nan)
subset['Neck_circum'] = subset['Neck_circum'].astype(float)
mean_Neck_circum=subset['Neck_circum'].mean()
subset['Neck_circum'].fillna(mean_Neck_circum, inplace=True)
subset=subset.drop(['V1BA07c','V1BA07b','V1BA07a'],axis=1)

#Anxious
subset=subset.rename(columns = {'V1EA02a':'Anxious'})
subset.Anxious[subset.Anxious == 1] = 'Not at all'
subset.Anxious[subset.Anxious == 2] = 'Somewhat'
subset.Anxious[subset.Anxious == 3] = 'VeryMuch'
subset['Anxious'].fillna('Not Avail', inplace=True)

#Diabetes:
subset=subset.rename(columns = {'V2AE04':'Genes_Diabetes'})
subset.Genes_Diabetes[subset.Genes_Diabetes == '1'] = 'Yes'
subset.Genes_Diabetes[subset.Genes_Diabetes == '2'] = 'No'
subset['Genes_Diabetes']=subset[["Genes_Diabetes"]].astype(str).replace('D',np.nan)
subset['Genes_Diabetes']=subset[["Genes_Diabetes"]].astype(str).replace('nan',np.nan)
subset['Genes_Diabetes'].fillna('Not Avail', inplace=True)

#Diabetes:
subset=subset.rename(columns = {'V2AE04':'Genes_Diabetes'})
subset.Genes_Diabetes[subset.Genes_Diabetes == '1'] = 'Yes'
subset.Genes_Diabetes[subset.Genes_Diabetes == '2'] = 'No'
subset['Genes_Diabetes']=subset[["Genes_Diabetes"]].astype(str).replace('D',np.nan)
subset['Genes_Diabetes']=subset[["Genes_Diabetes"]].astype(str).replace('nan',np.nan)
subset['Genes_Diabetes'].fillna('Not Avail', inplace=True)
#Par_Father:
subset["Par_father"] = 0
subset.loc[(subset.V2AE04a1a==2.0) |(subset.V2AE04a2a==2.0) |(subset.V2AE04a3a==2.0) |(subset.V2AE04a4a==2.0) , 'Par_father' ] = 1

#Par_Mother
subset["Par_Mother"] = 0
subset.loc[(subset.V2AE04a1a==1.0) |(subset.V2AE04a2a==1.0) |(subset.V2AE04a3a==1.0) |(subset.V2AE04a4a==1.0) , 'Par_Mother' ] = 1
#subset['Par_father'].fillna(0, inplace=True)

#Par_Brother
subset["Par_Brother"] = 0
subset.loc[(subset.V2AE04a1a==3.0) |(subset.V2AE04a2a==3.0) |(subset.V2AE04a3a==3.0) |(subset.V2AE04a4a==3.0) , 'Par_Brother' ] = 1
#subset['Par_father'].fillna(0, inplace=True)

#Par_Sis
subset["Par_Sis"] = 0
subset.loc[(subset.V2AE04a1a==4.0) |(subset.V2AE04a2a==4.0) |(subset.V2AE04a3a==4.0) |(subset.V2AE04a4a==4.0) , 'Par_Sis' ] = 1
#subset['Par_father'].fillna(0, inplace=True)

#Par_Sis_onset
subset["Par_Sis_onset"] = 'Not Avail'
subset.loc[((subset.V2AE04a1a==4.0) & (subset.V2AE04a1b=='1')), 'Par_Sis_onset'] = 'Juvenile'
subset.loc[((subset.V2AE04a2a==4.0) & (subset.V2AE04a2b=='1')), 'Par_Sis_onset'] = 'Juvenile'
subset.loc[((subset.V2AE04a3a==4.0) & ((subset.V2AE04a3b=='1') | (subset.V2AE04a3b==1))), 'Par_Sis_onset'] = 'Juvenile'
subset.loc[((subset.V2AE04a4a==4.0) & (subset.V2AE04a4b==1)), 'Par_Sis_onset'] = 'Juvenile'

subset.loc[((subset.V2AE04a1a==4.0) & (subset.V2AE04a1b=='2')), 'Par_Sis_onset'] = 'Adult'
subset.loc[((subset.V2AE04a2a==4.0) & (subset.V2AE04a2b=='2')), 'Par_Sis_onset'] = 'Adult'
subset.loc[((subset.V2AE04a3a==4.0) & ((subset.V2AE04a3b=='2') | (subset.V2AE04a3b==2))), 'Par_Sis_onset'] = 'Adult'
subset.loc[((subset.V2AE04a4a==4.0) & (subset.V2AE04a4b==2)), 'Par_Sis_onset'] = 'Adult'

#Par_Bro_onset
subset["Par_Bro_onset"] = 'Not Avail'
subset.loc[((subset.V2AE04a1a==3.0) & (subset.V2AE04a1b=='1')), 'Par_Bro_onset'] = 'Juvenile'
subset.loc[((subset.V2AE04a2a==3.0) & (subset.V2AE04a2b=='1')), 'Par_Bro_onset'] = 'Juvenile'
subset.loc[((subset.V2AE04a3a==3.0) & ((subset.V2AE04a3b=='1') | (subset.V2AE04a3b==1))), 'Par_Bro_onset'] = 'Juvenile'
subset.loc[((subset.V2AE04a4a==3.0) & (subset.V2AE04a4b==1)), 'Par_Bro_onset'] = 'Juvenile'

subset.loc[((subset.V2AE04a1a==3.0) & (subset.V2AE04a1b=='2')), 'Par_Bro_onset'] = 'Adult'
subset.loc[((subset.V2AE04a2a==3.0) & (subset.V2AE04a2b=='2')), 'Par_Bro_onset'] = 'Adult'
subset.loc[((subset.V2AE04a3a==3.0) & ((subset.V2AE04a3b=='2') | (subset.V2AE04a3b==2))), 'Par_Bro_onset'] = 'Adult'
subset.loc[((subset.V2AE04a4a==3.0) & (subset.V2AE04a4b==2)), 'Par_Bro_onset'] = 'Adult'

#Par_dad_onset
subset["Par_dad_onset"] = 'Not Avail'
subset.loc[((subset.V2AE04a1a==2.0) & (subset.V2AE04a1b=='1')), 'Par_dad_onset'] = 'Juvenile'
subset.loc[((subset.V2AE04a2a==2.0) & (subset.V2AE04a2b=='1')), 'Par_dad_onset'] = 'Juvenile'
subset.loc[((subset.V2AE04a3a==2.0) & ((subset.V2AE04a3b=='1') | (subset.V2AE04a3b==1))), 'Par_dad_onset'] = 'Juvenile'
subset.loc[((subset.V2AE04a4a==2.0) & (subset.V2AE04a4b==1)), 'Par_dad_onset'] = 'Juvenile'

subset.loc[((subset.V2AE04a1a==2.0) & (subset.V2AE04a1b=='2')), 'Par_dad_onset'] = 'Adult'
subset.loc[((subset.V2AE04a2a==2.0) & (subset.V2AE04a2b=='2')), 'Par_dad_onset'] = 'Adult'
subset.loc[((subset.V2AE04a3a==2.0) & ((subset.V2AE04a3b=='2') | (subset.V2AE04a3b==2))), 'Par_dad_onset'] = 'Adult'
subset.loc[((subset.V2AE04a4a==2.0) & (subset.V2AE04a4b==2)), 'Par_dad_onset'] = 'Adult'

#Par_maa_onset
subset["Par_maa_onset"] = 'Not Avail'
subset.loc[((subset.V2AE04a1a==1.0) & (subset.V2AE04a1b=='1')), 'Par_maa_onset'] = 'Juvenile'
subset.loc[((subset.V2AE04a2a==1.0) & (subset.V2AE04a2b=='1')), 'Par_maa_onset'] = 'Juvenile'
subset.loc[((subset.V2AE04a3a==1.0) & ((subset.V2AE04a3b=='1') | (subset.V2AE04a3b==1))), 'Par_maa_onset'] = 'Juvenile'
subset.loc[((subset.V2AE04a4a==1.0) & (subset.V2AE04a4b==1)), 'Par_maa_onset'] = 'Juvenile'

subset.loc[((subset.V2AE04a1a==1.0) & (subset.V2AE04a1b=='2')), 'Par_maa_onset'] = 'Adult'
subset.loc[((subset.V2AE04a2a==1.0) & (subset.V2AE04a2b=='2')), 'Par_maa_onset'] = 'Adult'
subset.loc[((subset.V2AE04a3a==1.0) & ((subset.V2AE04a3b=='2') | (subset.V2AE04a3b==2))), 'Par_maa_onset'] = 'Adult'
subset.loc[((subset.V2AE04a4a==1.0) & (subset.V2AE04a4b==2)), 'Par_maa_onset'] = 'Adult'

subset=subset.drop(['V2AE04a1a' ,'V2AE04a1b' ,'V2AE04a2a' ,'V2AE04a2b' ,'V2AE04a3a' ,'V2AE04a3b' ,'V2AE04a4a' ,'V2AE04a4b' ,'V2AE04a5a' ,'V2AE04a5b' ,'V2AE04a6a' ,'V2AE04a6b'],axis=1)


#V2_Weight
subset=subset.rename(columns = {'V2BA01_LB':'V2_Weight'})
subset['V2_Weight']=subset[["V2_Weight"]].astype(str).replace('D',np.nan)
subset['V2_Weight'] = subset['V2_Weight'].astype(float)
mean_V2_Weight=subset['V2_Weight'].mean()
subset['V2_Weight'].fillna(mean_V2_Weight, inplace=True)

#V2BP_systolic
subset["V2_Systolic_BP"] = subset['V2BA02a2']
subset.loc[subset.V2BA02a2.isnull(), "V2_Systolic_BP"] =subset['V2BA02a1']
subset['V2_Systolic_BP']=subset[["V2_Systolic_BP"]].astype(str).replace('D',np.nan)
subset['V2_Systolic_BP']=subset[["V2_Systolic_BP"]].astype(str).replace('S',np.nan)
subset['V2_Systolic_BP'] = subset['V2_Systolic_BP'].astype(float)
mean_V2_Systolic_BP=subset['V2_Systolic_BP'].mean()
subset['V2_Systolic_BP'].fillna(mean_V2_Systolic_BP, inplace=True)
subset=subset.drop(['V2BA02a2','V2BA02a1'],axis=1)

#V2BP_diastolic
subset["V2_Systolic_BP"] = subset['V2BA02b2']
subset.loc[subset.V2BA02b2.isnull(), "V2_diaSystolic_BP"] =subset['V2BA02b1']
subset['V2_diaSystolic_BP']=subset[["V2_diaSystolic_BP"]].astype(str).replace('D',np.nan)
subset['V2_diaSystolic_BP']=subset[["V2_diaSystolic_BP"]].astype(str).replace('S',np.nan)
subset['V2_diaSystolic_BP'] = subset['V2_diaSystolic_BP'].astype(float)
mean_V2_diaSystolic_BP=subset['V2_diaSystolic_BP'].mean()
subset['V2_diaSystolic_BP'].fillna(mean_V2_diaSystolic_BP, inplace=True)
subset=subset.drop(['V2BA02b2','V2BA02b1'],axis=1)

mean_mets=subset['METs'].mean()
subset['METs'].fillna(mean_mets, inplace=True)

#V2_self_Weight
subset=subset.rename(columns = {'V2BA01a':'V2_self_Weight'})
subset['V2_self_Weight'].fillna('Not Avail', inplace=True)

subset['V2_Systolic_BP']=subset[["V2_Systolic_BP"]].astype(str).replace('D',np.nan)
subset['V2_Systolic_BP']=subset[["V2_Systolic_BP"]].astype(str).replace('S',np.nan)
subset['V2_Systolic_BP'] = subset['V2_Systolic_BP'].astype(float)
mean_V2_Systolic_BP=subset['V2_Systolic_BP'].mean()
subset['V2_Systolic_BP'].fillna(mean_V2_Systolic_BP, inplace=True)


# Joing the prior table with that food data
subset_data=pd.merge(subset,sel_fooddata,how='left', on='STUDYID')
# Missing value treatment

subset_data=subset_data.fillna(subset_data.mean())

final_process=subset_data
final_process=final_process.drop(['STUDYID'], axis=1)

#dummy creation of the categorical varaibles

dummies = pd.get_dummies(final_process[['Education','Preparedness','Self_reported_weight','Anxious','Genes_Diabetes','V2_self_Weight','Race','Par_father','Par_Mother','Par_Brother','Par_Sis','Par_Sis_onset','Par_Bro_onset','Par_dad_onset','Par_maa_onset']]).rename(columns=lambda x: 'Category_' + str(x))
final_process = pd.concat([final_process, dummies], axis=1)
final_process = final_process.drop(['Education','Preparedness','Self_reported_weight','Anxious','Genes_Diabetes','V2_self_Weight','Race','Par_father','Par_Mother','Par_Brother','Par_Sis','Par_Sis_onset','Par_Bro_onset','Par_dad_onset','Par_maa_onset'], axis=1)


#Rescaling
y=final_process['Target_gb']
X=final_process.drop(['Target_gb','Category_Education_Not Avail' ,'Category_Preparedness_Not Avail' ,'Category_Self_reported_weight_Not Avail' ,'Category_Anxious_Not Avail' ,'Category_Genes_Diabetes_Not Avail' ,'Category_V2_self_Weight_Not Avail' ,'Category_Par_Sis_onset_Not Avail' ,'Category_Par_Bro_onset_Not Avail' ,'Category_Par_dad_onset_Not Avail' ,'Category_Par_maa_onset_Not Avail'], axis=1)
scaler = MinMaxScaler()
X_scale = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


#%%
#No oversampling
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.2, random_state=100)

print('Original dataset samples per class {}'.format(Counter(y_train)))
print('Original dataset samples per class {}'.format(Counter(y_test)))
#Oversamplying

ros = RandomOverSampler(random_state=100,ratio=1)
X_train_res, y_train_res = ros.fit_sample(X_train, y_train)

#sm=SMOTENC(random_state=100,categorical_features=[13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46])
#X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
#print('Original dataset samples per class {}'.format(Counter(y_train)))
#print('Resampled dataset samples per class {}'.format(Counter(y_train_res)))


#%%
#Logistic regression
logreg = LogisticRegression(class_weight='balanced')
param = {'C':[0.001,0.003,0.005,0.01,0.03,0.05,0.1,0.3,0.5,1,2,3,3,4,5,10,20]}
#clf = GridSearchCV(logreg,param,scoring='roc_auc',refit=True,cv=10)
#clf.fit(X_train,y_train)

ss = StratifiedShuffleSplit(n_splits=5,test_size=0.4, random_state=1000)

clf = GridSearchCV(logreg,param, scoring='roc_auc',cv=ss)
#sorted(clf.cv_results_.keys())
clf.fit(X_train_res, y_train_res)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
logit_roc_auc = roc_auc_score(y_test, y_pred)
print(logit_roc_auc)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#print((list(X_test),clf.best_estimator_.coef_))


#%%
#SVM

from sklearn.svm import SVC  
svclassifier = SVC(class_weight='balanced')  
#svclassifier.fit(X_train_res, y_train_res)  

#y_pred = svclassifier.predict(X_test)

param = {'C':[0.001,0.03,0.3,0.5,1,5,10,20,30],'kernel':['linear','rbf']}
#clf = GridSearchCV(logreg,param,scoring='roc_auc',refit=True,cv=10)
#clf.fit(X_train,y_train)

ss = StratifiedShuffleSplit(n_splits=5,test_size=0.4, random_state=150)

clf = GridSearchCV(svclassifier,param, scoring='roc_auc',cv=ss)
#sorted(clf.cv_results_.keys())
clf.fit(X_train_res,y_train_res)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()


plt.rc("font", size=14)
logit_roc_auc = roc_auc_score(y_test, y_pred)
print(logit_roc_auc)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


#%%
#Random forest
feature_list = list(X.columns)

#sm = SMOTE(random_state=2,ratio=0.7)
#X_train, y_train = sm.fit_sample(X_train, y_train.ravel())
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier(class_weight='balanced')
#rfc.fit(X_train,y_train)


param = { 
    'n_estimators': [100, 200, 700],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' :[3,5,7],
    'criterion': ['gini','entropy']
}
ss = StratifiedShuffleSplit(n_splits=5,test_size=0.4, random_state=0)

clf = GridSearchCV(rfc,param, scoring='roc_auc',cv=ss)
#sorted(clf.cv_results_.keys())
clf.fit(X_train_res,y_train_res)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
logit_roc_auc = roc_auc_score(y_test, y_pred)
print(logit_roc_auc)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()



#%%
#nested crossvalidation

NUM_TRIALS=2
nested_scores=[]
from sklearn.svm import SVC  
svclassifier = SVC(class_weight='balanced')  
param = {'C':[0.001,0.03,0.3,0.5,1,5,10,20,30],'kernel':['linear','rbf']}
for i in range(NUM_TRIALS):

    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
    inner_cv = StratifiedShuffleSplit(n_splits=5,test_size=0.2, random_state=i)
    outer_cv = StratifiedShuffleSplit(n_splits=5,test_size=0.2, random_state=i)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=svclassifier, param_grid=param,scoring='roc_auc', cv=inner_cv)
    clf.fit(X_scale, y)

    # Nested CV with parameter optimization
    nested_score = cross_val_score(clf, X=X_scale, y=y, scoring='roc_auc',cv=outer_cv)
    print(nested_score)
    nested_scores.append(nested_score)

score = nested_scores


print("Average difference of {0:6f} with std. dev. of {1:6f}."
      .format(np.mean(score), np.std(score)))

#%%
#undersampling and adaboost
def adaboost(X_train, X_test, y_train):
    model = AdaBoostClassifier(n_estimators=100, random_state=42)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test) 
    return y_pred

X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.2, random_state=100)
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)
y_rus = adaboost(X_res, X_test, y_res)

classification_report(y_test, y_rus)
logit_roc_auc = roc_auc_score(y_test, y_rus)
print(logit_roc_auc)

#%%undersample and bagging using SVM
import numpy as np
from imblearn.ensemble import BalancedBaggingClassifier
list1=[]
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.2)
    svclassifier = SVC(class_weight='balanced',C=1,kernel='linear')  
    bbc = BalancedBaggingClassifier(base_estimator=svclassifier,sampling_strategy='auto',replacement=True,random_state=100,n_estimators=50)
    bbc.fit(X_train, y_train) 
    BalancedBaggingClassifier(...)
    y_pred = bbc.predict(X_test)
    
    classification_report(y_test, y_pred)
    logit_roc_auc = roc_auc_score(y_test, y_pred)
    list1.append(logit_roc_auc)
print(np.mean(list1))
print(np.std(list1))
#%%

#undersample and bagging using Ada
import numpy as np
from imblearn.ensemble import BalancedBaggingClassifier
list1=[]
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.2)
#    ada = AdaBoostClassifier(n_estimators=100, random_state=42) 
    log=LogisticRegression(class_weight='balanced')
    bbc = BalancedBaggingClassifier(base_estimator=log,sampling_strategy='auto',replacement=True,random_state=100,n_estimators=50)
    bbc.fit(X_train, y_train) 
    BalancedBaggingClassifier(...)
    y_pred = bbc.predict(X_test)
    
    classification_report(y_test, y_pred)
    logit_roc_auc = roc_auc_score(y_test, y_pred)
    list1.append(logit_roc_auc)
print(np.mean(list1))
print(np.std(list1))

#%%PCA
from sklearn.decomposition import PCA
#pca = PCA(n_components=2)
#pca.fit(X)

pca = PCA().fit(X_scale)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')




