#%%
#Importing required packages
import os
import numpy as np
import pandas as pd
import glob
from collections import Counter
import random 

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler #Normalizing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

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
from sklearn.utils import check_random_state
 
from scipy.optimize import minimize

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.optimize import fmin
import time


from sklearn.datasets import make_classification
#abspath = os.path.abspath("__file__")
#dname = os.path.dirname(abspath)
#print(dname)
os.chdir("C:/Users/Yashu Dhatrika/Desktop/handover files")

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

#Variable selection
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


#Merging the prior data with the food data
subset_data=pd.merge(subset,sel_fooddata,how='left', on='STUDYID')
#Missing value treatment
subset_data=subset_data.fillna(subset_data.mean())

final_process=subset_data
final_process=final_process.drop(['STUDYID'], axis=1)

#Creating the dummy variables for the features
dummies = pd.get_dummies(final_process[['Education','Preparedness','Self_reported_weight','Anxious','Genes_Diabetes','V2_self_Weight','Race','Par_father','Par_Mother','Par_Brother','Par_Sis','Par_Sis_onset','Par_Bro_onset','Par_dad_onset','Par_maa_onset']]).rename(columns=lambda x: 'Category_' + str(x))
final_process = pd.concat([final_process, dummies], axis=1)
final_process = final_process.drop(['Education','Preparedness','Self_reported_weight','Anxious','Genes_Diabetes','V2_self_Weight','Race','Par_father','Par_Mother','Par_Brother','Par_Sis','Par_Sis_onset','Par_Bro_onset','Par_dad_onset','Par_maa_onset'], axis=1)

#Rescaling
y=final_process['Target_gb']
X=final_process.drop(['Target_gb','Category_Education_Not Avail' ,'Category_Preparedness_Not Avail' ,'Category_Self_reported_weight_Not Avail' ,'Category_Anxious_Not Avail' ,'Category_Genes_Diabetes_Not Avail' ,'Category_V2_self_Weight_Not Avail' ,'Category_Par_Sis_onset_Not Avail' ,'Category_Par_Bro_onset_Not Avail' ,'Category_Par_dad_onset_Not Avail' ,'Category_Par_maa_onset_Not Avail'], axis=1)
scaler = MinMaxScaler()
X_scale = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

#%% Train and test split
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.2, random_state=100)
complete_train=pd.concat([X_train, y_train], axis=1)

#%% Intiating the delta values
k=3
data_points=X_train.shape[0]
prob_target_delta=np.random.rand(data_points,k)/np.random.rand(data_points,k).sum(axis=1)[:,None]
d1={}


#%% Mixture of experts model using the upscaling method for multinomial regression-Method1
start = time.time()
pointer=0
while pointer==0:
    def expand_dataset(X, y_proba, factor=10, random_state=None):
        rng = check_random_state(random_state)
        n_classes = y_proba.shape[1]
        classes = np.arange(n_classes, dtype=int)
        for x, probs in zip(X, y_proba):
            for label in rng.choice(classes, size=factor, p=probs):
                yield x, label          
    X_adj=list(expand_dataset(complete_train.values,prob_target_delta))
    train_adj=[]
    target_adj_delta=[]
    act_target=[]
    for i in range(0,len(X_adj)):
        train_adj.append(X_adj[i][0][0:137])
        act_target.append(X_adj[i][0][137])
        target_adj_delta.append(X_adj[i][1])
    clf1 = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',penalty='l2',C=0.1,max_iter=5)
    clf1.fit(train_adj, target_adj_delta)
    pred_val=clf1.predict(train_adj)
    pred_prob=clf1.predict_proba(train_adj)
    #running the logistic regression K times using the gamma0 as their weights
    data_points=len(train_adj)
    k=3
#    d1={}
#    for x in range(k):
#            d1["gamma{0}".format(x)]=[ float(1/k) for _ in range(data_points)]
    prob_target_delta=np.random.rand(data_points,k)/np.random.rand(data_points,k).sum(axis=1)[:,None]
     
    prob_dic={}
    for j in range(k):
        clf = LogisticRegression(class_weight='balanced',penalty='l2')
        clf.fit(train_adj, act_target,sample_weight=list(prob_target_delta[:,j]))
        
        prob_dic[k]=clf.predict_proba(train_adj)[:,1]
        pred_val=clf.predict(train_adj)
    
    keys=prob_dic.keys()
    prob_output2 = np.array([prob_dic[i] for i in keys]).transpose()    
    updated_gamma=(pred_prob*prob_output2)
    
    norm_gamma=updated_gamma/updated_gamma.sum(axis=1)[:,None]
    his_gamma=prob_target_delta
    prob_target_delta=norm_gamma
    pointer=+1
print(len(prob_target_delta))    
while (np.linalg.norm(prob_target_delta-his_gamma)>0.001 or pointer<25):
    print(pointer)
    def expand_dataset(X, y_proba, factor=10, random_state=None):
        rng = check_random_state(random_state)
        n_classes = y_proba.shape[1]
        classes = np.arange(n_classes, dtype=int)
        for x, probs in zip(X, y_proba):
            for label in rng.choice(classes, size=factor, p=probs):
                yield x, label          
    X_adj=list(expand_dataset(complete_train.values,prob_target_delta))
    train_adj=[]
    target_adj_delta=[]
    act_target=[]
    for i in range(0,len(X_adj)):
        train_adj.append(X_adj[i][0][0:137])
        act_target.append(X_adj[i][0][137])
        target_adj_delta.append(X_adj[i][1])
    clf1 = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',penalty='l2',C=0.1,max_iter=i+5)
    clf1.fit(train_adj, target_adj_delta)
    pred_val=clf1.predict(train_adj)
    pred_prob=clf1.predict_proba(train_adj)
    #running the logistic regression K times using the gamma0 as their weights
    data_points=len(train_adj)
    K=3
#    d1={}
#    for x in range(k):
#            d1["gamma{0}".format(x)]=[ float(1/k) for _ in range(data_points)]
#    prob_target_delta=np.random.rand(data_points,k)/np.random.rand(data_points,k).sum(axis=1)[:,None]
     
    prob_dic={}
    for j in range(K):
        clf = LogisticRegression(class_weight='balanced',penalty='l2')
#        param = {'C':[0.1,1,10]}
#        clf = GridSearchCV(logreg,param,scoring='roc_auc',refit=True,cv=3)
        print('Went into this loop')
        clf.fit(train_adj, act_target,sample_weight=list(prob_target_delta[:,j]))
        
        prob_dic[k]=clf.predict_proba(train_adj)[:,1]
        pred_val=clf.predict(train_adj)
    
    keys=prob_dic.keys()
    prob_output2 = np.array([prob_dic[i] for i in keys]).transpose()    
    updated_gamma=(pred_prob*prob_output2)
    
    norm_gamma=updated_gamma/updated_gamma.sum(axis=1)[:,None]
    his_gamma=prob_target_delta
    prob_target_delta=norm_gamma
    pointer=pointer+1
    

end = time.time()
print(end - start)


#%%  building multinomial regression from scratch and testing on same random generated data

Xtrain=np.random.rand(10,5)
Ytrain=np.random.rand(10,3)
bias=np.random.rand(10,3)
Ytrain=Ytrain/Ytrain.sum(axis=1)[:,None]
W0=np.random.rand(3,5)
#print(Ytrain)
def softmax(W):
   vec=np.dot(Xtrain,W.reshape((3, 5)).T)
#   vec=np.add(vec,b)
   vec1=np.exp(vec)
   res=vec1.T/np.sum(vec1,axis=1)
   return res.T
#yhat=softmax(W)
#print(yhat.shape)
#print(len(Ytrain))

#
reg_lambda=1

def cross_entropy(W,reg_lambda):
    Y=W.reshape(3,5)
    sum1=0
    for i in range(len(Xtrain)):
        for k in range(3):
            sum1+=(Ytrain[i][k]*np.log(softmax(Y)[i][k]))
#            print('went into this loop')
    w2=np.sum(np.sum(np.abs(Y)**2,axis=-1))        
    return (-sum1+reg_lambda/(2.0) * (w2))
def first_order_grad(W,reg_lambda):
#    print(W.shape)
    Y=W.reshape(3,5)
    return np.array((np.dot(Xtrain.T,((-softmax(Y) + Ytrain)))+(reg_lambda/(2.0) * ((Y.T))))).flatten()

def second_ord_grad(W):
#    print(W.shape)
    Y=W.reshape(3,5)
    Ident_mat=np.identity(3)
    Hesian_matrix=np.zeros((3,3),dtype=object)
    for k in range(3):
        for j in range(3):
            Hesian_matrix[k][j]=np.zeros((5,5),dtype=object)
            for i in range(len(Xtrain)):
                Hesian_matrix[k][j]=-np.dot(Ytrain[i][k],Ident_mat[k][j]-softmax(Y)[i][j])*np.dot(Xtrain[i].reshape((5,1)),Xtrain[i].reshape((1,5)))+reg_lambda/(2.0)*np.identity(5)
#                print(Hesian_matrix[k][j].shape)
                return np.array(Hesian_matrix).flatten()
                
#                           

#res = minimize(cross_entropy, W0, method='Newton-CG',jac=first_order_grad, hess=second_ord_grad,options={'xtol': 1e-8, 'disp': True})    
###res.W
#    
res = minimize(cross_entropy, W0, method='Newton-CG',jac=first_order_grad,args=(reg_lambda),options={'xtol': 1e-8, 'disp': True})    
res.x
  

#%%


#%%
#simulating the data:

from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
# generate 2d classification dataset 
simulated_data, y = make_blobs(n_samples=2000, centers=3, n_features=10,random_state=1000)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=simulated_data[:,0], y=simulated_data[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()

Y_simulated=np.random.randint(2, size=len(simulated_data))

#%%
#complete_train= pd.DataFrame(np.concatenate((mixed_X, mixed_Y.reshape((1000,1))), axis=1))

complete= pd.DataFrame(np.concatenate((simulated_data, Y_simulated.reshape((2000,1))), axis=1))

#%%

complete_train=complete[:1000]
test=complete[1000:]
test_data=simulated_data[1000:]


#%%

#Method1: Running the mixture of experts on simulated data
k=3
data_points=complete_train.shape[0]
prob_delta=np.random.rand(data_points,k)
prob_target_delta=prob_delta/prob_delta.sum(axis=1)[:,None]



start = time.time()
pointer=0
while pointer==0:
    def expand_dataset(X, y_proba, factor=10, random_state=None):
        rng = check_random_state(random_state)
        n_classes = y_proba.shape[1]
        classes = np.arange(n_classes, dtype=int)
        for x, probs in zip(X, y_proba):
            for label in rng.choice(classes, size=factor, p=probs):
                yield x, label          
    X_adj=list(expand_dataset(complete_train.values,prob_target_delta))
    train_adj=[]
    target_adj_delta=[]
    act_target=[]
    print('Only once')
    for i in range(0,len(X_adj)):
        train_adj.append(X_adj[i][0][0:10])
        act_target.append(X_adj[i][0][10])
        target_adj_delta.append(X_adj[i][1])
    clf1 = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',penalty='l2',C=0.1,max_iter=20)
    clf1.fit(train_adj, target_adj_delta)
    pred_val=clf1.predict(train_adj)
    pred_prob=clf1.predict_proba(train_adj)
    #running the logistic regression K times using the gamma0 as their weights
    data_points=len(train_adj)
    k=3
    prob_delta=np.random.rand(data_points,k)
    prob_target_delta=prob_delta/prob_delta.sum(axis=1)[:,None]
     
    prob_dic={}
    for j in range(k):
        clf = LogisticRegression(class_weight='balanced',penalty='l2')
        clf.fit(train_adj, act_target,sample_weight=list(prob_target_delta[:,j]))
        
        prob_dic[j]=clf.predict_proba(train_adj)[:,1]
        pred_val=clf.predict(train_adj)
    
    keys=prob_dic.keys()
    prob_output2 = np.array([prob_dic[i] for i in keys]).transpose()    
    updated_gamma=(pred_prob*prob_output2)
    
    norm_gamma=updated_gamma/updated_gamma.sum(axis=1)[:,None]
    his_gamma=prob_target_delta
    prob_target_delta=norm_gamma
    pointer=+1
 
error_test=[]  
while (pointer<500):
    print(pointer)        
    X_adj=list(expand_dataset(complete_train.values,prob_target_delta))
    train_adj=[]
    target_adj_delta=[]
    act_target=[]
    for i in range(0,len(X_adj)):
        train_adj.append(X_adj[i][0][0:10])
        act_target.append(X_adj[i][0][10])
        target_adj_delta.append(X_adj[i][1])
    clf1 = LogisticRegression(random_state=100, solver='lbfgs',multi_class='multinomial',penalty='l2',C=1,max_iter=100)
    clf1.fit(train_adj, target_adj_delta)
    pred_val=clf1.predict(train_adj)
    pred_val_test=clf1.predict(test_data)
    test_data1=pd.DataFrame(np.concatenate((test, pred_val_test.reshape((1000,1))), axis=1))
    test_data1.columns = ['zero','one','two','three','four','five','six','seven','eight','nine','ten','ele']
    class0=test_data1[test_data1.ele ==0].drop(['ele'],axis=1)
    class0_label=pd.DataFrame(class0['ten']).values.tolist()
    class0_data=class0.drop(['ten'],axis=1)
    class1=test_data1[test_data1.ele ==1].drop(['ele'],axis=1)
    class1_label=pd.DataFrame(class1['ten']).values.tolist()
    class1_data=class1.drop(['ten'],axis=1)
    class2=test_data1[test_data1.ele ==2].drop(['ele'],axis=1)
    class2_label=pd.DataFrame(class2['ten']).values.tolist()
    class2_data=class2.drop(['ten'],axis=1)
    test_actuallabel=class0_label+class1_label+class2_label
    test_actuallabel = [item for sublist in test_actuallabel for item in sublist]
    
    pred_prob=clf1.predict_proba(train_adj)
    #running the logistic regression K times using the gamma0 as their weights
    data_points=len(train_adj)
    K=3
    pred_label_test=[]
    for j in range(K):
        clf = LogisticRegression(class_weight='balanced',penalty='l2')
        print('Went into this loop')
        clf.fit(train_adj, act_target,sample_weight=list(prob_target_delta[:,j]))
        prob_dic[j]=clf.predict_proba(train_adj)[:,1]
        pred_val=clf.predict(train_adj)
        if j==0 and len(class0_data)>0:
            pre=clf.predict(class0_data)
            pred_label_test.append(pre.tolist())
        if j==1 and len(class1_data)>0:
            pre=clf.predict(class1_data)
            pred_label_test.append(pre.tolist())
        if j==2 and len(class2_data)>0:
            pre=clf.predict(class2_data)
            pred_label_test.append(pre.tolist())
    pred_label_test = [item for sublist in pred_label_test for item in sublist]
    
    list3=list(np.array(pred_label_test) - np.array(test_actuallabel))

    error=sum(1 for x in list3 if x!=0)/len(list3)
    error_test.append(error)


    keys=prob_dic.keys()
    prob_output2 = np.array([prob_dic[i] for i in keys]).transpose()    
    updated_gamma=(pred_prob*prob_output2)
    
    norm_gamma=updated_gamma/updated_gamma.sum(axis=1)[:,None]
    his_gamma=prob_target_delta
    prob_target_delta=norm_gamma
    pointer=pointer+1
    print(np.linalg.norm(prob_target_delta-his_gamma))
    

end = time.time()
print(end - start)
#%%


#print(error_test)
x=list(range(len(error_test)))
plt.plot(x, error_test) 
plt.show()
#%%


#Method 2 :Running EM algorithm using the multinomial logistic regression  from scratch on Simulated data

classes=3

inter=np.ones([1000,1 ], dtype = int)
Xtrain=np.concatenate((inter,simulated_data[:1000]),axis=1)
data_points=Xtrain.shape[0]
Ytrain=Y_simulated[:1000]
Xtest=simulated_data[1000:]
Ytest=Y_simulated[1000:]

#%%
num_feat=Xtrain.shape[1]
prob_delta=np.random.rand(data_points,classes)
his_gamma=np.random.rand(data_points,classes)
prob_target_delta=prob_delta/prob_delta.sum(axis=1)[:,None]
W0=np.random.uniform(low=-1, high=1, size=(3,11))



#%%
pointer=1
while (np.linalg.norm(prob_target_delta-his_gamma)>0.001 or pointer<25):
    print(np.linalg.norm(prob_target_delta-his_gamma))
    def softmax(W):
        vec=np.dot(Xtrain,W.reshape((classes, num_feat)).T)
    #   vec=np.add(vec,b)
        vec1=np.exp(vec)
        res=vec1.T/np.sum(vec1,axis=1)
        return res.T
    reg_lambda=0
    def cross_entropy(W):
        Y=W.reshape((classes,num_feat))
        sum1=0
        for i in range(len(Xtrain)):
            for k in range(classes):
                sum1-=(prob_target_delta[i][k]*np.log(softmax(Y)[i][k]))
    #            print('went into this loop')
        w2=np.sum(np.sum(np.abs(Y)**2,axis=-1))        
        return (sum1+reg_lambda/(2.0) * (w2))
    def first_order_grad(W):
    #    print(W.shape)
        grad_1=[]
        Y=W.reshape((classes,num_feat))
        for k in range(classes):
            grad_1.append(list((np.dot(Xtrain.T,((-softmax(Y)[:,k] + prob_target_delta[:,k])))+(reg_lambda/(2.0) * ((Y[k].T))))))
        grad_1 = [item for sublist in grad_1 for item in sublist]    
        return grad_1    
    
    res = minimize(cross_entropy, W0, method='BFGS',jac=first_order_grad,options={'xtol': 1e-8, 'disp': True})    
    W0=res.x
    pred_prob=softmax(W0)
        #running the logistic regression K times using the gamma0 as their weights
    data_points=len(Xtrain)
    classes=3
    prob_delta=np.random.rand(data_points,classes)
    prob_target_delta=prob_delta/prob_delta.sum(axis=1)[:,None]
     
    prob_dic={}
    for j in range(classes):
        clf = LogisticRegression(class_weight='balanced',penalty='l2')
        clf.fit(Xtrain,Ytrain,sample_weight=list(prob_target_delta[:,j]))
        
        prob_dic[j]=clf.predict_proba(Xtrain)[:,1]
        pred_val=clf.predict(Xtrain)
    
    keys=prob_dic.keys()
    prob_output2 = np.array([prob_dic[i] for i in keys]).transpose()    
    updated_gamma=(pred_prob*prob_output2)
    
    norm_gamma=updated_gamma/updated_gamma.sum(axis=1)[:,None]
    his_gamma=prob_target_delta
    prob_target_delta=norm_gamma
    pointer=+1
#%%
print(np.linalg.norm(prob_target_delta-his_gamma))
print(his_gamma.shape)