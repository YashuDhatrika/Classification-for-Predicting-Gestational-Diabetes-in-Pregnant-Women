# -*- coding: utf-8 -*-
"""
Created on Wed May  8 08:46:52 2019

@author: Yashu Dhatrika
"""

#CLA, CMB,CUA,CUB and Drugs pregnancy

import os
import numpy as np
import pandas as pd
import glob
from collections import Counter
import random 

os.chdir("C:/Users/Yashu Dhatrika/Desktop/handover files")

#%%
 #Joining all tables(#CLA, CMB,CUA,CUB,food frequency,Visit1,Vist2,Physical Activity,Screening ,Pregnancy outcome and Drugs pregnancy)
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

#%%
#print(len(dfs))


#%%
consol=pd.merge(unique1, dfs[0], how='left', on='STUDYID')
for i in range(1,len(dfs)):
        if i!=4 and i!=5 and i!=6:
            consol=pd.merge(consol, dfs[i], how='left', on='STUDYID')
        
#%% Taking the relevant features from drugs data and aggregating on studyID level
drugs_data=dfs[4]

sub_data=drugs_data[['StudyID','VXXC01e']]
sub_data.columns=['STUDYID','Reasoncode']
dummies = pd.get_dummies(sub_data[['Reasoncode']]).rename(columns=lambda x: 'Category_' + str(x))
subdata1 = pd.concat([sub_data, dummies], axis=1)
subdata2=subdata1.drop(['Reasoncode'],axis=1)
subdata3=subdata2.groupby('STUDYID',as_index=False)[['Category_Reasoncode_-8', 'Category_Reasoncode_aa1', 'Category_Reasoncode_ab1', 'Category_Reasoncode_ac1', 'Category_Reasoncode_ad1', 'Category_Reasoncode_ae1', 'Category_Reasoncode_af1', 'Category_Reasoncode_ag1', 'Category_Reasoncode_ah1', 'Category_Reasoncode_ai1', 'Category_Reasoncode_ak1', 'Category_Reasoncode_al1', 'Category_Reasoncode_am1', 'Category_Reasoncode_an1', 'Category_Reasoncode_ao1', 'Category_Reasoncode_ap1', 'Category_Reasoncode_aq1', 'Category_Reasoncode_ar1', 'Category_Reasoncode_as1', 'Category_Reasoncode_at1', 'Category_Reasoncode_au1', 'Category_Reasoncode_av1', 'Category_Reasoncode_aw1', 'Category_Reasoncode_ax1', 'Category_Reasoncode_ay1', 'Category_Reasoncode_az1', 'Category_Reasoncode_ba1', 'Category_Reasoncode_bb1', 'Category_Reasoncode_bc1', 'Category_Reasoncode_bc2', 'Category_Reasoncode_bc3', 'Category_Reasoncode_bd1', 'Category_Reasoncode_bd2', 'Category_Reasoncode_bd3', 'Category_Reasoncode_bd4', 'Category_Reasoncode_bd5', 'Category_Reasoncode_bd6', 'Category_Reasoncode_bd8', 'Category_Reasoncode_bd9', 'Category_Reasoncode_be1', 'Category_Reasoncode_bf1', 'Category_Reasoncode_bs1', 'Category_Reasoncode_ca1', 'Category_Reasoncode_ca2', 'Category_Reasoncode_cb1', 'Category_Reasoncode_cc1', 'Category_Reasoncode_cd1', 'Category_Reasoncode_ce1', 'Category_Reasoncode_cf1', 'Category_Reasoncode_cg1', 'Category_Reasoncode_ch1', 'Category_Reasoncode_ci1', 'Category_Reasoncode_d2a', 'Category_Reasoncode_da1', 'Category_Reasoncode_db1', 'Category_Reasoncode_db2', 'Category_Reasoncode_de1', 'Category_Reasoncode_df1', 'Category_Reasoncode_dz1']].sum()

master_data=pd.merge(consol, subdata3, how='left', on='STUDYID')

#%% Taking relevant features from food data

sel_fooddata=dfs[5].iloc[:,np.r_[0,21:74,699:736]]

master_data=pd.merge(master_data,sel_fooddata, how='left', on='STUDYID')

#%% Taking relevant features from the physical activity table

phys_act=dfs[6][['STUDYID','METs']]
phys_act=phys_act.groupby('STUDYID', as_index=False).agg({"METs": "mean"})
master_data=pd.merge(master_data,phys_act, how='left', on='STUDYID')
#%%
master_data.to_csv('master.csv')


