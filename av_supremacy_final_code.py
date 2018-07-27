#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 03:31:16 2018

@author: burhanusman
"""

#Importing the required Packages
import os 
os.chdir('/Users/burhanusman/Documents/Competitions/Data_supremacy')
import lightgbm as lgb
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder



#Function for label encoding the columns of a datafream
def label_encode_df(dataframe,cols):
    for col in cols:
        le=LabelEncoder()
        dataframe[col]=le.fit_transform(dataframe[col].astype(str))

#Function for target encoding categorical variables
#df_fit is the dataframe from where the target encoding variables are obtained
#df_tranform is the dataframe where the target encoded variables are imputed to
def target_encode_mean(df_fit,df_transform,col,target):
    group_mean=pd.DataFrame(df_fit.groupby([col])[target].mean())
    group_mean.columns=[col+"_"+target+"_mean"]
    group_mean.reset_index(inplace=True)
    df_transform=df_transform.merge(group_mean,how="left",on=[col])
    return df_transform
        
#Reading the train data
train = pd.read_csv("train.csv")

#Creating folds to out-of-fold predictions stacking
train["fold"]=0
i=1
for tr,ts in KFold(n_splits=5,shuffle=True,random_state=5).split(train):
    train.loc[list(ts),"fold"]=i
    i=i+1

#Reading the test data
test = pd.read_csv("test.csv")

#Stacking dataframe to hold the predicted outputs 
stack=pd.DataFrame()
stack["enrollee_id"]=train.enrollee_id
stack["fold"]=train.fold
stack["target"]=train.target

#Defining catboost models to be stacked(Only used one model in the final submisson)
model1={"model_name" : "CatBoost1", "n_estimators":540,"model_vars" :['city', 'gender',
       'relevent_experience', 'enrolled_university', 'education_level',
       'major_discipline', 'experience', 'company_size', 'company_type',
       'last_new_job', 'training_hours','NA_type'],"cat_vars" :12, "seed" :30}
models=[model1]

#Loop for iteratively training on 4folds and predicing on the 5th
#We obtain a dataframe where no. of columns = number of models used, and no. of rows = rows in train data
for model in models:
    stack[model["model_name"]]=0
    comb=pd.concat([train,test])
    comb.reset_index(inplace=True,drop=True)
    NA_cols=["company_size","company_type","education_level","enrolled_university","experience","gender",
    "last_new_job","major_discipline"]
    for col in NA_cols:
        comb["isna_"+col]=comb[col].isna().astype(int)
    comb["NA_type"]=''
    for col in NA_cols:
        comb["NA_type"]=comb["NA_type"].astype(str)+"_"+comb[col].isna().astype(int).astype(str)
    label_encode_df(comb,model["model_vars"][0:model["cat_vars"]])
    for col in model["model_vars"][0:model["cat_vars"]]:
        comb[col]=comb[col].astype(str)
    for i in range(1,6):
        print("Running Model " +model["model_name"]+" for fold "+str(i))
        comb["dataset"]="train"
        len_train=18359
        comb.loc[len_train:,"dataset"]="test"
        comb.loc[comb.fold==i,"dataset"]="valid"
        y=comb.loc[comb.dataset=="train","target"].values
        y_test=comb.loc[comb.dataset=="valid","target"].values
        x=comb.loc[comb.dataset=="train",model["model_vars"]].values
        x_test=comb.loc[comb.dataset=="valid",model["model_vars"]].values
        cat_model=CatBoostClassifier(eval_metric="AUC",n_estimators=model["n_estimators"],random_state=model["seed"])
        cat_model.fit(x,y,cat_features=list(range(0,model["cat_vars"])),verbose=False)
        stack.loc[stack.fold==i,model["model_name"]]=cat_model.predict_proba(comb.loc[comb.dataset=="valid",model["model_vars"]].values)[:,1]
        
#Training the above models on the full train dataset and predicting for the test data
#We obtain a dataframe where no. of columns = number of models used, and no. of rows = rows in test data
stack_test=pd.DataFrame()
stack_test["enrollee_id"]=test.enrollee_id
for model in models:
    stack_test[model["model_name"]]=0
    comb=pd.concat([train,test])
    comb.reset_index(inplace=True,drop=True)
    NA_cols=["company_size","company_type","education_level","enrolled_university","experience","gender",
    "last_new_job","major_discipline"]
    for col in NA_cols:
        comb["isna_"+col]=comb[col].isna().astype(int)
    comb["NA_type"]=''
    for col in NA_cols:
        comb["NA_type"]=comb["NA_type"].astype(str)+"_"+comb[col].isna().astype(int).astype(str)
    label_encode_df(comb,model["model_vars"][0:model["cat_vars"]])
    for col in model["model_vars"][0:model["cat_vars"]]:
        comb[col]=comb[col].astype(str)
    print("Running Model " + model["model_name"] + " on the test data")
    comb["dataset"]="train"
    len_train=18359
    comb.loc[len_train:,"dataset"]="test"
    y=comb.loc[comb.dataset=="train","target"].values
    x=comb.loc[comb.dataset=="train",model["model_vars"]].values
    cat_model=CatBoostClassifier(eval_metric="AUC",n_estimators=model["n_estimators"],random_state=model["seed"])
    cat_model.fit(x,y,cat_features=list(range(0,model["cat_vars"])),verbose=False)
    stack_test[model["model_name"]]=cat_model.predict_proba(comb.loc[comb.dataset=="test",model["model_vars"]].values)[:,1]
    

#Adding a LightGBM model to the stack 
stack["lgb_model"]=0
lgb_vars=['city', 'city_development_index', 'company_size', 'company_type',
       'education_level', 'enrolled_university', 'experience',
       'gender', 'last_new_job', 'major_discipline','relevent_experience',
       'training_hours',
       'city_target_mean',
        'enrolled_university_target_mean',
       'education_level_target_mean', 'major_discipline_target_mean',
       'experience_target_mean',
        'last_new_job_target_mean',
        'NA_type_target_mean']
cat_vars=['city', 'company_size', 'company_type',
       'education_level', 'enrolled_university',
       'gender', 'last_new_job', 'major_discipline', 'relevent_experience',      
        'experience','NA_type','training_hours']

#LightGBM
for i in range(1,6):
    print("Running Model LGBM for fold"+str(i))
    comb=pd.concat([train,test])
    comb.reset_index(inplace=True,drop=True)
    NA_cols=["company_size","company_type","education_level","enrolled_university","experience","gender",
    "last_new_job","major_discipline"]
    for col in NA_cols:
        comb["isna_"+col]=comb[col].isna().astype(int)
    comb["NA_type"]=''
    for col in NA_cols:
        comb["NA_type"]=comb["NA_type"].astype(str)+"_"+comb[col].isna().astype(int).astype(str)
    comb["dataset"]="train"
    len_train=18359
    comb.loc[len_train:,"dataset"]="test"
    comb.loc[comb.fold==i,"dataset"]="valid"
    for col in cat_vars:
            comb=target_encode_mean(comb[comb.dataset=="train"],comb,col,"target")
    label_encode_df(comb,cat_vars)
    y=comb.loc[comb.dataset=="train","target"].values
    y_test=comb.loc[comb.dataset=="valid","target"].values
    x=comb.loc[comb.dataset=="train",lgb_vars].values
    x_test=comb.loc[comb.dataset=="valid",lgb_vars].values
    lgbtrain=lgb.Dataset(x,y,feature_name=lgb_vars,free_raw_data=True)
    lgbvalid=lgb.Dataset(x_test,y_test,feature_name=lgb_vars,free_raw_data=True)
    #LGB Model
    parameters = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting': 'gbdt',
        'num_leaves': 200,
        'feature_fraction': 0.65,
        'bagging_fraction': 0.85,
        #'bagging_freq': 20,
        'learning_rate': 0.1,
        'verbose': 1,
        'max_bin' :10,
        'min_gain_to_split' : 1,
        'seed': 0
    }
    lgb_model=lgb.train(parameters,lgbtrain,num_boost_round=20,)
    stack.loc[stack.fold==i,"lgb_model"]=lgb_model.predict(comb.loc[comb.dataset=="valid",lgb_vars].values)

#LightGBM predictions for the test data
stack_test["lgb_model"]=0
print("Running LightGBM on the test data")
comb=pd.concat([train,test])
comb.reset_index(inplace=True,drop=True)
NA_cols=["company_size","company_type","education_level","enrolled_university","experience","gender",
"last_new_job","major_discipline"]
for col in NA_cols:
    comb["isna_"+col]=comb[col].isna().astype(int)
comb["NA_type"]=''
for col in NA_cols:
    comb["NA_type"]=comb["NA_type"].astype(str)+"_"+comb[col].isna().astype(int).astype(str)
comb["dataset"]="train"
len_train=18359
comb.loc[len_train:,"dataset"]="test"
for col in cat_vars:
        comb=target_encode_mean(comb[comb.dataset=="train"],comb,col,"target")
label_encode_df(comb,cat_vars)
y=comb.loc[comb.dataset=="train","target"].values
y_test=comb.loc[comb.dataset=="valid","target"].values
x=comb.loc[comb.dataset=="train",lgb_vars].values
x_test=comb.loc[comb.dataset=="valid",lgb_vars].values
lgbtrain=lgb.Dataset(x,y,feature_name=lgb_vars,free_raw_data=True)
lgbvalid=lgb.Dataset(x_test,y_test,feature_name=lgb_vars,free_raw_data=True)
parameters = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting': 'gbdt',
    'num_leaves': 200,
    'feature_fraction': 0.65,
    'bagging_fraction': 0.85,
    #'bagging_freq': 20,
    'learning_rate': 0.1,
    'verbose': 1,
    'max_bin' :10,
    'min_gain_to_split' : 1,
    'seed': 0
}
lgb_model=lgb.train(parameters,lgbtrain,num_boost_round=20)
stack_test["lgb_model"]=lgb_model.predict(comb.loc[comb.dataset=="test",lgb_vars].values)

#Fitting a meta-model over the stacked predictions and making the final predictions
lr_model=LogisticRegression()
lr_model.fit(X=stack[["CatBoost1","lgb_model"]],y=stack.target)
stack_test["target"]=lr_model.predict_proba(X=stack_test[["CatBoost1","lgb_model"]])[:,1]
stack_test[["enrollee_id","target"]].to_csv("sub_final.csv",index=False)


