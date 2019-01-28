
# coding: utf-8

# In[87]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import random
import math
import statistics

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,roc_auc_score,auc,roc_curve

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier)
import lightgbm as lgb


# # Data input

# In[88]:


train=pd.read_csv('uconn_comp_2018_train.csv')
test=pd.read_csv('uconn_comp_2018_test.csv')


# In[89]:


all_data = pd.concat((train, test)).reset_index(drop=True)
print("Total data size：",all_data.shape)


# # Explore missing data

# In[90]:


print('***************************EDA*****************************')
print('Checking missing value of categorical data')
col=['gender','marital_status','high_education_ind','address_change_ind',
     'living_status','claim_day_of_week','accident_site','witness_present_ind',
     'channel','policy_report_filed_ind','vehicle_category','vehicle_color'
    ]
for i in col:
    l=list(set(all_data[i]))
    print('Unique value of column:%s'%i,l)


# In[91]:


print('Fill in missing value case by case...')
## fill marital_status with random choice 
all_data['m']=list(np.random.choice([0, 1], size=(len(all_data),), p=[1-np.mean(all_data['marital_status']), np.mean(all_data['marital_status'])]))
all_data['marital_status'].fillna(all_data['m'],inplace=True)
all_data.drop('m',axis=1,inplace=True)

## fill witness_present_ind with random choice
all_data['w']=list(np.random.choice([0,1],size=(len(all_data),),p=[1-np.mean(all_data['witness_present_ind']),np.mean(all_data['witness_present_ind'])]))
all_data['witness_present_ind'].fillna(all_data['w'],inplace=True)
all_data.drop('w',axis=1,inplace=True)

## drop fraud==-1
all_data=all_data[all_data['fraud']!=-1]

## fill claim_est_payout with mean
all_data['claim_est_payout'].fillna(np.mean(all_data['claim_est_payout']),inplace=True)

## fill age_of_vehicle with median since the distribution is left skewed
all_data['age_of_vehicle'].fillna(statistics.median(all_data['age_of_vehicle']),inplace=True)


# # Convert outlier

# In[92]:


# convert outliers of age_of_drivers >=100
all_data.loc[all_data['age_of_driver']>100,'age_of_driver']=statistics.median(all_data['age_of_driver'])


# # feature engineering

# In[93]:


print('***************************Start feature engineering*****************************')

# Convert zipcode to geocoding info
zip2loc = pd.read_csv("zip2loc.csv")
all_data = pd.merge(all_data, zip2loc[["Zipcode", "Lat", "Long", "State",'City','EstimatedPopulation','TotalWages','TaxReturnsFiled']], left_on = 'zip_code', right_on = "Zipcode", how = 'left')
all_data = all_data.drop(["zip_code"], axis = 1)
all_data = all_data.rename(columns = {"Lat" : "latitude", "Long" : "longitude", "State" : "state"})
all_data['state']=pd.Categorical(all_data['state']).codes
all_data['City']=pd.Categorical(all_data['City']).codes

# filling missing zipcode with random choice of 5 cluster center
t=all_data[(all_data['fraud'].isnull().values==False)& (all_data['Zipcode'].isnull().values==False)][['latitude','longitude']]
kmeans = KMeans(n_clusters=5, random_state=0).fit(t)
center=kmeans.cluster_centers_

mask=all_data['Zipcode'].isnull().values==True
all_data.loc[mask,'latitude']=all_data['latitude'].apply(lambda row: random.sample(center[:,0].tolist(),1)[0])
all_data.loc[mask,'longitude']=all_data['longitude'].apply(lambda row: random.sample(center[:,1].tolist(),1)[0])
all_data['latitude']=all_data['latitude'].apply(pd.to_numeric, errors='coerce')
all_data['longitude']=all_data['longitude'].apply(pd.to_numeric, errors='coerce')
# change claim_date from date type to month, day, year
all_data['claim_date']=pd.to_datetime(all_data['claim_date'])
all_data['month']=all_data['claim_date'].dt.month
all_data['day']=all_data['claim_date'].dt.day
all_data['year']=all_data['claim_date'].dt.year

# create age group
all_data.loc[:,'age_group'] = all_data['age_of_driver'].apply(lambda x: math.floor(x/10))
all_data['age_group'] = pd.Categorical(all_data['age_group']).codes

#create gender marriage the value correspond to the prob of fraud
m0 = (all_data['gender'] == 'M') & (all_data['marital_status'] == 0)
m1 = (all_data['gender'] == 'M') & (all_data['marital_status'] == 1)
f0 = (all_data['gender'] == 'F') & (all_data['marital_status'] == 0)
f1 = (all_data['gender'] == 'F') & (all_data['marital_status'] == 1)
all_data.loc[m0,'gender_marrige'] = 2
all_data.loc[m1,'gender_marrige'] = 4
all_data.loc[f0,'gender_marrige'] = 3
all_data.loc[f1,'gender_marrige'] = 5
group = all_data.groupby('gender_marrige')['fraud'].sum()/len(all_data)
group2 = all_data.groupby('marital_status')['fraud'].sum()
#all_data = all_data.drop(['gender','marital_status'], axis =1 )

#safty_rating, if safty rating > 30, low prob of fraud
all_data.loc[all_data['safty_rating']<30, 'safty_rating_new'] = 1
all_data.loc[all_data['safty_rating']>=30, 'safty_rating_new'] = 0
#all_data = all_data.drop('safty_rating', axis = 1)

#not sure price/income
all_data.loc[:,'price/income'] = all_data['vehicle_price']/all_data['annual_income']
#all_data = all_data.drop(['vehicle_price','annual_income'],axis=1)

#claim/income
all_data.loc[:,'claim/income'] = all_data['claim_est_payout']/all_data['annual_income']
#all_data = all_data.drop(['claim_est_payout','annual_income'],axis=1)

#claim*liab,
all_data.loc[:,'claim*liab'] = all_data['claim_est_payout']*all_data['liab_prct']
#all_data = all_data.drop(['claim_est_payout','liab_prct'],axis=1)

#claim/income/liab
all_data.loc[:,'claim/income/liab'] = all_data['claim_est_payout']/(all_data['annual_income']*all_data['liab_prct'])
#all_data = all_data.drop(['claim_est_payout','liab_prct','annual_income'],axis=1)

#lat*long
all_data['lat*long']=all_data['latitude']*all_data['longitude']
print('**************************New features are finished*****************************')

print('**************************Change categorical data to numeric data**************************')
categorical_features=['gender','living_status','claim_day_of_week','accident_site','gender_marrige','safty_rating_new',
                      'channel','vehicle_category','vehicle_color','past_num_of_claims']
for f in categorical_features:
    all_data = pd.concat([all_data, pd.get_dummies(all_data[f], prefix = f)], axis=1)
all_data = all_data.drop(categorical_features, axis = 1)


# In[94]:


print('***************************Split train and test dataset and reset index*****************************')
train = all_data[all_data['fraud'].isnull().values==False]
test = all_data[all_data['fraud'].isnull().values==True]

train=train.reset_index()
test=test.reset_index()

train.to_csv('train_final.csv',index=False)
test.to_csv('test_final.csv',index=False)


# ## Modeling

# In[96]:


print('***************************Select features for modeling*****************************')
train=pd.read_csv('train_final.csv')
test=pd.read_csv('test_final.csv')

variable_names = list(train)
do_not_use_for_training = ['index','claim_date',
                         'claim_number',
                         'fraud',
                         'Zipcode',
                         'state',
                         'City',
                         'EstimatedPopulation',
                         'TotalWages',
                         'TaxReturnsFiled',
                          'claim/income/liab']

feature_names = [f for f in variable_names if f not in do_not_use_for_training]
print('features selected are:',feature_names)
print('***************************Data standardization*****************************')
X_train1 = train[feature_names]
Y_train1 = train['fraud']

X_train,X_val,Y_train,Y_val = train_test_split(X_train1, Y_train1,test_size = 0.2,random_state=10)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_val_std=sc.transform(X_val)

X_test = test[feature_names]
X_test_std=sc.transform(X_test)
print('Data preprossessing is finished. Now start to select models.')


# ## Xgboost

# In[104]:


print('***************************Start model*****************************')
print('***************************use grid search to choose best parameters for each model *****************************')
print('Model 1: Xgboost')

param_test = {'n_estimators':range(50,200,50),
             'max_depth':range(2,5,1),
             'min_child_weight':range(0,2,1),
             'subsample':[i/100.0 for i in range(75,90,5)],
             'colsample_bytree':[i/100.0 for i in range(75,90,5)],
             'reg_alpha':[0, 0.001, 0.01],
             'learning_rate':[0, 0.001, 0.01,0.1,0.2]}
clf = GridSearchCV(estimator = xgb.XGBClassifier(gamma=0,scale_pos_weight=1,objective= 'binary:logistic',silent=1,booster='gbtree', nthread=4, seed=27),
                   param_grid = param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

best_xgb = clf.fit(X_train_std, Y_train)
print('Best parameters of Xgboost:')
print('Best n_estimators:', best_xgb.best_estimator_.get_params()['n_estimators'])
print('Best max_depth:', best_xgb.best_estimator_.get_params()['max_depth'])
print('Best min_child_weight:', best_xgb.best_estimator_.get_params()['min_child_weight'])
print('Best subsample:', best_xgb.best_estimator_.get_params()['subsample'])
print('Best colsample_bytree:', best_xgb.best_estimator_.get_params()['colsample_bytree'])
print('Best reg_alpha:', best_xgb.best_estimator_.get_params()['reg_alpha'])
print('Best learning_rate:', best_xgb.best_estimator_.get_params()['learning_rate'])
print('Best score of Xgboost:', best_xgb.best_score_)


# ## GradientBoosting

# In[ ]:


print('Model 2: Gradient Boosting')

# use grid search choose best parameters 
param_test = {'n_estimators':range(50,150,10),
             'min_samples_split':range(100,600,100),
             'min_samples_leaf':range(20,100,10),
             'learning_rate':[0.01,0.05,0.08,0.1]}

clf= GridSearchCV(GradientBoostingClassifier(max_features='sqrt',subsample=0.8,random_state=10), 
                                      param_test, cv=5, n_jobs=4,scoring='roc_auc',iid=False)

best_gbc=clf.fit(X_train_std,Y_train)
print('Best parameters of Gradient Boosting:')
print('Best n_estimators:', best_gbc.best_estimator_.get_params()['n_estimators'])
print('Best min_samples_split:', best_gbc.best_estimator_.get_params()['min_samples_split'])
print('Best min_samples_leaf:', best_gbc.best_estimator_.get_params()['min_samples_leaf'])
print('Best learning_rate:', best_gbc.best_estimator_.get_params()['learning_rate'])
print('Best score of Gradient Boosting:', best_gbc.best_score_)


# ## Lightgbm

# In[ ]:


print('Model 3: Lightgbm')
# use grid search choose best parameters 
param_test= {'num_leaves':range(2,6,1),
       'learning_rate':[0.001,0.01,0.05,0.1,0.2]}

clf = GridSearchCV(lgb.LGBMClassifier(application='binary',objective='binary',metric='auc',
                                     is_unbalance=True,boosting='gbdt',verbose=0
                                     ),param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

best_lgb = clf.fit(X_train_std, Y_train)
print('Best parameters of Lightgbm:')
print('Best num_leaves:', best_lgb.best_estimator_.get_params()['num_leaves'])
print('Best learning_rate:', best_lgb.best_estimator_.get_params()['learning_rate'])
print('Best score of Lightgbm:', best_lgb.best_score_)


# ## Ensemble tree based models

# In[101]:


print('***************************Ensemble tree based models*****************************')
print('***************************Select best weight for each model utilizing grid search*****************************')

w1=[i/100 for i in range(0,50,1)]
w2=[i/100 for i in range(50,100,1)]


score=pd.DataFrame(columns=['xgb_wt','gb_wt','lgb_wt','ensembled_score'])

base_model=[xgb.XGBClassifier(n_estimators=100,
                            learning_rate =0.1,
                            max_depth=3,
                            min_child_weight=1,
                            gamma=0,
                            subsample=0.8,
                            colsample_bytree=0.75,
                            objective= 'binary:logistic',
                            nthread=12,
                            scale_pos_weight=1,
                            reg_alpha=0.01,
                            seed=27),
            GradientBoostingClassifier(n_estimators=100,
                               learning_rate=0.1,
                               max_depth=3,
                               max_features='sqrt',
                               min_samples_split=300,
                               min_samples_leaf=40,
                               subsample=0.8,
                               random_state=10),
            lgb.LGBMClassifier(application='binary',objective='binary',metric='auc',is_unbalance=True,boosting='gbdt',
                           num_leaves=3,learning_rate=0.2,verbose=0)]

x_train,x_test,y_train,y_test = train_test_split(X_train_std, Y_train,test_size = 0.2,random_state = 10)
clf_xgb=base_model[0].fit(x_train,y_train)
clf_gb=base_model[1].fit(x_train,y_train)
clf_lgb=base_model[2].fit(x_train,y_train)

fpr,tpr,thresholds=roc_curve(Y_val,clf_xgb.predict_proba(X_val_std)[:,1])
xgb_score=auc(fpr,tpr)
fpr,tpr,thresholds=roc_curve(Y_val,clf_gb.predict_proba(X_val_std)[:,1])
gb_score=auc(fpr,tpr)
fpr,tpr,thresholds=roc_curve(Y_val,clf_lgb.predict_proba(X_val_std)[:,1])
lgb_score=auc(fpr,tpr)
print('Test AUC score of model Xgboost is:%f'%xgb_score)
print('Test AUC score of model Gradient boosting is:%f'%gb_score)
print('Test AUC score of model Lightgbm is:%f'%lgb_score)

fpr,tpr,thresholds=roc_curve(y_test,clf_xgb.predict_proba(x_test)[:,1])
xgb_score=auc(fpr,tpr)
fpr,tpr,thresholds=roc_curve(y_test,clf_gb.predict_proba(x_test)[:,1])
gb_score=auc(fpr,tpr)
fpr,tpr,thresholds=roc_curve(y_test,clf_lgb.predict_proba(x_test)[:,1])
lgb_score=auc(fpr,tpr)
print('Val AUC score of model Xgboost is:%f'%xgb_score)
print('Val AUC score of model Gradient boosting is:%f'%gb_score)
print('Val AUC score of model Lightgbm is:%f'%lgb_score)

print('***************************Now ensemble Xgboost, Gradientboosting, Lightgbm models.***************************')

index=0
for i in w1:
    for j in w2:
        w3=1-i-j
        if w3>=0:
            score.loc[index,'xgb_wt']=i
            score.loc[index,'gb_wt']=w3
            score.loc[index,'lgb_wt']=j
            test_pred=i*clf_xgb.predict_proba(X_val_std)[:,1]+j*clf_gb.predict_proba(X_val_std)[:,1]+w3*clf_lgb.predict_proba(X_val_std)[:,1]
            fpr,tpr,thresholds=roc_curve(Y_val,test_pred)
            score.loc[index,'ensembled_score']=auc(fpr,tpr)
            score.loc[index,'type']='test_score'
            index+=1
            score.loc[index,'xgb_wt']=i
            score.loc[index,'gb_wt']=w3
            score.loc[index,'lgb_wt']=j
            val_pred=i*clf_xgb.predict_proba(x_test)[:,1]+j*clf_gb.predict_proba(x_test)[:,1]+w3*clf_lgb.predict_proba(x_test)[:,1]
            fpr,tpr,thresholds=roc_curve(y_test,val_pred)
            score.loc[index,'ensembled_score']=auc(fpr,tpr)
            score.loc[index,'type']='val_score'
            index+=1
            
print('Best test AUC score of ensembled model:%f'%score[score.type=='test_score']['ensembled_score'].max())
print('Best val AUC score of ensembled model:%f'%score[score.type=='val_score']['ensembled_score'].max())


# # output result

# In[103]:


print('***************************Predict test dataset using three tree models***************************')
y_test_pred_xgb=clf_xgb.predict_proba(X_test_std)[:,1]
y_test_pred_gb=clf_gb.predict_proba(X_test_std)[:,1]
y_test_pred_lgb=clf_lgb.predict_proba(X_test_std)[:,1]
y_test_pred=0.14*y_test_pred_xgb+0.36*y_test_pred_gb+0.5*y_test_pred_lgb
result=pd.DataFrame()
result['claim_number']=test['claim_number']
result['fraud']=y_test_pred.tolist()
result.to_csv('Team_WOT_prediction_final.csv', index = False)
print('Output is finished.')

