# -*- coding: utf-8 -*-
from pandas import Series, DataFrame
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict

#predicting airbnb new users' conversion within 60 days after signing up
#data source: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data


########################load original datasets################################
user = pd.read_csv("train_users_2.csv")
user.head()

#convert time columns to datetime/timestamp types 
user['date_first_booking'] = pd.to_datetime(user['date_first_booking'])
user['date_account_created'] = pd.to_datetime(user['date_account_created'])
user['timestamp_first_active'] = pd.to_datetime(user['timestamp_first_active'], format = '%Y%m%d%H%M%S', errors = 'coerce')

#####################Exploratory data analysis#################################
#inspect data 
user.describe()
user['gender'].value_counts()  
user['signup_method'].value_counts()
user['signup_flow'].value_counts()
user['language'].value_counts()
user['first_affiliate_tracked'].value_counts()
user['first_device_type'].value_counts()
user['first_browser'].value_counts()
### age has values that are out of a reasonable range
###there are missing values labeled as'-unknown-','untracked','Other/Unknown'
###there are rare levels in 'signup_flow', 'language', 'first_browser'

#age
hist(user.age.dropna())
hist(user.age.values, 20, range = [0,100])

#date lag of first booking after account created
user_booked = user[user.date_first_booking.isnull() == False]
user['book_lag'] = (user_booked.date_first_booking - user_booked.date_account_created).dt.days
plt.hist(user.book_lag.dropna())
np.percentile(user.book_lag.dropna(), 80)
np.percentile(user.book_lag.dropna(), 60)
### 60% users who booked book within 6 days after created account; 
### 80% users who booked book within 58 days after created account; 
### we need to define the target variable (book or not) with a time range
### based on the exploratory analysis, let's predict whether a customer will book within 60 days after creating account


#date lag of account created after first active 
#signup_lag = (user.date_account_created - pd.to_datetime(user.timestamp_first_active.dt.date)).dt.days
#signup_lag.describe()
#np.percentile(signup_lag, 90)
### for users who signed up, over 99% signed up in the same day of fist active 


######################## data cleaning #######################################
#missing values
##replacing missing values with NaN
user = user.replace(['-unknown-','untracked','Other/Unknown'], np.nan)
##percentage of missing values in each column 
(user.isnull().sum() / user.shape[0]) * 100
##assuming make missing values in gender, age, and first_affiliate_tracked are 
##because users didn't provide those information or didn't have a first affiliate tracked
##convert them into be another level 
user['age_missing'] = np.where(user['age'].isnull(),1,0) 
user['gender'].replace(np.nan, 'missing', inplace = True)
user['first_affiliate_tracked'].replace(np.nan, 'missing',inplace = True)

#outliers 
##replacing outliers in age with NaN
user['age'] = np.where(np.logical_or(user['age']>90, user['age']<5), np.nan, user['age']) 

#######################Feature engineering###################################
#time related features 
user['dac_year'] = user.date_account_created.apply(lambda x: str(x.year))
user['dac_month'] = user.date_account_created.apply(lambda x: str(x.month))
user['dac_dfw'] = user.date_account_created.apply(lambda x: str(x.dayofweek))


#categorical variables
user['signup_flow'] = user['signup_flow'].astype(str)

cate_feats = ['dac_year', 'dac_month', 'dac_dfw', 'gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 
'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']



##combine rare levels in categorical variables as 'others'
def combine_rare_level(min_pct, f):
    df = DataFrame(user[f].value_counts())
    df['pct'] = df[f]/user.shape[0]
    rare_level = df[df.pct < min_pct].index
    for l in rare_level:
        user[f] = np.where(user[f] == l, 'other', user[f])        
for f in cate_feats:
    combine_rare_level(0.01, f)
    
##One-hot-encoding features for categorical variables
for f in cate_feats:
    user_dummy = pd.get_dummies(user[f], prefix=f)
    user = user.drop([f], axis=1)
    user = pd.concat((user, user_dummy), axis=1)


#continuous variables
##discretize continues variable 'age' 
##One-hot encoding of the buckets
###input desired upper limit of each buckets in ascending order
interv =  [15, 20, 25, 30, 35, 40, 60, 100]
def get_interv_value(age):
    iv = 15
    for i in range(len(interv)):
        if age < interv[i]:
            iv = i 
            break
    return iv
user['age_interv'] = user.age.apply(lambda x: get_interv_value(x))
age_dummy = pd.get_dummies(user.age_interv, prefix='age_interv')
user = user.drop(['age_interv'], axis=1)
user = pd.concat((user, age_dummy), axis=1)



#########################Modeling preparation#################################
#create target variable
user['book'] = np.where( user['book_lag'] <= 60, 1, 0)

#the data contains users first active before 12/5/2015, remove users who had not created account for 60 days
user = user[user.date_account_created < '2015-10-05']

#count of two categories of the target variable
user.book.value_counts()
### the target variable is unbalanced: 1/3 booked, 2/3 not booked 
###give higher weight to class 1: 142089.0/71326.0 = 1.99 instead of 1
        
#split features and targets 
X = user.drop(['book','id','date_account_created','timestamp_first_active',
'date_first_booking','country_destination','book_lag','age'],axis =1)
Y = user.book

#Spliting data into training, testing set 
np.random.seed(0)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.2, stratify =Y)


#write processed data to csv for building discriptive models in R
#processed = pd.concat([X,Y], axis = 1)
#processed.to_csv("processed.csv")

################# Logistic regression with L2 Regularization ###################
# create an logistic classifier instance 
clf = linear_model.LogisticRegression(penalty = 'l2', solver = 'liblinear', C=1e5, class_weight = {1:1.99, 0:1}, random_state = 0)
## c is the inverse of regularization strength, smaller values specify stronger regularization.
## solver parameter options:{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}

# Train the model, select model based on cross-validation performance
scores = cross_validate(clf, Xtrain, Ytrain, scoring= ['f1','roc_auc'],
                         cv=10, return_train_score=False)
sorted(scores.keys())
print("F1: %0.2f (+/- %0.2f)" % (scores['test_f1'].mean(), scores['test_f1'].std() * 2)) #0.55
print("AUC: %0.2f (+/- %0.2f)" % (scores['test_roc_auc'].mean(), scores['test_roc_auc'].std() * 2)) #0.70

# generalization performance on test set
predicted_clf = cross_val_predict(clf, Xtest, Ytest, cv=10)
report_clf = classification_report(Ytest, predicted_clf)
print(report_clf)



############ Logistic regression through recursive feature selection ###########

rfecv = RFECV(clf, step=2, cv=StratifiedKFold(2), scoring='roc_auc')
rfecv = rfecv.fit(Xtrain, Ytrain)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# Reduce X to the selected features and then return the score of the underlying estimator.
rfecv.score(Xtrain, Ytrain)  #nb of correct classification: 0.69

# Selected features
feature_names = Xtrain.columns.values
selected_f = feature_names[rfecv.get_support()]
print selected_f

# generalization performance on test set
predicted_rfecv = rfecv.predict(Xtest)
report_rfecv = classification_report(Ytest, predicted_rfecv)
print(report_rfecv)


####################### Random Forest #########################################
# create an random forest classifier instance
rfc = RandomForestClassifier(n_estimators = 10, max_depth=15, min_samples_split = 20, 
max_features = 50, random_state=0,class_weight = {1:1.99,0:1})

# Train the model, select model based on cross-validation performance
scores_rfc = cross_validate(rfc, Xtrain, Ytrain, scoring= ['f1','roc_auc'],
                            cv=5, return_train_score=False)

print("F1: %0.2f (+/- %0.2f)" % (scores_rfc['test_f1'].mean(), scores_rfc['test_f1'].std() * 2)) #0.58
print("AUC: %0.2f (+/- %0.2f)" % (scores_rfc['test_roc_auc'].mean(), scores_rfc['test_roc_auc'].std() * 2)) #0.76

# generalization performance on test set
predicted_rfc = cross_val_predict(rfc, Xtest, Ytest, cv=10)
report_rfc = classification_report(Ytest, predicted_rfc)
print(report_rfc) 


########################## Gradiant Boosting Tree ############################
grd = GradientBoostingClassifier(n_estimators = 20, learning_rate=0.1, 
      max_depth=5, min_samples_split=10, max_leaf_nodes = 30, min_impurity_decrease=0.01, random_state=0)
## no parameter for class weight yet (use over sampling instead)

# Train the model, select model based on cross-validation performance
scores_grd = cross_validate(grd, Xtrain, Ytrain, scoring= ['f1','roc_auc'],
                            cv=2, return_train_score=False)
                            
print("F1: %0.2f (+/- %0.2f)" % (scores_rfc['test_f1'].mean(), scores_grd['test_f1'].std() * 2)) #0.58
print("AUC: %0.2f (+/- %0.2f)" % (scores_rfc['test_roc_auc'].mean(), scores_grd['test_roc_auc'].std() * 2)) #0.76

# generalization performance on test set
predicted_grd = cross_val_predict(grd, Xtest, Ytest, cv=10)
report_grd = classification_report(Ytest, predicted_grd)
print(report_grd) 


########################## Naive Bayes ########################################
# create an random forest classifier instance
nb = BernoulliNB()

# Train the model, select model based on cross-validation performance
scores_nb = cross_validate(nb, Xtrain, Ytrain, scoring= ['f1','roc_auc'],
                            cv=10, return_train_score=False)

print("F1: %0.2f (+/- %0.2f)" % (scores_nb['test_f1'].mean(), scores_nb['test_f1'].std() * 2)) #0.52
print("AUC: %0.2f (+/- %0.2f)" % (scores_nb['test_roc_auc'].mean(), scores_nb['test_roc_auc'].std() * 2)) #0.66


# generalization performance on test set
predicted_nb = cross_val_predict(nb, Xtest, Ytest, cv=10)
report_nb = classification_report(Ytest, predicted_nb)
print(report_nb)


########################### Business Impact ###################################
best_prediction = predicted_rfc

#conversion rate of two segments
df_results = DataFrame({'pred':best_prediction, 'actual':Ytest.values})
seg_book = df_results[df_results.pred == 1]
seg_nbook = df_results[df_results.pred == 0]
conv_rate_seg_book = float(seg_book[seg_book.actual == 1].shape[0])/float(seg_book.shape[0])
conv_rate_seg_nbook = float(seg_nbook[seg_nbook.actual == 1].shape[0])/float(seg_nbook.shape[0])

print ('The conversion rate of the predicted booking segment: {:04.2f}%').format(conv_rate_seg_book * 100)
print ('The conversion rate of the predicted non-booking segment: {:04.2f}%').format(conv_rate_seg_nbook * 100)
print ('The conversion rate of the predicted booking segment is {:03.2f} times of the predicted non-booking segment').format(conv_rate_seg_book/conv_rate_seg_nbook)
## the users predicted as non-booking could be targeted with with marketing activities such as coupon to increase their conversion rate
## the users can be separated into more segments based on their ranking of propensity to convert to achieve more sophisticated segmentation and targeting
## to be continued: adding more user activity features into the model
