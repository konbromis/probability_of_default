# -*- coding: utf-8 -*-
"""

@author: Konstantinos Bromis
"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import dask.dataframe as dd
import time

#path = "C:\\Users\\konstantinos\\Desktop\\Project_Antonis\\mortgages3.csv"
path = "E:\\project\\mortgages3.csv"

# Convert csv file to Dataframe
mortgages = pd.read_csv(path, dtype={'Cod_BR_Del': 'float64', 
                               'Rate2_Type':'object', 
                               'Rate3_Type': 'object', 
                               'Addr_Tx': 'object', 
                               'Cust_Afm': 'object'}, 
                               engine='python', error_bad_lines=False)

# Calculate Months in arrears (Num_Month_End_Del)
delq1 = mortgages['Num_Day_End_Del'] == 0
delq2 = mortgages['Num_Day_End_Del'] <=30
delq3 = mortgages['Num_Day_End_Del'] <=60
delq4 = mortgages['Num_Day_End_Del'] <=90
delq5 = mortgages['Num_Day_End_Del'] <=120
delq6 = mortgages['Num_Day_End_Del'] <=150
delq7 = mortgages['Num_Day_End_Del'] <=180
delq8 = mortgages['Num_Day_End_Del'] >=181

mortgages['Num_Month_End_Del'] = np.select([delq1, delq2, delq3, delq4, delq5, delq6, delq7, delq8], [0, 1, 2, 3, 4, 5, 6, 7], default=0)

""" Independent Variables"""

# Transform Dates
from datetime import datetime
mortgages['Date_Proc'] = list(map(lambda x: datetime.strptime(x,'%d%b%Y:00:00:00.000').date(), mortgages['Date_Proc']))
mortgages['Date_Open'] = list(map(lambda x: datetime.strptime(x,'%d%b%Y:00:00:00.000').date(), mortgages['Date_Open']))
mortgages['Date_Birth'] = list(map(lambda x: datetime.strptime(x,'%d%b%Y:00:00:00.000').date(), mortgages['Date_Birth']))
mortgages['Date_End_New'] = list(map(lambda x: datetime.strptime(x,'%d%b%Y:00:00:00.000').date(), mortgages['Date_End_New']))

# Calculate Customers' Age
mortgages['Age_Customer'] = mortgages['Date_Proc'] - mortgages['Date_Birth']
mortgages['Age_Customer']=mortgages['Age_Customer']/np.timedelta64(1,'Y')

# Create bins for the Customers' Age
bins = [0, 20, 30, 40, 50, 60, 70, 80, 120]
group_names = ['0-20', '20-30', '30-39', '40-49', '50-59', '60-69', '70-79', '80-120']
mortgages['Age_Customer'] = pd.cut(x = mortgages['Age_Customer'], bins = bins, labels = group_names)


# Trasform postcode to certain bins
mortgages['Addr_Tx'] = mortgages['Addr_Tx'].astype(float)
cut_labels = ['Other1', 'Athens', 'Other2', 'Thess', 'Other3']
cut_bins = [0, 10430, 19600, 53437, 57500, 85700]
mortgages['Post_Code'] = pd.cut(x = mortgages['Addr_Tx'], bins = cut_bins, labels = cut_labels)
mortgages['Post_Code'] = mortgages['Post_Code'].replace({'Other1':'Other', 'Other2': 'Other', 'Other3': 'Other'})

""" Create Behavioural Independent variables"""

# Calculate Age of Loan
mortgages['Age_of_Loan'] = mortgages['Date_Proc'] - mortgages['Date_Open']
mortgages['Age_of_Loan']=mortgages['Age_of_Loan']/np.timedelta64(1,'M')

# Calculate months since expiring
mortgages['Maturity'] = mortgages['Date_End_New'] - mortgages['Date_Proc']
mortgages['Maturity']=mortgages['Maturity']/np.timedelta64(1,'M')

# Calculate sum of last year's months that exceed 30 days of targets from previous year of each month"""
mortgages['Months_over_30_days'] = np.multiply(list(map(lambda x : x > 30, mortgages['Num_Day_End_Del'])), 1)
mortgages['12_month_over_30'] = mortgages.groupby('Account')['Months_over_30_days'].apply(lambda x: x.rolling(min_periods=0, window=12).sum())

# Create behavioural variable BalPctInitAmt = Balance / Initial Ammount (Amnt_Ledger / Amnt_Egr)
mortgages['BalPctInitAmt'] = mortgages['Amnt_Ledger']/mortgages['Amnt_Egr']

# Create behavioural variable MaxConsMDelqGT1 (Maximun consecutive months with delinquency>1 month (30 days))
from functions import getMaxLength
mortgages['MaxConsMDelqGT1'] = mortgages.groupby('Account')['Months_over_30_days'].apply(lambda x: x.rolling(min_periods=0, window=12).apply(lambda x: getMaxLength(x))) 

# Maximum delinquency in the last 12 months (MaxDelqL12M)
mortgages['MaxDelqL12M'] = mortgages.groupby('Account')['Num_Month_End_Del'].apply(lambda x: x.rolling(min_periods=0, window=12).max())

# Consecutive months without changes in Delinquency (ConsNoChgDelq)
from functions import getconsecmaxnochng
mortgages['ConsNoChgDelq'] = mortgages.groupby('Account')['Num_Month_End_Del'].apply(lambda x: x.rolling(min_periods=0, window=12).apply(lambda x: getconsecmaxnochng(x))) 

""" Create Dependent Variable"""

# Create target labels
Amnt_Over_Ninety = mortgages['Amnt_Buck_4'] + mortgages['Amnt_Buck_5']  + \
                    mortgages['Amnt_Buck_6'] + mortgages['Amnt_Buck_12'] + mortgages['Amnt_Buck_13']
mortgages['Targets'] = np.logical_and(mortgages['Num_Day_End_Del'] > 90, np.logical_and(mortgages['Amnt_Pst_Due'] > 100,
                          Amnt_Over_Ninety > .05*mortgages['Amnt_Inst']))
mortgages['Targets'] = np.multiply(mortgages['Targets'], 1)
mortgages['Targets'].value_counts() # Number of defaults

# Calculate sum of targets from previous year of each month
mortgages['Targets_Sum'] = mortgages.groupby('Account')['Targets'].apply(lambda x: x.rolling(min_periods=0, window=12).sum())
threshold_indices = mortgages['Targets_Sum'] > 1
mortgages['Targets_Sum'][threshold_indices] = 1

"""Preparation and Preprocesssing of new Dataframe mortgages_new"""

# Drop Targets with '1' value
mortgages = mortgages[mortgages['Targets'] != 1]

# Delete non-activated status
delete_status = [6, 7]
mortgages = mortgages[~mortgages['Status'].isin(delete_status)]

# Create new Dataframe with the Independent Variables of interest
mortgages_new = mortgages[['Date_Proc', 'Account', 'Num_Borr', 'Num_Quar', 'Cod_Curr', 'Amnt_Inst', 'Amnt_Int', 'Rate1_Type', 'Rate1_Value',
                           'Rate_Sub', 'Cod_Sub', 'Cod_Loan', 'Cod_Freq', 'Cod_Grace', 'Cod_Spv', 'Num_Month_Grace', 'Ind_Insur',
                           'Ind_Spv', 'Cust_Type', 'Cod_Gender', 'Cod_Marital', 'Cod_Educ', 
                           'Cod_Occ_1', 'Num_Child', 'Ind_Tel', 'Age_Customer', 'Post_Code', 'Age_of_Loan', 'Maturity', 'Months_over_30_days', 
                           '12_month_over_30', 'BalPctInitAmt', 'MaxConsMDelqGT1', 'MaxDelqL12M', 'ConsNoChgDelq', 'Targets_Sum']].copy()

# Replace NaN values
mortgages_new = mortgages_new.fillna({'Cod_Sub': 0, 'Cod_Grace': 0, 'Ind_Insur': 0, 'Post_Code': 'NaN'}) # Ind_Insur has two categories in 2016 and in 2017


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
label_list = ['Cod_Curr', 'Rate1_Type', 'Rate_Sub', 'Num_Month_Grace', 'Cust_Type', 'Cod_Loan', 'Cod_Freq', 
              'Cod_Gender', 'Cod_Marital', 'Age_Customer', 'Post_Code', 'Ind_Insur']
mortgages_new[label_list] = mortgages_new[label_list].apply(labelencoder_X.fit_transform)

# Correlation between categorical variables "Cramer's V"
from functions import cramers_v
from calculate_woe_iv import calc_iv
categorical_variables = mortgages_new.copy()
for col in ['Num_Borr', 'Num_Quar', 'Cod_Curr','Rate1_Type', 'Rate_Sub', 'Cod_Sub', 'Cod_Loan', 
            'Cod_Freq', 'Cod_Grace', 'Cod_Spv', 'Num_Month_Grace', 'Ind_Insur', 
            'Ind_Spv', 'Cust_Type', 'Cod_Gender', 'Cod_Marital', 'Cod_Educ', 
            'Cod_Occ_1', 'Ind_Tel', 'Age_Customer', 'Post_Code']:
    categorical_variables[col] =  categorical_variables[col].astype('category')

categorical_variables = categorical_variables.select_dtypes(include = ['category'])
thresh = 0.80
for i in range(0, len(categorical_variables.columns)):
    for j in range(i+1, len(categorical_variables.columns)):
        confusion_matrix = pd.crosstab(categorical_variables[categorical_variables.columns[i]], 
                                       categorical_variables[categorical_variables.columns[j]]).as_matrix()
        if cramers_v(confusion_matrix) > thresh:
            # Exclude variable with lower IV
            iv_1 = calc_iv(mortgages_new, categorical_variables.columns[i], 'Targets_Sum')
            iv_2 = calc_iv(mortgages_new, categorical_variables.columns[j], 'Targets_Sum')
            if iv_1[0]>iv_2[0]:
                mortgages_new = mortgages_new.drop([categorical_variables.columns[j]], axis = 'columns')
            else:
                mortgages_new = mortgages_new.drop([categorical_variables.columns[j]], axis = 'columns')
            print('Correlation between categorical variables', categorical_variables.columns[i], 'and', 
                  categorical_variables.columns[j], ':', cramers_v(confusion_matrix))

#Using Pearson Correlation
cor = mortgages_new.corr()
cor_2 = cor.unstack().sort_values().drop_duplicates()

# Exclude variables after correlation analysis based on the lower IV
mortgages_new = mortgages_new.drop(['MaxConsMDelqGT1', '12_month_over_30'], axis = 'columns')

# Create a new dataframe with data from year 2016 (These data will be used for model training)
start_date = pd.to_datetime('2016-02-01').date()
end_date = pd.to_datetime('2017-01-01').date()
mask = (mortgages_new['Date_Proc'] >= start_date) & (mortgages_new['Date_Proc'] <= end_date)
mortgages_new_2016 = mortgages_new.loc[mask]  # Keep only rows from year 2016
mortgages_new_2016 = mortgages_new_2016.drop(['Date_Proc', 'Account'], axis = 1) # Drop Date_Proc and Account columns
cols = list(mortgages_new_2016.columns.values) #Make a list of all of the columns in mortgages_new_2016
cols.pop(cols.index('Targets_Sum')) #Remove Targets_Sum from list
mortgages_new_2016 = mortgages_new_2016[cols+['Targets_Sum']] #Create new dataframe with columns in the order you want

# Splitting data to X and y
X = mortgages_new_2016.iloc[:, :-1].values
y = mortgages_new_2016.iloc[:, -1].values

# Splitting data to training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scale = [3, 4, 6, 17, 21, 22, 23, 24, 25, 26]
X_train[:, X_scale] = scaler.fit_transform(X_train[:, X_scale])
X_test[:, X_scale] = scaler.fit_transform(X_test[:, X_scale])

#Building the optimal model using Backward Elimination
import statsmodels.api as sm
X_train = np.append(arr=np.ones([X_train.shape[0],1]).astype(int), values = X_train, axis = 1) # Add the intercept
X_opt = np.arange(0,28) 
numVars = len(X_train[X_opt][0])
for i in range(0, numVars):
    regressor_OLS = sm.OLS(y_train, X_train[:, X_opt]).fit()
    maxVar = max(regressor_OLS.pvalues).astype(float)
    if maxVar > 0.05:
        for j in range(0, numVars - i):
            if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                X_opt = np.delete(X_opt, j) # Indices from variables that pass the backward elimination are saved on X_opt array

print(regressor_OLS.summary())
if X_opt[0] == 0:
    X_opt = np.delete(X_opt, 0)
X_opt = X_opt - 1

# temp_X after backward elimination
temp_X = mortgages_new_2016.iloc[:, X_opt]

# Create dummy variables for categorical variables (> 2 categories) and delete one dummy variable
onehotlist = ['Num_Borr', 'Rate_Sub', 'Cod_Freq', 'Cod_Grace', 'Age_Customer']
temp_X = pd.get_dummies(temp_X, prefix = onehotlist, columns = onehotlist)
temp_X = temp_X.drop(['Num_Borr_1.0', 'Rate_Sub_0', 'Cod_Grace_0.0', 
                                              'Cod_Freq_0', 'Age_Customer_0'], 
                axis = 'columns')
X = temp_X.iloc[:,:].values

# Splitting data to training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scale = [1, 2, 4, 6, 8, 9, 10, 11, 12, 13]
X_train[:, X_scale] = scaler.fit_transform(X_train[:, X_scale])
X_test[:, X_scale] = scaler.fit_transform(X_test[:, X_scale])

# Logistic Regression

# Train the classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred_prob = classifier.predict_proba(X_test)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

# Logistic regression with stats model
import statsmodels.api as sm
logit_model=sm.Logit(y_train, X_train)
result=logit_model.fit()
print(result.summary2())
# Run model to 2017 accounts

# Splitting data to X_test_2017 and y_test_2017
start_date = pd.to_datetime('2017-02-01').date()
end_date = pd.to_datetime('2018-01-01').date()
mask = (mortgages_new['Date_Proc'] >= start_date) & (mortgages_new['Date_Proc'] <= end_date)
mortgages_new_2017 = mortgages_new.loc[mask]  # Keep only rows from year 2016
mortgages_new_2017 = mortgages_new_2017.drop(['Date_Proc', 'Account'], axis = 1) # Drop Date_Proc and Account columns
cols = list(mortgages_new_2017.columns.values) #Make a list of all of the columns in mortgages_new_2016
cols.pop(cols.index('Targets_Sum')) #Remove Targets_Sum from list
mortgages_new_2017 = mortgages_new_2017[cols+['Targets_Sum']] #Create new dataframe with columns in the order you want
y_test_2017 = mortgages_new_2017.iloc[:, -1].values
X_test_2017 = mortgages_new_2017.iloc[:, X_opt] # Keep variables after backward elimination

# Create dummy variables for categorical variables (> 2 categories) and delete one dummy variable
X_test_2017 = pd.get_dummies(X_test_2017, prefix = onehotlist, columns = onehotlist)
X_test_2017 = X_test_2017.drop(['Num_Borr_1.0', 'Rate_Sub_0', 'Cod_Grace_0.0', 
                                              'Cod_Freq_0', 'Age_Customer_0'], 
                axis = 'columns')
X_test_2017 = X_test_2017.iloc[:, :].values

# Normalize features
X_test_2017[:, X_scale] = scaler.fit_transform(X_test_2017[:, X_scale])

# Predict Probability of default
y_pred_prob_2017 = classifier.predict_proba(X_test_2017)
y_pred_prob_2017 = y_pred_prob_2017[:, 1]
# Predicting the Test set results
y_pred_2017 = classifier.predict(X_test_2017)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test_2017, y_pred_2017)


# Model Evaluation

# R2 and adjusted R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test_2017, y_pred_2017)

# AUC
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

fpr, tpr, thresholds = roc_curve(y_test_2017, y_pred_2017)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();

# Mean absolute error, mean squared error
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae_lr = mean_absolute_error(y_test_2017, y_pred_prob_2017)
mse_lr = mean_squared_error(y_test_2017, y_pred_prob_2017)
me_lr = np.mean(y_test_2017 - y_pred_prob_2017)
# gini
gini = (2 * roc_auc) - 1









