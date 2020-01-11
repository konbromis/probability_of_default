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

# Create dummy variables for categorical variables (> 2 categories) and delete one dummy variable
onehotlist = ['Num_Borr', 'Num_Quar', 'Rate_Sub', 'Cod_Loan', 'Cod_Freq', 'Cod_Grace', 'Num_Month_Grace', 'Cod_Marital', 
              'Cod_Gender', 'Cod_Educ', 'Cod_Occ_1', 'Post_Code', 'Age_Customer']
mortgages_new = pd.get_dummies(mortgages_new, prefix = onehotlist, columns = onehotlist)
mortgages_new = mortgages_new.drop(['Num_Borr_1.0', 'Num_Quar_0.0', 'Rate_Sub_0', 'Cod_Loan_0', 'Cod_Freq_0', 'Cod_Grace_0.0', 
                                              'Num_Month_Grace_0', 'Cod_Marital_0', 'Cod_Gender_0', 'Cod_Educ_0', 'Cod_Occ_1_0', 'Post_Code_0', 
                                              'Age_Customer_0'], axis = 'columns')

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
X_scale = [1, 2, 4, 10, 12, 13, 14, 15, 16, 17, 18, 19]
X_train[:, X_scale] = scaler.fit_transform(X_train[:, X_scale])
X_test[:, X_scale] = scaler.fit_transform(X_test[:, X_scale])

# Part 2 Create ANN model

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Find the optimal value for regularizer L1
values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
all_train, all_test = list(), list()
for param in values:
	# define model
	model = Sequential()
	model.add(Dense(units= 80, kernel_initializer= 'uniform', activation= 'relu', kernel_regularizer= keras.regularizers.l1(param), input_dim = 158))
	model.add(Dense(units= 1, kernel_initializer= 'uniform', activation= 'sigmoid'))
	model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
	# fit model
	model.fit(X_train, y_train, batch_size =  10, epochs = 100)
	# evaluate the model
	train_acc = model.evaluate(X_train, y_train, verbose=0)
	test_acc = model.evaluate(X_test, y_test, verbose=0)
	print('Param: %f, Train: %.3f, Test: %.3f' % (param, train_acc[1], test_acc[1]))
	all_train.append(train_acc)
	all_test.append(test_acc)
# plot train and test accuracies based on the different L1 Values - Validation Curve
import matplotlib.pyplot as pyplot
all_train_2  = []
for i in range(len(all_train)):
    all_train_2.append(all_train[i][1])
    
all_test_2  = []
for i in range(len(all_test)):
    all_test_2.append(all_test[i][1])

pyplot.semilogx(values, all_train_2, label='train', marker='o')
pyplot.semilogx(values, all_test_2, label='test', marker='o')
pyplot.legend()
pyplot.show() # Found that the L1 value that reduces overfitting is 0.01

# Initialize the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units= 80, kernel_initializer= 'uniform', activation= 'relu', kernel_regularizer= keras.regularizers.l1(0.01), 
                     input_dim = 158))

# Adding the output layer
classifier.add(Dense(units= 1, kernel_initializer= 'uniform', activation= 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size =  10, epochs = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred_prob = classifier.predict(X_test)
y_pred = classifier.predict_classes(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_pred)

# Splitting data to X_test_2017 and y_test_2017
start_date = pd.to_datetime('2017-02-01').date()
end_date = pd.to_datetime('2018-01-01').date()
mask = (mortgages_new['Date_Proc'] >= start_date) & (mortgages_new['Date_Proc'] <= end_date)
mortgages_new_2017 = mortgages_new.loc[mask]  # Keep only rows from year 2017
mortgages_new_2017 = mortgages_new_2017.drop(['Date_Proc', 'Account'], axis = 1) # Drop Date_Proc and Account columns
cols = list(mortgages_new_2017.columns.values) #Make a list of all of the columns in mortgages_new_2017
cols.pop(cols.index('Targets_Sum')) #Remove Targets_Sum from list
mortgages_new_2017 = mortgages_new_2017[cols+['Targets_Sum']] #Create new dataframe with columns in the order you want

# Splitting data to X_test_2017 and y_test_2017
X_test_2017 = mortgages_new_2017.iloc[:, :-1].values
y_test_2017 = mortgages_new_2017.iloc[:, -1].values

# Normalize features
X_scale = [1, 2, 4, 10, 12, 13, 14, 15, 16, 17, 18, 19]
X_test_2017[:, X_scale] = scaler.fit_transform(X_test_2017[:, X_scale])

# Predicting the 2017 Test set results
y_pred_prob_2017 = classifier.predict(X_test_2017)
y_pred_prob_2017 = y_pred_prob_2017.ravel()
y_pred_2017 = classifier.predict_classes(X_test_2017)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_2017 = confusion_matrix(y_test_2017, y_pred_2017)

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

mae_knn = mean_absolute_error(y_test_2017, y_pred_prob_2017)
mse_knn = mean_squared_error(y_test_2017, y_pred_prob_2017)
me_lr = np.mean(y_test_2017 - y_pred_prob_2017)

# gini
gini = (2 * roc_auc) - 1




# Calculate Kolmogorov Smirnov test result
test_target = pd.Series(y_test, name = 'Test_Target')
test_prob= pd.Series(np.squeeze(y_pred_prob), name = 'Test_Prob')
ks= pd.concat([test_target, test_prob], axis = 1)

# Group Probabilities by quartiles and run program to make 10 bins for the probabilities
ks['decile'] = pd.cut(ks['Test_Prob'], bins = [-np.inf, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.99],
  labels = ['1','2','3','4','5','6','7','8','9'])

ks.columns = ['Defaulter', 'Probability', 'Decile']

# Create Non_Defaulter column
ks['Non_Defaulter'] = 1 - ks['Defaulter']

# Pivot table to make the chart table to calculate KS statistics
df1 = pd.pivot_table(data = ks, index = ['Decile'], values = ['Defaulter', 'Non_Defaulter', 'Probability'], 
                     aggfunc = {'Defaulter': [np.sum], 'Non_Defaulter': [np.sum], 'Probability': [np.min, np.max]})

df1.columns = ['Defaulter_Count','Non-Defaulter_Count','max_score','min_score']
df1['Total_Cust'] = df1['Defaulter_Count']+df1['Non-Defaulter_Count']

# Sort min_score in descending order
df2 = df1.sort_values(by = 'min_score', ascending = False)

# Calculate the defaulters and non-defaulters rate per decile
df2['Default_Rate'] = (df2['Defaulter_Count'] / df2['Total_Cust']).apply('{0:.2%}'.format)
default_sum = df2['Defaulter_Count'].sum()
non_default_sum = df2['Non-Defaulter_Count'].sum()
df2['Default %'] = (df2['Defaulter_Count']/default_sum).apply('{0:.2%}'.format)
df2['Non_Default %'] = (df2['Non-Defaulter_Count']/non_default_sum).apply('{0:.2%}'.format)
df2

# Calculate KS Statistics using the above values
df2['ks_stats'] = np.round(((df2['Defaulter_Count'] / df2['Defaulter_Count'].sum()).cumsum() -(df2['Non-Defaulter_Count'] / df2['Non-Defaulter_Count'].sum()).cumsum()), 4) * 100
df2


# Find the KS Statistics value which is the max of KS statistics scored for each decile
flag = lambda x: '*****' if x == df2['ks_stats'].max() else ''
df2['max_ks'] = df2['ks_stats'].apply(flag)
df2





















