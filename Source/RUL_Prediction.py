import os
import pandas as pd
import numpy as np
import pickle
import scipy.stats
# Create the blank dataframe
combined_data = pd.DataFrame()
# Set the file directory
data_dir = '../Data/'
# Iterate through all the files
for file_name in os.listdir(data_dir):
    df = pd.read_csv(os.path.join(data_dir, file_name), sep='\t')
    # To convert into numpy values
    df_b1 = df.iloc[:,0].values
    # To calculate the mean
    df_b1_mean = np.mean(np.absolute(df_b1)) 
    # To calculate RMS
    df_b1_rms = np.sqrt((np.sum(df_b1**2))/len(df_b1))  
    # To calculate kurtosis
    df_b1_kurt = scipy.stats.kurtosis(df_b1,fisher=False)
    # Concate into Pandas DataFrame
    df_1 = pd.concat([pd.Series(df_b1_mean),pd.Series(df_b1_rms),pd.Series(df_b1_kurt)],axis=1)
    df_1.index = [file_name]
    # Append individual dataframes to create a single combined dataset
    combined_data = combined_data.append(df_1)   
# Insert Column headers    
combined_data.columns = ['Bearing1_Mean','Bearing1_RMS','Bearing1_Kurt']
# Set the data_time index and ensuring the proper format
combined_data.index = pd.to_datetime(combined_data.index, format='%Y.%m.%d.%H.%M.%S')
# Sort the index in chronological order
combined_data = combined_data.sort_index()
# Drop last 2 rows
combined_data = combined_data[:-2]
# Reset Index
new_comb_data = combined_data.copy()
new_comb_data = new_comb_data.reset_index()
new_comb_data.columns = ['date_time','Bearing1_Mean','Bearing1_RMS','Bearing1_Kurt']

# Calculate the Remaining Useful Life (RUL) interms of Fraction Failing
# Ex: If Fraction Failing = 10%, then RUL is 90%
samples = new_comb_data.Bearing1_RMS[:450]
sample_mean = samples.mean() 
samples_full = new_comb_data.Bearing1_RMS.values
# ffr - feature used to calculate the Fraction Failing
ffr = np.abs(samples_full - sample_mean)

# Estimate shape parameter(alpha) & scale Parameter (beta)
from reliability.Fitters import Fit_Weibull_2P
wb = Fit_Weibull_2P(failures=ffr,show_probability_plot=False)

from reliability.Distributions import Weibull_Distribution
dist = Weibull_Distribution(alpha=0.0135, beta=0.475861)  # this created the distribution object
dist.CDF(show_plot=False)

# select required columns
rul_pred = new_comb_data[['date_time','Bearing1_RMS','Bearing1_Kurt']]
# Add ffr feature to dataframe
rul_pred['RMS_Feature'] = ffr

# We are creating 5 output classes based on RMS_Feature fitted on Weibull Distribution
'''
Class | RMS_Feature (Range) | Fraction Failing | RUL
------|---------------------|------------------|----
1 | 0.0-0.001 | 0-20% | 80%
2 | 0.001-0.0035 | 20-40% | 60%
3 | 0.0035-0.011 | 40-60% | 40%
4 | 0.011-0.037 | 60-80% | 20%
5 | > 0.037 | 80-100% | < 20%
'''

# Create funtion to find the corresponding class from RMS_Feature
def rul_class(x):
    if x < 0.001:
        return 1
    elif ((x>0.001) & (x<0.0035)):
        return 2
    elif ((x>0.0035) & (x<0.011)):
        return 3
    elif ((x>0.011) & (x<0.037)):
        return 4
    else:
        return 5

rul_pred['Class'] = rul_pred.RMS_Feature.apply(rul_class)

# To Create a Dataframe with Features (Bearing1_RMS, Bearing1_kurt) Current and Previous Value
rul_pred_df = rul_pred.copy()
rul_pred_df = rul_pred_df.drop(['RMS_Feature'],axis=1)
rul_pred_df['Bearing1_RMS_Prev'] = rul_pred_df['Bearing1_RMS'].shift(1)
rul_pred_df['Bearing1_Kurt_Prev'] = rul_pred_df['Bearing1_Kurt'].shift(1)
rul_pred_df.dropna(inplace=True)

# Class Prediction
# Train test split
from sklearn.model_selection import train_test_split
X = rul_pred_df.drop(['date_time','Class'],axis=1)
y = rul_pred_df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
clf_dt_1 = DecisionTreeClassifier(max_depth=3,random_state=0)
clf_dt_1.fit(X_train,y_train)

# Save the model
with open('Models/DT_Classifier.pkl', 'wb') as f:
    pickle.dump(clf_dt_1, f)
print("Decision Tree Classifier Model Saved")