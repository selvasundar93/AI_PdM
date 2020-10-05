# Import Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import pickle

# Load and Combine the Dataset
combined_data = pd.DataFrame()
data_dir = 'Data/'
# Iterate through all the files
for file_name in os.listdir(data_dir):
    df = pd.read_csv(os.path.join(data_dir, file_name), sep='\t')
    # To convert into numpy values
    df_b1 = df.iloc[:,0].values
    df_b2 = df.iloc[:,1].values
    df_b3 = df.iloc[:,2].values
    df_b4 = df.iloc[:,3].values
    # To calculate the mean
    df_b1_mean = np.mean(np.absolute(df_b1))
    df_b2_mean = np.mean(np.absolute(df_b2))
    df_b3_mean = np.mean(np.absolute(df_b3))
    df_b4_mean = np.mean(np.absolute(df_b4))
    # To calculate RMS
    df_b1_rms = np.sqrt((np.sum(df_b1**2))/len(df_b1))
    df_b2_rms = np.sqrt((np.sum(df_b2**2))/len(df_b2))
    df_b3_rms = np.sqrt((np.sum(df_b3**2))/len(df_b3))
    df_b4_rms = np.sqrt((np.sum(df_b4**2))/len(df_b4))
    # To calculate kurtosis
    df_b1_kurt = scipy.stats.kurtosis(df_b1,fisher=False)
    df_b2_kurt = scipy.stats.kurtosis(df_b2,fisher=False)
    df_b3_kurt = scipy.stats.kurtosis(df_b3,fisher=False)
    df_b4_kurt = scipy.stats.kurtosis(df_b4,fisher=False)
    # Concate into Pandas DataFrame
    df_1 = pd.concat([pd.Series(df_b1_mean),pd.Series(df_b2_mean),pd.Series(df_b3_mean),pd.Series(df_b4_mean),\
                      pd.Series(df_b1_rms),pd.Series(df_b2_rms),pd.Series(df_b3_rms),pd.Series(df_b4_rms),\
                      pd.Series(df_b1_kurt),pd.Series(df_b2_kurt),pd.Series(df_b3_kurt),pd.Series(df_b4_kurt)],axis=1)
    df_1.index = [file_name]   
    # Append individual dataframes to create a single combined dataset
    combined_data = combined_data.append(df_1)
    
# Insert Column headers    
combined_data.columns = ['Bearing1_Mean','Bearing2_Mean','Bearing3_Mean','Bearing4_Mean',\
                         'Bearing1_RMS','Bearing2_RMS','Bearing3_RMS','Bearing4_RMS',\
                        'Bearing1_Kurt','Bearing2_Kurt','Bearing3_Kurt','Bearing4_Kurt']
combined_data.index = pd.to_datetime(combined_data.index, format='%Y.%m.%d.%H.%M.%S')

# Sort the index in chronological order
combined_data = combined_data.sort_index()
# Drop Last 2 rows
combined_data = combined_data[:-2]

# Data Visualization
fig, ax = plt.subplots(figsize=(15,7))
ax.plot(combined_data['Bearing1_Mean'], label='Bearing1 Mean', color='red',linewidth=1)
ax.plot(combined_data['Bearing1_RMS'], label='Bearing1 RMS', color='blue', linewidth=1)
plt.xlabel('Date_Time')
plt.ylabel("Amplitude")
plt.legend(loc='best')
ax.set_title('Bearing1 Sensor Data', fontsize=16)
plt.show()
fig, ax = plt.subplots(figsize=(15,7))
ax.plot(combined_data['Bearing1_Kurt'], label='Bearing1 Kurtosis', color='green', linewidth=1)
plt.xlabel('Date_Time')
plt.ylabel("Amplitude")
plt.legend(loc='best')
ax.set_title('Bearing1 Sensor Data', fontsize=16)
plt.show()

# Convert index into Samples (time steps) for easier interpretation
new_comb_data = combined_data.copy()
new_comb_data['time_steps'] = np.arange(0,len(combined_data))
new_comb_data.index = new_comb_data['time_steps']
new_comb_data.drop(['time_steps'],axis=1,inplace=True)

# Split the Dataset into Training and Test set
training = new_comb_data.iloc[:450,:]
test = new_comb_data.iloc[450:,:]
test.reset_index(inplace=True)

# Function to Visualize the results
def viz_result(data,pred,title):
    fig, ax = plt.subplots(nrows=2,figsize=(15,8))
    ax[0].plot(data, label='Bearing Signal', color='red',linewidth=1)
    ax[1].scatter(range(0,len(pred)),pred, label='Anomaly (-1)')
    ax[0].legend()
    ax[1].legend()
    plt.show()
    viz_df=pd.DataFrame({'Bearing':data,'No_Anomaly':pred})
    plt.figure(figsize=(15,8))
    sns.scatterplot(x=range(0,len(viz_df)),y=viz_df.Bearing,hue=viz_df.No_Anomaly,palette="deep")
    plt.plot(data, color='black',linewidth=1,alpha=0.5)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.show()
    
# Local Outlier Factor on RMS values
from sklearn.neighbors import LocalOutlierFactor
lof_rms = LocalOutlierFactor(n_neighbors=20, contamination=0.002,novelty=True)
lof_rms.fit(training['Bearing1_RMS'].values.reshape(-1,1))

# Testing on Entire dataset
pred_lof_rms = lof_rms.predict(new_comb_data['Bearing1_RMS'].values.reshape(-1,1))
unique_elements, counts_elements = np.unique(pred_lof_rms, return_counts=True)
print("LOF - RMS - Testing on Entire dataset")
print("Anomaly\t"+str(unique_elements[0])+"\t"+str(counts_elements[0]))
try:
    print("Normal \t"+str(unique_elements[1])+"\t"+str(counts_elements[1]))
except:
    print("No Normal data")
viz_result(new_comb_data['Bearing1_RMS'],pred_lof_rms,'Bearing Vibration (RMS) - Local Outlier Factor')

# Save the model
with open('Models\lof_rms_trained_model.pkl', 'wb') as f:
    pickle.dump(lof_rms, f)

# Local Outlier Factor on Mean values
from sklearn.neighbors import LocalOutlierFactor
lof_mean = LocalOutlierFactor(n_neighbors=20, contamination=0.004,novelty=True)
lof_mean.fit(training['Bearing1_Mean'].values.reshape(-1,1))

# Testing on Entire dataset
pred_lof_mean = lof_mean.predict(new_comb_data['Bearing1_Mean'].values.reshape(-1,1))
unique_elements, counts_elements = np.unique(pred_lof_mean, return_counts=True)
print("LOF - Mean - Testing on Entire dataset")
print("Anomaly\t"+str(unique_elements[0])+"\t"+str(counts_elements[0]))
try:
    print("Normal \t"+str(unique_elements[1])+"\t"+str(counts_elements[1]))
except:
    print("No Normal data")
viz_result(new_comb_data['Bearing1_Mean'],pred_lof_mean,'Bearing Vibration (Mean) - Local Outlier Factor')

# Save the model
with open('Models\lof_mean_trained_model.pkl', 'wb') as f:
    pickle.dump(lof_mean, f)