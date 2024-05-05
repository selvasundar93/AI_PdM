# Import Libraries
import os
import pandas as pd
import numpy as np
import scipy.stats
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,TimeDistributed, RepeatVector
import joblib
from sklearn.neighbors import LocalOutlierFactor

# Load and Combine the Dataset
combined_data = pd.DataFrame()
data_dir = '../Data/'
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

    # To calculate kurtosis
    df_b1_kurt = scipy.stats.kurtosis(df_b1,fisher=False)

    # Concate into Pandas DataFrame
    df_1 = pd.concat([pd.Series(df_b1_mean),pd.Series(df_b2_mean),pd.Series(df_b3_mean),pd.Series(df_b4_mean),\
                      pd.Series(df_b1_rms),pd.Series(df_b1_kurt)],axis=1)
    df_1.index = [file_name]   
    # Append individual dataframes to create a single combined dataset
    combined_data = combined_data.append(df_1)
    
# Insert Column headers    
combined_data.columns = ['Bearing1_Mean','Bearing2_Mean','Bearing3_Mean','Bearing4_Mean',\
                         'Bearing1_RMS','Bearing1_Kurt']
combined_data.index = pd.to_datetime(combined_data.index, format='%Y.%m.%d.%H.%M.%S')

# Sort the index in chronological order
combined_data = combined_data.sort_index()
# Drop Last 2 rows
combined_data = combined_data[:-2]

# Convert index into Samples (time steps) for easier interpretation
new_comb_data = combined_data.copy()
new_comb_data['time_steps'] = np.arange(0,len(combined_data))
new_comb_data.index = new_comb_data['time_steps']
new_comb_data.drop(['time_steps'],axis=1,inplace=True)

# Split the Dataset into Training and Test set
training = new_comb_data.iloc[:450,:]
test = new_comb_data.iloc[450:,:]
test.reset_index(inplace=True)

# Local Outlier Factor on RMS values
lof_rms = LocalOutlierFactor(n_neighbors=20, contamination=0.002,novelty=True)
lof_rms.fit(training['Bearing1_RMS'].values.reshape(-1,1))

# Save the model
with open('Models/lof_rms_trained_model.pkl', 'wb') as f:
    pickle.dump(lof_rms, f)
print("LOF-RMS Model Saved")

# Local Outlier Factor on Mean values
lof_mean = LocalOutlierFactor(n_neighbors=20, contamination=0.004,novelty=True)
lof_mean.fit(training['Bearing1_Mean'].values.reshape(-1,1))

# Save the model
with open('Models/lof_mean_trained_model.pkl', 'wb') as f:
    pickle.dump(lof_mean, f)
print("LOF-Mean Model Saved")

# Deep Learning LSTM - Autoencoder
train = combined_data[:'2004-02-15 12:42:39']
test = combined_data['2004-02-15 12:52:39':]

# To Consider only Bearing Mean values
train = train.iloc[:,:4]
test = test.iloc[:,:4]

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)
joblib.dump(scaler, "Models/scaler_file")

# Reshape Inputs for LSTM
X_train = train_scaled.reshape(train_scaled.shape[0], 1, train_scaled.shape[1])
X_test = test_scaled.reshape(test_scaled.shape[0], 1, test_scaled.shape[1])

# LSTM Autoencoder - DL Model
dl_model = Sequential([
    LSTM(16, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(4, activation='relu', return_sequences=False),
    RepeatVector(X_train.shape[1]),
    LSTM(4, activation='relu', return_sequences=True),
    LSTM(16, activation='relu', return_sequences=True),
    TimeDistributed(Dense(X_train.shape[2]))])
dl_model.compile(optimizer='adam', loss='mae')

nb_epochs = 100
batch_size = 10
dl_model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size, validation_split=0.05)
dl_model.save("Models/LSTM_Autoencoder.h5")
print("LSTM-Autoencoder Model Saved")