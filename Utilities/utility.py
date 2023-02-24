import pandas as pd
import streamlit as st
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from iosense import *
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import MinMaxScaler


def data_import(device_id, start_time, end_time, sensors, target_column, filter_by_timestamp, percentage_of_datasplit,
                sariam_freqq):
    '''
        Importing data from iosense.io libarary
        device_id:- device Id required to import data
        start_time:- starting time stamp (first data point)
        end_time:- end time stamp (last data point)
        sensors:-sensor ids to import data(features,ex.D1,D2,D0)
        target_col:-dependent column(ex.Foraward active energy)
        filterBy:-frequency of time stamp to select(ex.1MIN,5MIN,15MIN)
        percentage:- percentage of data to split(ex. 0.8,0.7)
        sariam_freqq:- Seasonal factor for Sarimax
    '''
    # extracting data by UI and API key
    print('++',sensors)
    a = dataAccess("b319b59c-9c7d-4acd-a070-f960d3f68f3c", "data.iosense.io")
    df = (a.dataQuery(device_id, start_time=start_time, end_time=end_time, sensors=[sensors], cal=False, bands=None,
                      echo=True))

    # changing time zone of date time
    df["time"] = pd.to_datetime(df["time"], utc=False)
    df['time'] = df['time'].dt.tz_convert('Asia/Kolkata')
    df['time'] = df['time'].dt.tz_localize(None)
    df['time'] = df['time'].dt.round('30S')

    # Resampling data to 30 seconds to avoid time gap between data
    df = df.resample('30s', on='time').mean().ffill()
    df.reset_index(inplace=True)

    return df, target_column, filter_by_timestamp, percentage_of_datasplit, sariam_freqq, device_id, start_time, end_time

df,target_column,filter_by_timestamp,percentage_of_datasplit,sariam_freqq,device_id, start_time, end_time=data_import(device_id="YBEM_B1",start_time='2022-01-01',
                                                           end_time='2022-12-31',sensors='D43',
                                                           target_column="Forward Active Energy (D43)",
                                                           filter_by_timestamp="1HR",
                                                           percentage_of_datasplit=0.8,sariam_freqq="H")




def date_time_features_extraction(df, target_column, filter_by_timestamp, percentage_of_datasplit, sariam_freqq):
    """
    Extracting diffrent features from date time variable
    month,day,week,hour and weekday
    """
    # Calculating energy difference by subtracting first and privious values
    df["forward_active_energy_difference"] = df["Forward Active Energy (D43)"].diff(periods=1)
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["week"] = df["time"].dt.isocalendar().week
    df["hour"] = df["time"].dt.hour
    df["weekday"] = df["time"].dt.weekday
    target_column = 'forward_active_energy_difference'

    return df, target_column, filter_by_timestamp, percentage_of_datasplit, sariam_freqq



df,target_column,filter_by_timestamp,percentage_of_datasplit,sariam_freqq=date_time_features_extraction(df,target_column,
                                                            filter_by_timestamp,percentage_of_datasplit,sariam_freqq)



def filterbytimestamp(df,filter_by_timestamp,percentage_of_datasplit,sariam_freqq,target_column):
    '''
    Filtering date time stamp by diffirent frequencies
    1MIN:-Filters data by per min
    5MIN:-Filters data by per 5 min
    15MIN:-Filters data by per 15 min
    30MIN:-Filters data by per 30 min
    1HR:-Filters data by per 1 Hour
    1DAY:-Filters data by per 1 Day
    '''
    if filter_by_timestamp=="1MIN":
        filter_df=df[(df["time"].dt.second==0)].reset_index(drop=True)
    elif filter_by_timestamp=="5MIN":
        filter_df=df[(df["time"].dt.minute%5==0)&(df["time"].dt.second==0)].reset_index(drop=True)
    elif filter_by_timestamp=="15MIN":
        filter_df=df[(df["time"].dt.minute%15==0)&(df["time"].dt.second==0)].reset_index(drop=True)
    elif filter_by_timestamp=="30MIN":
        filter_df=df[(df["time"].dt.minute%30==0)&(df["time"].dt.second==0)].reset_index(drop=True)
    elif filter_by_timestamp=="1HR":
        filter_df=df[(df["time"].dt.minute==0)&(df["time"].dt.second==0)].reset_index(drop=True)
    elif filter_by_timestamp=="1DAY":
        filter_df=df[(df["time"].dt.hour==0)&(df["time"].dt.minute==0)&(df["time"].dt.second==0)].reset_index(drop=True)
    else:
        filter_df=df
    return filter_df,percentage_of_datasplit,sariam_freqq,target_column



df,percentage_of_datasplit,sariam_freqq,target_column=filterbytimestamp(df,filter_by_timestamp,
                                                                        percentage_of_datasplit,sariam_freqq,target_column)



def data_scaling(df, percentage_of_datasplit, sariam_freqq, target_column):
    """
    scaling values in 0,1 range by using min max scaler
    """
    scaler = MinMaxScaler()
    scaled_df = df.drop(columns=["time"])
    scaled_df = pd.DataFrame(
        scaler.fit_transform(scaled_df), columns=scaled_df.columns)
    scaled_df["time"] = df["time"]
    return scaled_df, percentage_of_datasplit, sariam_freqq, target_column

scaled_df,percentage_of_datasplit,sariam_freqq,target_column=data_scaling(df,percentage_of_datasplit,sariam_freqq,target_column)


def univarient_lstm_data_scaling(train_df,test_df,percentage_of_datasplit,sariam_freqq,target_column):
    """
    scaling values in 0,1 range by using min max scaler
    """
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_train=scaler.fit_transform(np.array(train_df[target_column]).reshape(-1,1))
    scaled_test = scaler.transform(np.array(test_df[target_column]).reshape(-1,1))
    return scaled_train,scaled_test,percentage_of_datasplit,sariam_freqq,target_column,scaler


def data_split(df, percentage_of_datasplit, sariam_freqq, target_column):
    """
 percentage_of_datasplit,sariam_freqq,target_col=targetain and test formate by usning percentage(ex..0.8,0.7)
    """
    split_len = math.floor(len(df) * percentage_of_datasplit)
    train_df = df[:split_len]
    test_df = df[split_len:]
    print(train_df.shape, test_df.shape)
    return train_df, test_df, sariam_freqq, target_column

train_df,test_df,sariam_freqq,target_column=data_split(scaled_df,percentage_of_datasplit,sariam_freqq,target_column)


def reshaping_lstm_df(x_train):
    """
       It changes shape of array from 2 dimensions to 3 dimensions to feed lstm model
    """
    # reshape input to be [samples, time steps, features] required for LSTM
    x_train =x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
    print(x_train.shape)
    return x_train

def graph_plots(train_df, test_df, predicted_values,target_column):
    """
        plotting trained,test and predicted energy differnce plot to understand distribution of prediction over test data
    """
    plt.figure(figsize=(15, 10))
    plt.plot(train_df["time"], train_df[target_column])
    # plt.axvline(x = len(train_df["time"]), color = 'black', linestyle='dashed')

    plt.plot(test_df["time"], test_df[target_column])
    plt.plot(test_df["time"], predicted_values, color='green')
    plt.title("Trained, Test and predicted energy diffrence")
    plt.xlabel('Date')
    plt.ylabel('Energy difference')
    plt.legend(["trained_energy", "tested_energy", "predicted_energy"], loc="upper right")
    plt.show()

    """
        plotting test and predicted energy differnce plot to understand distribution of prediction over test data 
    """
    plt.figure(figsize=(15, 10))
    plt.plot(test_df["time"], test_df[target_column])
    plt.plot(test_df["time"], predicted_values, color='green')
    plt.title("Test and predicted energy diffrence")
    plt.xlabel('Date')
    plt.ylabel('Energy difference')
    plt.legend(["test_df_energy", "predicted_energy"], loc="upper right")
    plt.show()
    return train_df, test_df, predicted_values



def calculate_mape(train_df,test_df,predicted_values,target_column):
    """
    calculating mean absolute percentile error(mape) to cheak diffrence camparison of predicted values over actual values
    """
    actual_values=test_df[target_column]
    K=(abs(predicted_values-actual_values)/actual_values)#.mean()
    K.replace([np.inf, -np.inf], np.nan, inplace=True)
    K.dropna(inplace=True)
    mape=K.mean()
    print("Mean Absolute Percentage Error:",mape)
    return mape


def correlation_coefficient(train_df, test_df, predicted_values,target_column):
    """
    calculating correlation coefficient of predicted values and actual values
    """
    actual_values = test_df[target_column]

    corr = np.corrcoef(predicted_values, actual_values)[0, 1]
    print("Correlation between the Actual and the Forecast:", corr)
    return corr


def calculate_minmax(train_df, test_df, predicted_values,target_column):
    """
    calculating min max to cheak diffrence camparison of predicted values over actual values
    """
    actual_values = test_df[target_column]

    mins = np.amin(np.hstack([predicted_values[:, None],
                              test_df[target_column][:, None]]), axis=1)
    maxs = np.amax(np.hstack([predicted_values[:, None],
                              test_df[target_column][:, None]]), axis=1)
    minmax = 1 - np.mean(mins / maxs)

    print("Min-Max Error", minmax)
    return minmax