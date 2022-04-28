import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

### Add column consisting of 0 or 1 basedd on comparison between D th price and D+N th price
### where D = date, N = holding period
### If D+N th price > D th price => 1
### Otherwise 0
def profits(df,N):

    df_copy = df
    num_row = df.shape[0]
    signals = np.full(num_row, np.nan)
    header = 'signals_' + str(N) + 'd'
    index = 0
    for i in range(167, 167+num_row-N):
        if df['Price'][i] < df['Price'][i+N]:
            signals[index] = 1
        else:
            signals[index] = 0
        
        index += 1

    df_copy[header] = signals.tolist()
    
    return df_copy

### Scale all the features in the dataframe

## df = dateframe
## type: 1 = Normalize + Standardize
##       2 = Normalize only
##       3 = Standardize only
def scale(df, type):
    scaler1 = MinMaxScaler(feature_range=(-1,1))
    scaler2 = MinMaxScaler(feature_range=(0,1))
    standard_scaler = StandardScaler() 

    ## Normalize -> Standardize
    if type == 1:

        for column in df.columns:
            if min(df[column]) >= 0:
                df[column] = scaler2.fit_transform(np.array(df[column]).reshape(-1,1))
                df[column] = standard_scaler.fit_transform(np.array(df[column]).reshape(-1,1))
            else:
                df[column] = scaler1.fit_transform(np.array(df[column]).reshape(-1,1))
                df[column] = standard_scaler.fit_transform(np.array(df[column]).reshape(-1,1))

    ## Normalize only
    elif type == 2:

        for column in df.columns:
            if min(df[column]) >= 0:
                df[column] = scaler2.fit_transform(np.array(df[column]).reshape(-1,1))
                # df[column] = standard_scaler.fit_transform(np.array(df[column]).reshape(-1,1))
            else:
                df[column] = scaler1.fit_transform(np.array(df[column]).reshape(-1,1))
                # df[column] = standard_scaler.fit_transform(np.array(df[column]).reshape(-1,1))

    ## Standardize only
    elif type == 3:
        for column in df.columns:
            df[column] = standard_scaler.fit_transform(np.array(df[column]).reshape(-1,1))

    return df


### Scale and split the dataframe into x_train, x_test, y_train, y_test

## df = dateframe
## N = day used to generate signal
## type: 1 = Normalize + Standardize
##       2 = Normalize only
##       3 = Standardize only
def scale_split(df, N, type):

    target_header = 'signals_' + str(N) + 'd'
    data_length = df.shape[0] - N

    y_data = df[target_header]
    x_data = df.drop(['Date','signals_7d','signals_14d','signals_30d','signals_1d'], axis=1)

    x_data = scale(x_data, type)

    y_data = y_data[:data_length]
    x_data = x_data[:data_length]

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, shuffle=True)
    # x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.50, shuffle=True)

    return x_train, x_test, y_train, y_test