import pandas as pd
import os
import datetime
import numpy as np

def load_chart_data(filename):
    df=pd.read_csv('{0}_1h.csv'.format(filename))
    df=df.set_index('o_t')
    df=df[['o_p','h_p','l_p','c_p','v']]
    df.columns=['open', 'high', 'low', 'close', 'volume']
    df=df.reindex(index=df.index[::-1])
    df.head()
    check = 'BTC'
    path_dir = 'binance'
    file_list = os.listdir(path_dir)
    cut=len(df)
    for files in file_list:
        if files.find(check) is not -1 and files[len(check)] != check and files.find(filename) is -1:
            df1=pandas.read_csv('binance/{0}'.format(files))
            df1.set_index('o_t')
            df1 = df1.reindex(index=df1.index[::-1])
            df1.head()
            name=files[:]
            if len(df)<len(df1):
                df1=df1[-cut:]
            df['{0}_c_p'.format(name.replace('BTC_1h.csv', ''))] = np.nan
            df['{0}_c_p'.format(name.replace('BTC_1h.csv',''))] = df1[['o_t','c_p']].set_index(('o_t'))
            df['{0}_c_p'.format(name.replace('BTC_1h.csv', ''))] =df['{0}_c_p'.format(name.replace('BTC_1h.csv',''))].mul(df['close'])
    df=df.reindex(index=df.index[::-1])
    df.head()
    a=df.iloc[1,8]
    #df.fillna(value=0, inplace=True)
    df.to_csv('output.csv', index='False')


makedtf('BTCUSDT')