import logging
import datetime
import settings
import data_manager
import pandas as pd
#Backtesting에 사용할 수 있는 데이터 형태로 만들어줌, stock_code에 원하는 데이터 형식 넣어주면 out.csv 파일로 만들어줌
stock_code = 'BTCUSDT'
df=pd.read_csv('{0}_1h.csv'.format(stock_code))
df['o_t']=df['o_t']/1000
df['o_t']=pd.to_datetime(df['o_t'],unit='s')
df=df[['o_t','o_p','h_p','l_p','c_p','v']]
print (df.head())
df.to_csv('out.csv', encoding='utf-8',index=False, date_format='%Y-%m-%d %H:%M:%S')