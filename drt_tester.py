from __future__ import (absolute_import, division, print_function,unicode_literals)
import logging
import os
import settings
import data_manager
from policy_learner import PolicyLearner
import numpy as np
import pandas as pd
import backtrader as bt
import datetime
import os.path
import sys
import backtrader.indicators as btind
import locale
from environment import Environment
from agent_custom import Agent
from policy_network import PolicyNetwork
import math
import matplotlib.pyplot as plt

'''
기존의 main 함수에 backtesting을 합쳐서 drt를 변경해주며 돌려서 데이터를 저장해주는 코드
'''
if __name__ == '__main__':
    stock_code = 'BTCUSDT'
    #drt, 매수 횟수, 매도 횟수, confusion matrix의 accuracy/precision/recall과 최종 backtesting 포트폴리오 가치를 저장
    drts=[]
    buy=[]
    sell=[]
    acc=[]
    pre=[]
    rec=[]
    pv=[]
    for drt in range(10,90):
        drt=drt/100.
        drts.append(drt)
        log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
        timestr = settings.get_time_str()
        if not os.path.exists('logs/%s' % stock_code):
            os.makedirs('logs/%s' % stock_code)
        file_handler = logging.FileHandler(filename=os.path.join(
            log_dir, "%s_%s.log" % (stock_code, timestr)), encoding='utf-8')
        stream_handler = logging.StreamHandler()
        file_handler.setLevel(logging.DEBUG)
        stream_handler.setLevel(logging.INFO)
        logging.basicConfig(format="%(message)s",
                        handlers=[file_handler, stream_handler], level=logging.DEBUG)

    
        chart_data = data_manager.load_chart_data(
            os.path.join(settings.BASE_DIR,
                         '{}'.format(stock_code)))
        prep_data = data_manager.preprocess(chart_data)
        training_data = data_manager.build_training_data(prep_data)

   
        training_data = training_data.loc['2018-01-01 01:00:00':]
    


        features_chart_data = ['o_t', 'open', 'high', 'low', 'close', 'volume']
        chart_data = training_data[features_chart_data]

        '''
'ADA_c_p','ADX_c_p','AMB_c_p','ARK_c_p','ARN_c_p','AST_c_p','BAT_c_p','BCC_c_p','BCD_c_p','BCPT_c_p','BNB_c_p','BNT_c_p','BQX_c_p','BTG_c_p','BTS_c_p','CDT_c_p','CMT_c_p','CND_c_p','DASH_c_p','DGD_c_p','DLT_c_p','DNT_c_p','ELF_c_p','ENG_c_p','ENJ_c_p','EOS_c_p','ETC_c_p','ETH_c_p','EVX_c_p','FUEL_c_p','FUN_c_p','GAS_c_p','GTO_c_p','GVT_c_p','GXS_c_p','HSR_c_p','ICN_c_p','ICX_c_p','IOTA_c_p','KMD_c_p','KNC_c_p','LEND_c_p','LINK_c_p','LRC_c_p','LSK_c_p','LTC_c_p','MANA_c_p','MCO_c_p','MDA_c_p','MOD_c_p','MTH_c_p','MTL_c_p','NEO_c_p','NULS_c_p','OAX_c_p','OMG_c_p','OST_c_p','POE_c_p','POWR_c_p','PPT_c_p','QSP_c_p','QTUM_c_p','RCN_c_p','RDN_c_p','REQ_c_p','SALT_c_p','SNGLS_c_p','SNM_c_p','SNT_c_p','STORJ_c_p','STRAT_c_p','SUB_c_p','TNB_c_p','TNT_c_p','TRX_c_p','VEN_c_p','VIB_c_p','WABI_c_p','WAVES_c_p','WTC_c_p','XLM_c_p','XMR_c_p','XRP_c_p','XVG_c_p','XZC_c_p','YOYO_c_p','ZEC_c_p','ZRX_c_p',
        '''
        features_training_data = [

         'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio',
        'close_ma10_ratio', 'volume_ma10_ratio',
        'close_ma20_ratio', 'volume_ma20_ratio',
        'close_ma60_ratio', 'volume_ma60_ratio',
        'close_ma120_ratio', 'volume_ma120_ratio',
        'ema12','ema26','dn','mavg','up','pctB','macd','signal','cci'
        ]
        training_data = training_data[features_training_data]
        #print (training_data[:3])   
        training_data = training_data.dropna(axis=1)
        chart_data=chart_data.dropna(axis=1)
        chart_data = chart_data.loc[:'2018-06-30 23:00:00']
        training_data = training_data.loc[:'2018-06-30 23:00:00']
        delayed_reward_threshold=0.2
        policy_learner = PolicyLearner(
            stock_code=stock_code, chart_data=chart_data, training_data=training_data,  delayed_reward_threshold=drt, lr=.1)
        policy_learner.fit(balance=10000, num_epoches=10,discount_factor=0, start_epsilon=.5)
        

        model_dir = os.path.join(settings.BASE_DIR, 'models/%s' % stock_code)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'model_%s.h5' % timestr)
        policy_learner.policy_network.save_model(model_path)
        
        
        
        buycnt=0
        sellcnt=0
        confmat=[[0,0],[0,0]]
        class TestStrategy(bt.Strategy):
            
            def log(self, txt, dt=None):
                dt = dt or self.datas[0].datetime.datetime(0)
                print('%s, %s' % (dt, txt))

            def __init__(self):
                stock_code='BTCUSDT'
                global timestr #global 함수로 불러와 바로 사용
                model_ver=timestr
                self.dataclose = self.datas[0].close
                log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
                timestr = settings.get_time_str()
                if not os.path.exists('logs/%s' % stock_code):
                    os.makedirs('logs/%s' % stock_code)
                file_handler = logging.FileHandler(filename=os.path.join(
                    log_dir, "%s_%s.log" % (stock_code, timestr)), encoding='utf-8')
                stream_handler = logging.StreamHandler()
                file_handler.setLevel(logging.DEBUG)
                stream_handler.setLevel(logging.INFO)
                logging.basicConfig(format="%(message)s",
                        handlers=[file_handler, stream_handler], level=logging.DEBUG)

    
                chart_data = data_manager.load_chart_data(
                    os.path.join(settings.BASE_DIR,
                             '{}'.format(stock_code)))
                prep_data = data_manager.preprocess(chart_data)
                training_data = data_manager.build_training_data(prep_data)

   
                training_data = training_data.loc['2018-07-01 01:00:00':]
    


                features_chart_data = ['o_t', 'open', 'high', 'low', 'close', 'volume']
                chart_data = training_data[features_chart_data]


                features_training_data = [
             
         'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio',
        'close_ma10_ratio', 'volume_ma10_ratio',
        'close_ma20_ratio', 'volume_ma20_ratio',
        'close_ma60_ratio', 'volume_ma60_ratio',
        'close_ma120_ratio', 'volume_ma120_ratio',
        'ema12','ema26','dn','mavg','up','pctB','macd','signal','cci'
                                            ]
                training_data = training_data[features_training_data]
                #print (training_data[:3])   
                training_data = training_data.dropna(axis=1)
                chart_data=chart_data.dropna(axis=1)
                #chart_data = chart_data.loc[:1530352800000]
                #training_data = training_data.loc[:1530352800000]
                delayed_reward_threshold=.001
                lr=0.1
                self.TRADING_TAX =0
                self.TRADING_CHARGE=0
                self.stock_code = stock_code  
                self.chart_data = chart_data
                self.environment = Environment(chart_data)  
                self.agent = Agent(self.environment, delayed_reward_threshold=delayed_reward_threshold)
                self.training_data = training_data  
                self.sample = None
                self.pvdata=[]
                self.training_data_idx = -1
                self.num_features = self.training_data.shape[1] #+ self.agent.STATE_DIM
                self.policy_network = PolicyNetwork(
                input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=lr)
                model_path=os.path.join(settings.BASE_DIR,
                'models/{}/model_{}.h5'.format(stock_code,model_ver))
                self.policy_network.load_model(model_path=model_path)
                self.agent.set_balance(self.broker.getcash())
                self.epsilon=0
                self.num_stocks=0
                df=pd.read_csv('out.csv')
                self.stopit=df.loc[7692]['c_p'] #2903 4135 7692
        
        
            def next(self):
                global buycnt
                global sellcnt
                global confmat
                self.log('Close, %.2f' % self.dataclose[0])
                self.environment.observe()
                if len(self.training_data) > self.training_data_idx + 1:
                    self.training_data_idx += 1
                    self.sample = self.training_data.iloc[self.training_data_idx].tolist()
                next_sample= self.sample
                if next_sample is None:
                    return
                #print(self.sample)
                action, confidence, exploration = self.agent.decide_action(
                    self.policy_network, self.sample, self.epsilon)
                #print(action,confidence,exploration)
                if self.dataclose[0]==self.stopit:
                    next_price=self.dataclose[0]
                else:
                    next_price=self.dataclose[1]
                #next_price=self.dataclose[1]
                if action==Agent.ACTION_SELL:
                    sellcnt+=1
                    if self.dataclose[0]>next_price:
                        confmat[0][0]+=1
                    else:
                        confmat[0][1]+=1
                    #self.log('SELL CREATE, %.2f' % self.dataclose[0])
                    trading_unit = self.num_stocks*math.exp(float(confidence)-1)
                    invest_amount = next_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
                    self.num_stocks -= trading_unit  
                    self.sell(size=trading_unit)
                    
                    self.log('sell')

                elif action==Agent.ACTION_BUY:
                    buycnt+=1
                    if self.dataclose[0]<next_price:
                        confmat[1][1]+=1
                    else:
                        confmat[1][0]+=1
                    #self.log('BUY CREATE, %.2f' % self.dataclose[0])
                    trading_unit = self.broker.getcash()*math.exp(confidence-1)/(float(next_price)*(1+self.TRADING_CHARGE))  
                    self.num_stocks += trading_unit
                    self.buy(size=trading_unit)
                    self.log('buy')
                self.log('value : %.2f , cash : %.2f , action : %.2f , confidence : %.2f , stock : %.2f' % (self.broker.getvalue(),self.broker.getcash(),action, confidence,self.num_stocks))
                
        cerebro = bt.Cerebro(stdstats=True)
        cerebro.addstrategy(TestStrategy)
        modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
        data = bt.feeds.GenericCSVData(
            dataname='out.csv',

            fromdate=datetime.datetime(2018,6,30,23,0,0),
            timeframe=bt.TimeFrame.Minutes, compression=240,
            nullvalue=0.0,
            dtformat=('%Y-%m-%d %H:%M:%S'),
            datetime=0,
            high=1,
            low=2,
            open=3,
            close=4,
            volume=5,
            openinterest=-1
        )
        '''
        fromdate=datetime.datetime(2018,1,15,2,0,0),
        todate=datetime.datetime(2018,2,5,19,0,0),
    
        fromdate=datetime.datetime(2017,11,24,23,0,0),
        todate=datetime.datetime(2017,12,16,10,0,0),
        '''
        cerebro.adddata(data)
        cerebro.broker.setcash(100000.0)
        cerebro.addsizer(bt.sizers.FixedSize, stake=10)
        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

        cerebro.run()

        print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
        pv.append(cerebro.broker.getvalue())
        buy.append(buycnt)
        sell.append(sellcnt)
        acc.append((confmat[0][0]+confmat[1][1])/(confmat[0][0]+confmat[0][1]+confmat[1][0]+confmat[1][1])) if confmat[0][0]+confmat[0][1]+confmat[1][0]+confmat[1][1]!=0 else acc.append(0)
        pre.append(confmat[1][1]/(confmat[1][1]+confmat[1][0])) if confmat[1][1]+confmat[1][0]!=0 else pre.append(0)
        rec.append(confmat[1][1]/(confmat[1][1]+confmat[0][1])) if confmat[1][1]+confmat[0][1]!=0 else rec.append(0)
        cerebro.plot(iplot=True)[0][0].savefig('plot_drt{}.png'.format(drt))
        print(drts,sell,buy,pv,acc,pre,rec)
    
        my_dict = {"drt": drts, "buy":buy, "sell": sell, "acc":acc,"pre":pre,"rec":rec,"pv":pv}
        datas=pd.DataFrame(my_dict)
        datas.to_csv('checkdata.csv', index='False')
        #매번 drt를 바꿀때마다 csv파일로 저장
    plt.plot(drts,buy,label="buy")
    plt.plot(drts,sell,label="sell")
    plt.plot(drts,acc,label="acc")
    plt.plot(drts,pre,label="pre")
    plt.plot(drts,rec,label="rec")
    plt.plot(drts,pv,label="pv")
    plt.legend()
    plt.savefig('checkdata.png')
    #최종적으로 png파일로 저장
