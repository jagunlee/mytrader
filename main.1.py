import logging
import os
from mytrade1 import settings
from mytrade1 import data_manager
from mytrade1.policy_learner import PolicyLearner


if __name__ == '__main__':
    stock_code = 'output'
    model_ver = '20180712144711'

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
                     '{}.csv'.format(stock_code)))
    prep_data = data_manager.preprocess(chart_data)
    training_data = data_manager.build_training_data(prep_data)

   
    training_data = training_data[(training_data['o_t'] >= 1530352800000) &
                                  (training_data['o_t'] <= 1530842400000)]
    #training_data = training_data.dropna()
    print(training_data)

    features_chart_data = ['o_t', 'open', 'high', 'low', 'close', 'volume']
    chart_data = training_data[features_chart_data]


    features_training_data = ['ADA_c_p','ADX_c_p','AMB_c_p','ARK_c_p','ARN_c_p','AST_c_p','BAT_c_p','BCC_c_p','BCD_c_p','BCPT_c_p','BNB_c_p','BNT_c_p','BQX_c_p','BTG_c_p','BTS_c_p','CDT_c_p','CMT_c_p','CND_c_p','DASH_c_p','DGD_c_p','DLT_c_p','DNT_c_p','ELF_c_p','ENG_c_p','ENJ_c_p','EOS_c_p','ETC_c_p','ETH_c_p','EVX_c_p','FUEL_c_p','FUN_c_p','GAS_c_p','GTO_c_p','GVT_c_p','GXS_c_p','HSR_c_p','ICN_c_p','ICX_c_p','IOTA_c_p','KMD_c_p','KNC_c_p','LEND_c_p','LINK_c_p','LRC_c_p','LSK_c_p','LTC_c_p','MANA_c_p','MCO_c_p','MDA_c_p','MOD_c_p','MTH_c_p','MTL_c_p','NEO_c_p','NULS_c_p','OAX_c_p','OMG_c_p','OST_c_p','POE_c_p','POWR_c_p','PPT_c_p','QSP_c_p','QTUM_c_p','RCN_c_p','RDN_c_p','REQ_c_p','SALT_c_p','SNGLS_c_p','SNM_c_p','SNT_c_p','STORJ_c_p','STRAT_c_p','SUB_c_p','TNB_c_p','TNT_c_p','TRX_c_p','VEN_c_p','VIB_c_p','WABI_c_p','WAVES_c_p','WTC_c_p','XLM_c_p','XMR_c_p','XRP_c_p','XVG_c_p','XZC_c_p','YOYO_c_p','ZEC_c_p','ZRX_c_p',
        'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio',
        'close_ma10_ratio', 'volume_ma10_ratio',
        'close_ma20_ratio', 'volume_ma20_ratio',
        'close_ma60_ratio', 'volume_ma60_ratio',
        'close_ma120_ratio', 'volume_ma120_ratio'
    ]
    training_data = training_data[features_training_data]


    policy_learner = PolicyLearner(
        stock_code=stock_code, chart_data=chart_data, training_data=training_data)
    policy_learner.trade(balance=10000, model_path=os.path.join(settings.BASE_DIR, 'models/{}/model_{}.h5'.format(stock_code, model_ver)))

    '''
    model_dir = os.path.join(settings.BASE_DIR, 'models/%s' % stock_code)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model_%s.h5' % timestr)
    policy_learner.policy_network.save_model(model_path)
    '''