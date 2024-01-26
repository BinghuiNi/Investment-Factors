import pandas as pd
import os
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.chdir('E:\\FZquan\\mbars')

def read_mbars(date):
    date = pd.to_datetime(date).date()
    path = 'E:\\FZquan\\mbars'
    file_name = '{}\\{}.pkl'.format(path, date.strftime("%Y%m%d"))
    if_file_exist = os.path.exists(file_name)
    if if_file_exist:
        mbars = pd.read_pickle(file_name).astype(float)
        mbars.rename(items={'openprice': 'open', 'closeprice': 'close', 'highprice': 'high', 'lowprice': 'low',
                            'value': 'money'}, inplace=True)
        mbars.rename(items={'openPrice': 'open', 'closePrice': 'close', 'highPrice': 'high', 'lowPrice': 'low',
                            'totalVolume': 'volume', 'totalValue': 'money'}, inplace=True)
        return mbars
    else:
        return pd.Panel()

sample = read_mbars('20230418')
sample['volume'].replace(0,np.nan)
index=[i[0:8] for i in os.listdir()]
z = pd.DataFrame(index=index,columns=sample.minor_axis)
for i in tqdm(index):
    mbars = read_mbars(i)
    mbars['volume']=mbars['volume'].replace(0,np.nan)
    std = mbars['volume'].std(skipna=True)
    mean = mbars['volume'].mean(skipna=True)
    z.loc[i,:] = std / mean

def max_drawdown(series, name = 'net value'):
    if name == 'net value':
        net_value_index = series
    elif name == 'return':
        net_value_index = (1 + series).cumprod()
    else:
        print('illegal name!')

    peak_index = net_value_index.cummax()
    drawdown = (net_value_index - peak_index) / peak_index
    return drawdown.min()

%run Single_Factor_Analysis.py
C = SingleFactorAnalysis('000985.SH', '20190118', '20221231', 'URet_Factor', 'pct_close_next_close', long_low=0, dt_index='240m', n_group=10, neu_style=None, calc_crowd=False)
result_df = C.run_code(nextpct_shift=1)
result_df
