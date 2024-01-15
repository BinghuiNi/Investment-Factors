import sys
import time
import numpy as np
import pandas as pd
import datetime
import pickle
from dateutil.parser import parse
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from fastcache import lru_cache

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

path_factors = r'C:\Users\nibh\Desktop\FZ\daily_factor'
path_temp = r'C:\Users\nibh\Desktop\FZ\daily_factor\temp'

# DATA COLLECTION
from fastcache import lru_cache
import dcube as dc
pro = dc.pro_api(token)

@lru_cache(1)
def get_all_trade_days(start_date='20000101'):
    df = pro.trade_cal(exchange='', start_date=pd.to_datetime(start_date).strftime('%Y%m%d'),is_open='1')
    trade_days = df['cal_date'].tolist()
    all_days = pd.to_datetime(trade_days)
    all_days = [date.date() for date in all_days]
    return all_days

@lru_cache(1)
def get_near_trade_day(date,forward=True):
    trade_days = get_all_trade_days()
    date = pd.to_datetime(date).date()
    while date not in trade_days:
        if forward:
            date = (date - datetime.timedelta(days=1))
        else:
            date = (date + datetime.timedelta(days=1))
    return date
@lru_cache(1)

def get_aindex_daily(code: str = '000906.SH', start_date: str = '20001026', end_date: str = '20230426', elements = None):
    if not elements:
        df = pro.query('aindex_daily', code = code, fields='code,trade_date,close,preclose,pctchange')
    else:
        df = pro.query('aindex_daily', code = code, fields=elements)
    df.index = [datetime.datetime.strptime(i, '%Y%m%d').date() for i in df['trade_date']]
    df = df[(df['trade_date']<=end_date) & (df['trade_date']>=start_date)]
    return df.sort_index()

def get_stock_mins(code = '601901.SH', freq = '1min', start_date = '20100101', end_date = '20230415', fields = 'tradetime, code,openprice,closeprice, volume'):
    df = pro.query('stk_mins', code=code, freq=freq, exchange='', start_date=start_date, end_date=end_date,fields=fields)
    times = pd.Series([datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.000Z") for time in df['tradetime']])
    df.index = times
    dates = pd.Series([time.date() for time in times], index = df.index)
    df['date'] = dates
    df = df.sort_index()
    df = df.drop('tradetime', axis = 1)
    return df

def get_dcindex_df(code):
    if code.endswith('.WI'):
        index_df = pro.query('a_wdindex_members', code=code)
    elif code.endswith('.SI'):
        index_df = pro.query('index_member', index_code=code)
    elif code.endswith('.CI'):
        index_df = pro.query('a_index_memberscitics', index_code=code)
    else:
        index_df = pro.query('aindex_members', code=code)
    if 'in_date' in index_df.columns:
        indate_name = 'in_date'
    elif 'indate' in index_df.columns:
        indate_name = 'indate'
    pub_date = sorted(index_df[indate_name].tolist())[0]
    index_df[indate_name] = index_df[indate_name].replace(pub_date,'20040101')
    return index_df

def get_index_stocks_multdays(index_pool, start_date: str, end_date: str):
    start_date = pd.to_datetime(str(start_date)).date()
    end_date = pd.to_datetime(str(end_date)).date()
    trade_days = [date for date in get_all_trade_days() if start_date<=date<=end_date]
    index_info = get_dcindex_df(index_pool)
    pool_multdays = dict.fromkeys(trade_days)
    for date in trade_days:
        date_str = datetime.date.strftime(date, '%Y%m%d')
        df_one_day = index_info[(index_info['in_date']<=date_str) & ((index_info['out_date']>=date_str) | (index_info['cur_sign']==1))]
        pool_multdays[date] = df_one_day['con_code'].tolist()
    return pool_multdays

@lru_cache(8)
def read_single_factor_file_nocache(factor_name: str):
    path = r'C:\Users\nibh\Desktop\FZ\daily_factor\{}.csv'.format(factor_name)
    df = pd.read_csv(path, index_col=0)
    df.index = [datetime.datetime.strptime(i, '%Y-%m-%d').date() for i in df.index]
    return df
def get_single_factor_values(factor_name: str,start_date: datetime.date ,end_date: datetime.date,pool: list,cache=False):
    df = read_single_factor_file_nocache(factor_name)
    return df.loc[start_date:end_date, pool]

def read_pickle(date: str):
    try:
        path = r'E:/FZquan/mbars/' + date + '.pkl'
        with open(path, 'rb') as f:
            panal = pickle.load(f)
    except FileNotFoundError:
        path = r'E:/FZquan/k1mbars/' + date + '.pkl'
        with open(path, 'rb') as f:
            panal = pickle.load(f)
    return panal

# DEBUG
def nan(series: pd.Series):
    num = np.sum(np.isnan(series))
    ratio = num / len(series)
    print('Num of np.nan is:', num)
    print('Ratio of np.nan is:', ratio)
    return None
def scaler_byrow(df: pd.DataFrame):
    df1 = df.apply(lambda x: (x - np.nanmean(x)) / np.nanstd(x), axis = 1)
    return df1

def Sharpe(net_value, count=250):
    count = float(count)
    net_value = net_value.fillna(method='bfill')
    net_value = net_value / net_value.iloc[0]
    bench_pct = 0.0
    df_tmp = pd.DataFrame(net_value)
    df_tmp.columns = ['value']
    df_tmp['pct'] = df_tmp['value'].pct_change()
    annual_pct = df_tmp.ix[-1, 'value'] ** (count / len(df_tmp)) - 1
    sharpe = (annual_pct - bench_pct) / (df_tmp['pct'].std() * count ** 0.5)
    return sharpe
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




# SINGAL FACTOR ANALYSIS
class SingleFactorAnalysis(object):
    def __init__(self, index_pool, start_date, end_date, factor_name, nextpct, long_low='adaptation', neu_style=None,
                 dt_index=None, calc_crowd=False, ret_save=False, n_group=10, choose_group='group_01'):
        start_date = pd.to_datetime(str(start_date)).date()
        end_date = pd.to_datetime(str(end_date)).date()
        self.start_date = start_date
        self.end_date = end_date
        self.factor_name = factor_name
        print('factor_name: {}'.format(factor_name))
        self.nextpct = nextpct
        self.all_days = get_all_trade_days()
        # print('all_days',self.all_days)
        print('start_date,end_date', start_date, end_date)
        self.date_list = [date for date in self.all_days if start_date <= date <= end_date]
        self.index_pool = index_pool
        self.stock_pool_dict = get_index_stocks_multdays(index_pool, start_date, end_date)
        self.n_group = n_group
        self.long_low = long_low
        self.dt_index = dt_index
        self.neu_style = neu_style
        self.ret_save = ret_save

        # DRAW PERFORMANCE GRAPH
        self.choose_group = choose_group

        if index_pool == 'WIND_A_NOST':
            calc_crowd = True
        self.calc_crowd = calc_crowd

        if self.dt_index is not None:
            if dt_index == '240m':
                self.price2preclose_df = read_single_factor_file_nocache('EODP__S_DQ_PCTCHANGE') / 100
            elif dt_index == '0m':
                self.price2preclose_df = read_single_factor_file_nocache(
                    'EODP__S_DQ_ADJOPEN') / read_single_factor_file_nocache('EODP__S_DQ_ADJPRECLOSE') - 1
            else:
                self.price2preclose_df = read_single_factor_file_nocache('price2preclose_{}'.format(self.dt_index))

    def get_stocks(self, date):
        all_days = self.all_days
        di = all_days.index(date)
        date_tmp = all_days[di + self.nextpct_shift]
        if self.index_pool is None:
            stock_list = get_single_factor_values(self.factor_name, date, date).iloc[0].dropna().index
        else:
            stock_list = self.stock_pool_dict[date]
        #         stock_list = get_history_security_pool(start_date=date,end_date=date,index=self.index_pool)
        #         stock_list = state_filter(stock_list,date_tmp)
        if self.dt_index is not None:
            # print('date_tmp:{}'.format(date_tmp)
            price2preclose_oneday = self.price2preclose_df.loc[date_tmp]
            # DEL STOCKS WITH LIMIT MOVE
            stock_list_after_filter = price2preclose_oneday[price2preclose_oneday.abs() <= 0.099].reindex(
                stock_list).dropna().index.tolist()
            global test_1
            test_1 = self.price2preclose_df
            num = len(stock_list) - len(stock_list_after_filter)
            # print(date, 'NUM OF STOCKS WITH LIMIT MOVE:%s,NUM OF ALL STOCKS:%s'%(num,len(stock_list_after_filter)))
            return stock_list_after_filter
        else:
            return stock_list

    def run_code(self, nextpct_shift=0):
        df_corr = pd.DataFrame()
        df_ret_delay = pd.DataFrame()
        pct_group = pd.DataFrame()
        if self.calc_crowd:
            turnover_group = pd.DataFrame()
            pctstd_group = pd.DataFrame()
            beta_group = pd.DataFrame()
        pct_bench = pd.Series()
        periods = 1
        date_list = self.date_list
        n_group = self.n_group
        c_1 = self.factor_name
        c_2 = self.nextpct
        self.nextpct_shift = nextpct_shift
        long_low = self.long_low
        all_days = self.all_days
        factor_name = self.factor_name
        all_stock_pool = []

        for date in date_list[:]:
            stock_pool = self.get_stocks(date)
            all_stock_pool = list(set(all_stock_pool) | set(stock_pool))
        if self.neu_style is None:
            # print(all_stock_pool)
            self.factor_df_allstocks = get_single_factor_values(c_1, date_list[0], date_list[-1], pool=all_stock_pool,
                                                                cache=False)
            factor_df = self.factor_df_allstocks.T.reindex(all_stock_pool).T
            # factor_df = get_single_factor_values(c_1,date_list[0],date_list[-1],all_stock_pool,cache=False)
        else:
            factor_df = get_BarraNeu_factor_parr(c_1, date_list[0], date_list[-1], style=self.neu_style, cache=False)
        self.factor_df = factor_df

        date_list_exist = factor_df.dropna(how='all', axis=1).index.tolist()
        if c_2 == 'pct_close_next_close':
            nextpct_df = get_single_factor_values('EODP__S_DQ_PCTCHANGE', all_days[all_days.index(date_list[0]) + 1],
                                                  all_days[all_days.index(date_list[-1]) + nextpct_shift + 1],
                                                  all_stock_pool) / 100
            nextpct_df.index = pd.to_datetime([all_days[all_days.index(date) - 1] for date in nextpct_df.index])

        elif c_2 == 'pct_open_next_open':
            nextpct_df = get_single_factor_values('EODP__S_DQ_ADJOPEN', all_days[all_days.index(date_list[0]) + 1],
                                                  all_days[all_days.index(date_list[-1]) + nextpct_shift + 1],
                                                  all_stock_pool).pct_change()
            nextpct_df.index = pd.to_datetime([all_days[all_days.index(date) - 1] for date in nextpct_df.index])
        else:
            nextpct_df = get_single_factor_values(c_2, date_list[0],
                                                  all_days[all_days.index(date_list[-1]) + nextpct_shift],
                                                  all_stock_pool)
        if self.calc_crowd:
            turnover_df = get_single_factor_values('EODDI__S_DQ_TURN', all_days[all_days.index(date_list[0]) - 250],
                                                   all_days[all_days.index(date_list[-1])], all_stock_pool)
            turnover_df.index = pd.to_datetime(turnover_df.index)
            turnover_df = turnover_df.rolling(60).mean()
            pct_df = get_single_factor_values('EODP__S_DQ_PCTCHANGE', all_days[all_days.index(date_list[0]) - 250], \
                                              all_days[all_days.index(date_list[-1])], all_stock_pool) / 100
            pct_df.index = pd.to_datetime(pct_df.index)
            pctstd_df = pct_df.rolling(60).std()
            beta_df = get_single_factor_values('beta800_60d', all_days[all_days.index(date_list[0]) - 250], \
                                               all_days[all_days.index(date_list[-1])], all_stock_pool)
            beta_df.index = pd.to_datetime(beta_df.index)
        self.nextpct_df = nextpct_df

        stock_group = {}

        for date in tqdm(date_list[:]):
            #             if list(date_list).index(date)%250==0:
            #                 print (date)
            self.di = all_days.index(date)
            self.nextpct_date = all_days[self.di + nextpct_shift]
            stock_pool = self.get_stocks(date)
            factor = factor_df.loc[date, stock_pool]

            nextpct_oneday = nextpct_df.loc[self.nextpct_date, stock_pool]

            if self.calc_crowd:
                turnover_oneday = turnover_df.loc[date, stock_pool]
                pctstd_oneday = pctstd_df.loc[date, stock_pool]
                beta_oneday = beta_df.loc[date, stock_pool]
                df_tot = pd.concat([
                    factor.to_frame(c_1), nextpct_oneday.to_frame(c_2),turnover_oneday.to_frame('turnover'),
                    pctstd_oneday.to_frame('pctstd'),beta_oneday.to_frame('beta'),], axis=1)
            else:
                df_tot = pd.concat([
                    factor.to_frame(c_1),nextpct_oneday.to_frame(c_2),], axis=1)
            # print(df_tot.head(), date)
            df_tot = df_tot.fillna(df_tot.median())
            df_cop_sort = df_tot.sort_values(c_1)
            total_num = float(len(df_tot))
            group_num = float(len(df_tot)) / n_group
            stock_group_bydate = {}
            # print('{},group_num:{}'.format(date,group_num))
            for i in range(n_group):
                left_i = int(total_num * i / n_group)
                right_i = int(total_num * (i + 1) / n_group)
                stock_group_bydate['group_' + str(i + 1).zfill(2)] = df_cop_sort.iloc[left_i:right_i].index.tolist()
                pct_group.loc[date, 'group_' + str(i + 1).zfill(2)] = df_cop_sort.iloc[left_i:right_i][c_2].mean()
                if self.calc_crowd:
                    turnover_group.loc[date, 'group_' + str(i + 1).zfill(2)] = df_cop_sort.iloc[left_i:right_i][
                        'turnover'].mean()
                    pctstd_group.loc[date, 'group_' + str(i + 1).zfill(2)] = df_cop_sort.iloc[left_i:right_i][
                        'pctstd'].mean()
                    beta_group.loc[date, 'group_' + str(i + 1).zfill(2)] = df_cop_sort.iloc[left_i:right_i][
                        'beta'].mean()
            pct_bench[date] = nextpct_oneday.mean()
            stock_group[date] = stock_group_bydate

        self.factor = factor
        self.pct_group = pct_group
        if self.calc_crowd:
            self.turnover_group = turnover_group
            self.pctstd_group = pctstd_group
            self.beta_group = beta_group
        nv_ratio_group = pct_group + 1
        self.nv_ratio_group = nv_ratio_group
        nv_group = pct_group.cumsum() + 1
        self.nv_group = nv_group
        # nv_group.plot(figsize=(16,3))
        excess_nv_group = (nv_group.T - nv_group.mean(axis=1)).T
        self.excess_nv_group = excess_nv_group
        excess_nv_group.plot(figsize=(16, 3), grid='on', title='factor')

        choose_group = self.choose_group
        high_ret = pct_group['group_' + str(n_group).zfill(2)].cumsum()
        low_ret = pct_group[choose_group].cumsum()

        high_ret_cumprod = (pct_group['group_' + str(n_group).zfill(2)] + 1).cumprod()
        low_ret_cumprod = (pct_group[choose_group] + 1).cumprod()

        high2bench_ret = pct_group['group_' + str(n_group).zfill(2)] - pct_bench
        low2bench_ret = pct_group[choose_group] - pct_bench

        high2bench = pct_group['group_' + str(n_group).zfill(2)].cumsum() - pct_bench.cumsum()
        low2bench = ((pct_group[choose_group].cumsum() - pct_bench.cumsum()))
        low2high = -((pct_group['group_' + str(n_group).zfill(2)].cumsum() - pct_group[choose_group].cumsum()))

        self.high2bench = high2bench
        self.low2bench = low2bench
        self.low2high = low2high
        self.high2low = pct_group['group_' + str(n_group).zfill(2)].cumsum() - pct_group[choose_group].cumsum()
        self.high2low_cumprod = (pct_group['group_' + str(n_group).zfill(2)] + 1).cumprod() / (
                    pct_group[choose_group] + 1).cumprod()
        if self.calc_crowd:
            self.turnover_high2low = turnover_group['group_' + str(n_group).zfill(2)] / turnover_group[choose_group]
            self.pctstd_high2low = pctstd_group['group_' + str(n_group).zfill(2)] / pctstd_group[choose_group]
            self.beta_high2low = beta_group['group_' + str(n_group).zfill(2)] / beta_group[choose_group]
        plt.figure()
        if long_low is None:
            if high2bench.iloc[-1] > low2bench.iloc[-1]:
                long_low = 0
            else:
                long_low = 1
        if long_low == 1:
            long2bench = low2bench.copy()
            long_ret = low_ret.copy()
            long_ret_cumprod = low_ret_cumprod.copy()
        else:
            long2bench = high2bench.copy()
            long_ret = high_ret.copy()
            long_ret_cumprod = high_ret_cumprod.copy()

        self.long2bench = long2bench
        self.long_ret = long_ret
        self.long_ret_cumprod = long_ret_cumprod
        long2bench.plot(figsize=(16, 3), grid='on', title='long2bench_net_value_cumsum')
        plt.figure()
        long_ret.plot(figsize=(16, 3), grid='on', title='long_net_value_cumsum')
        plt.figure()
        long_ret_cumprod.plot(figsize=(16, 3), grid='on', title='long_net_value_cumprod')

        ret_record_df = pd.DataFrame()
        ret_record_df['long2bench'] = long2bench
        ret_record_df['long_ret'] = long_ret
        ret_record_df['long_ret_cumprod'] = long_ret_cumprod
        if self.ret_save:
            path = 'D:\\jupyter_script\\Quant\\FZZQ\\factor_ret_save'
            if not os.path.exists(path):
                os.makedirs(path)
            file_name = '{}\\{}_RetRecord.csv'.format(path, factor_name)
            ret_record_df.to_csv(file_name)
        df_corr = pd.DataFrame()
        df_ret_delay = pd.DataFrame()
        commision = 1.4e-3
        periods = 1
        if long_low == 1:
            long_ret = low2bench_ret.mean() - 1
        else:
            long_ret = high2bench_ret.mean() - 1
        self.stock_group = stock_group
        date_list = sorted(stock_group.keys())
        stock_group_change_ratio = pd.DataFrame()
        for date in date_list[1:]:
            pre_date = date_list[date_list.index(date) - 1]
            group_num = sorted(stock_group[date].keys())
            for group in group_num:
                stock_group_i = stock_group[date][group]
                stock_group_i_pre = stock_group[pre_date][group]
                stocks_num = (len(set(stock_group_i_pre)) * 0.5 + len(set(stock_group_i)) * 0.5)
                if stocks_num:
                    stock_group_change_ratio.loc[date, group] = len(
                        set(stock_group_i_pre) - set(stock_group_i)) / stocks_num
                else:
                    stock_group_change_ratio.loc[date, group] = 0
        stock_group_change_ratio_mean = stock_group_change_ratio.mean()
        stock_group_change_commision = stock_group_change_ratio * commision
        excess_pct = pct_group.mean() - pct_group.mean().mean()
        excess_pct_net = excess_pct - stock_group_change_commision.mean()
        excess_pct = pd.concat([stock_group_change_ratio_mean, excess_pct, excess_pct_net], axis=1)
        excess_pct.columns = ['turnover', 'gross', 'net']

        print(f'Sharpe ratio of long_ret_cumprod is: {Sharpe(net_value=long_ret_cumprod): .3f}')  # long2bench
        print(f'Sharpe ratio of long2bench is: {Sharpe(net_value=long2bench): .3f}')  # long2bench
        print(f'Max drawdown of long_ret_cumprod is: {max_drawdown(long_ret_cumprod) * 100: .2f}%')

        pd.set_option('display.width', 1000)
        return excess_pct






