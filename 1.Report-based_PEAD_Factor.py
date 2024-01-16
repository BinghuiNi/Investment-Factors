import pandas as pd
import datetime
from fastcache import lru_cache
import dcube as dc
import numpy as np
import sys
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
token = ''
pro = dc.pro_api(token)

stock = pd.read_csv(r'C:\Users\nibh\Desktop\FZ\daily_factor\EODP__S_DQ_PCTCHANGE.csv', index_col=0)
df_annc=pd.read_csv(r'C:\Users\nibh\Desktop\FZ\Performance_Preannouncement.csv',index_col=0)
df_report=pd.read_csv(r'C:\Users\nibh\Desktop\FZ\Research_Report.csv',index_col=0,dtype={'est_dt':str})
df_preclose=pd.read_csv(r'C:\Users\nibh\Desktop\FZ\daily_factor\EODP__S_DQ_ADJPRECLOSE.csv',index_col=0)
df_csi=pd.read_csv(r'C:\Users\nibh\Desktop\FZ\daily_factor\指数收盘价.csv',index_col=0)
df_csi=df_csi.drop('HS300',axis=1)
df_csi.index = [datetime.datetime.strptime(i, '%Y-%m-%d').date() for i in df_csi.index]
df_preclose.index = [datetime.datetime.strptime(i, '%Y-%m-%d').date() for i in df_preclose.index]

# COLLECT SELL-SIDE REPORTS DATA
df=pd.DataFrame()
for i in stock.columns:
    df1 = pro.forecast(ts_code=i, start_date='20110101', end_date='20211231', fields='ts_code,ann_date')
    df = df.append(df1, ignore_index=False)
df.index = df['ts_code']
df['ann_date']=[datetime.datetime.strptime(i, '%Y%m%d').date() for i in df['ann_date']]
df_annc = df.copy()

start_date = '20110101'
end_date = '20211231'
df = pro.query('ashare_earning_est',start_date='20110101', end_date='20211231', fields='s_info_windcode, est_dt,report_name,report_summary')
date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

for i in range(len(date_range)):
    start = date_range[i].strftime('%Y%m%d')
    end = (date_range[i] + pd.DateOffset(months=1) - pd.DateOffset(days=1)).strftime('%Y%m%d')
    df1 = pro.query('ashare_earning_est',start_date=start, end_date=end, fields='s_info_windcode,est_dt,report_name,report_summary')
    df=df.append(df1,ignore_index=True)
df_report=df.drop_duplicates()

df_annc.loc[:, 'ann_date'] = [datetime.datetime.strptime(i, '%Y-%m-%d').date() for i in df_annc['ann_date']]
df_report.loc[:, 'est_dt'] = [datetime.datetime.strptime(i, '%Y%m%d').date() for i in df_report['est_dt']]
df_report = df_report.sort_values(by=['s_info_windcode', 'est_dt'])

# DATA MATCHING
for i in range(len(df_annc)):
    stock = df_annc.iloc[i, 0]
    date = df_annc.iloc[i, 1]
    future_date = [date + datetime.timedelta(days=k) for k in range(1, 6)]
    mask = (df_report.loc[:, 's_info_windcode'] == stock) & (df_report['est_dt'].adf_anncly(lambda x: x in future_date))
    df_report.loc[mask, 'original'] = date
df_merge = pd.merge(df_annc, df_report, how='inner', left_on=['ts_code.1', 'ann_date'],
                    right_on=['s_info_windcode', 'original'])
df_merge1 = df_merge.drop(['s_info_windcode', 'original'], axis=1)

# GAIN ALL TRADING DAYS
@lru_cache(1)
def get_all_trade_days(start_date='20110101'):
    df = pro.trade_cal(exchange='', start_date=pd.to_datetime(start_date).strftime('%Y%m%d'), is_open='1')
    trade_days = df['cal_date'].tolist()
    all_days = pd.to_datetime(trade_days)
    all_days = [date.date() for date in all_days]
    return all_days

@lru_cache(1)
def get_near_trade_day(date, forward=True):
    trade_days = get_all_trade_days()
    date = pd.to_datetime(date).date()
    if forward:
        date = (date - datetime.timedelta(days=1))
        while date not in trade_days:
            date = (date - datetime.timedelta(days=1))
    else:
        date = (date + datetime.timedelta(days=1))
        while date not in trade_days:
            date = (date + datetime.timedelta(days=1))
    return date

# MATCH RETURNS
df_merge1 = df_merge1.sort_values(by=['ann_date', 'ts_code.1'])
df_merge1.reset_index(drop=True, inplace=True)
ret = pd.DataFrame(columns=[df_merge1['ann_date'], df_merge1['ts_code.1']])
for i in range(len(df_merge1)):
    before = get_near_trade_day(df_merge1['ann_date'][i], forward=True)
    after = get_near_trade_day(df_merge1['ann_date'][i], forward=False)
    csi_ret = (df_csi.loc[after] - df_csi.loc[before]) / df_csi.loc[before]
    df_merge1.loc[i, 'ret'] = float(
        (df_preclose.loc[after, df_merge1['ts_code.1'][i]] - df_preclose.loc[before, df_merge1['ts_code.1'][i]]) /
        df_preclose.loc[before, df_merge1['ts_code.1'][i]] - csi_ret)
df_merge2 = df_merge1.copy()
df_merge3=df_merge2.dropna().sort_values(by=['ann_date','ts_code.1'])
df_merge3 = df_merge3[(df_merge3['ann_date'] >= '2011-01-01') & (df_merge3['ann_date'] <= '2021-12-31')]

##### WORD COUNT
import jieba.posseg as pseg
def filter_words(text):
    words=pseg.cut(text)
    pos_tags=['n', 'nt', 'v', 'vd', 'vn', 'an', 'ad']
    filtered_words=[word for word,pos in words if pos in pos_tags]
    return filtered_words

titles=df_merge3['report_name'].astype(str)
summaries=df_merge3['report_summary'].astype(str)
titles_filter=[filter_words(title) for title in tqdm(titles)]
summaries_filter=[filter_words(summary) for summary in tqdm(summaries)]
title_text=[' '.join(title) for title in titles_filter]
summary_text=[' '.join(summary) for summary in summaries_filter]
df_merge3['report_name']=title_text
df_merge3['report_summary']=summary_text

df_merge4 = df_merge3.copy()
df_merge4['ann_date'] = [datetime.datetime.strptime(i, '%Y-%m-%d').date() for i in df_merge4['ann_date']]

# SPLIT DATA INTO SAMPLE_IN AND SAMPLE_OUT (TRAIN / TEST)
def split_dataset(df, current_date):
    sample_in_start = datetime.date(current_date.year - 2, current_date.month, current_date.day)
    sample_in_end = current_date - datetime.timedelta(days=1)
    sample_out_start = current_date
    sample_out_end = datetime.date(current_date.year + 1, current_date.month, current_date.day)
    sample_in_data = df[(df['ann_date'] >= sample_in_start) & (df['ann_date'] <= sample_in_end)]
    sample_out_data = df[(df['ann_date'] >= sample_out_start) & (df['ann_date'] < sample_out_end)]
    return sample_in_data, sample_out_data

# ROLLING WINDOWS SPLIT DATA
def generate_rolling_datasets(df, start_date, end_date):
    rolling_datasets = []
    while start_date <= end_date:
        sample_in_data, sample_out_data = split_dataset(df, start_date)
        rolling_datasets.append((sample_in_data, sample_out_data))
        start_date = datetime.date(start_date.year + 1, start_date.month, start_date.day)
    return rolling_datasets
rolling_datasets = generate_rolling_datasets(df_merge4, datetime.date(2013, 1, 1), datetime.date(2021, 1, 1))

from sklearn.feature_extraction.text import CountVectorizer
def generate_features_labels(rolling_datasets):
    vectorizer_title = CountVectorizer(max_features=100)
    vectorizer_summary = CountVectorizer(max_features=500)
    X_all = []
    Y_all = []
    X_outs = []
    data_outs = []
    i = 0
    for sample_in_data, sample_out_data in tqdm(rolling_datasets):
        # SAMPLE_IN TEXTS PROCESSING
        X_title = vectorizer_title.fit_transform(sample_in_data['report_name'])
        X_summary = vectorizer_summary.fit_transform(sample_in_data['report_summary'])
        X = np.hstack((X_title.toarray(), X_summary.toarray()))
        X = np.log(X + 1)
        X_all.append(X)
        ret_percentiles = sample_in_data['ret'].quantile([0.3, 0.7])
        Y = pd.cut(sample_in_data['ret'],
                   bins=[-float('inf'), ret_percentiles[0.3], ret_percentiles[0.7], float('inf')], labels=[-1, 0, 1])
        Y_all.append(Y)

        # SAMPLE_OUT TEXTS PROCESSING
        X_title_out = vectorizer_title.transform(sample_out_data['report_name'])
        X_summary_out = vectorizer_summary.transform(sample_out_data['report_summary'])
        X_out = np.hstack((X_title_out.toarray(), X_summary_out.toarray()))
        X_out = np.log(X_out + 1)
        X_outs.append(X_out)
        data_out = sample_out_data[['ann_date', 'ts_code.1']]
        data_outs.append(data_out)
    return X_all, Y_all, X_outs, data_outs
X_all, Y_all, X_outs, data_outs = generate_features_labels(rolling_datasets)

# LOGISTIC REGRESSION
# from sklearn.linear_model import LogisticRegression
# XGBoost
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score
stock_list = list(df_preclose.columns)
trade_day = df_preclose.index[
    (df_preclose.index >= datetime.date(2013, 1, 1)) & (df_preclose.index <= datetime.date(2021, 12, 31))]
def train_model(X_all, Y_all, X_outs, data_outs):
    factor_list = []
    for X, Y, X_out, data_out in tqdm(zip(X_all, Y_all, X_outs, data_outs)):
        param_grid = {
            'learning_rate': [0.025, 0.05, 0.1],
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5]        
        }
        prob_list = []
        for i in range(-1, 2):
            # lr_model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=20)
            xgb_model = XGBClassifier(objective='binary:logistic', random_state=20
            # GRID SEARCH FOR LAMBDA
            grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='roc_auc')
            # OvR
            Y_class = (Y == i).astype(int)
            grid_search.fit(X, Y_class)
            best_params = grid_search.best_params_
            final_model = XGBClassifier (learning_rate=best_params['learning_rate'],
                                         n_estimators=best_params['n_estimators'],
                                         max_depth=best_params['max_depth'],
                                         gamma=0.8, 
                                         subsample=0.7,
                                         seed=1024)
            final_model.fit(X, Y_class)
            # SAMPLE_OUT PREDICTION
            prob = pd.DataFrame(final_model.predict_proba(X_out))[1]
            prob_list.append(prob)
        # FACTOR DEVELOPMENT
        df_prob = pd.concat(prob_list, axis=1)
        log_odds = np.log(df_prob / (1 - df_prob))
        data_out = data_out.reset_index()
        data_out['SUE'] = log_odds.iloc[:, 2] - log_odds.iloc[:, 0]
        df_factor = data_out.groupby(['ts_code.1', 'ann_date'])['SUE'].mean().reset_index()
        df_factor = df_factor.pivot(index='ann_date', columns='ts_code.1', values='SUE')
        df_factor_fill = df_factor.reindex(columns=stock_list)
        factor_list.append(df_factor_fill)
    return factor_list

factor_list = train_model(X_all, Y_all, X_outs, data_outs)
factor = pd.concat(factor_list, axis=0)
factor = factor.reindex(index=trade_day)
factor_fill = factor.dropna(axis=1, how='all')

df_bool = factor_fill.notna()
array = np.where(df_bool == True)
array_index = np.argsort(array[1])
array2 = (array[0][array_index], array[1][array_index])
length = factor_fill.shape[0]
def filled(times):
    factor_filled = factor_fill.copy()
    decay_factor = 0.95
    for i, j in tqdm(zip(*array2)):
        col_data = factor_filled.iloc[:, j]
        for date in range(i + 1, i + times):
            if date >= length:
                break
            elif pd.isna(col_data[date]) == False:
                break
            else:
                col_data[date] = col_data[date - 1] * decay_factor
    return factor_filled
factor_filled_150 = filled(times=150)[filled(times=150).index >= datetime.date(2014, 10, 24)]
factor_filled_240 = filled(times=240)[filled(times=240).index >= datetime.date(2014, 10, 24)]

# NUM_STOCKS AFTER FILLING
import matplotlib.pyplot as plt
def count_non_null_values(df, times):
    non_null_counts = []
    for index, row in df.iterrows():
        non_null_count = row.count()
        non_null_counts.append(non_null_count)
    year_end_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='Y')
    plt.figure(figsize=(10, 6))
    plt.bar(df.index, non_null_counts)
    plt.xlabel('Date')
    plt.ylabel('The Number of Stocks')
    plt.title('Maximum Fill Count = ' + str(times))
    plt.suptitle('The Number of Stocks for Each Day', fontsize=16, fontweight='bold')
    plt.xticks(year_end_dates, rotation=90)
    plt.show()
count_non_null_values(factor_filled_150, times=150)
count_non_null_values(factor_filled_240, times=240)

%run Single_Factor_Analysis.py
C = SingleFactorAnalysis('000985.SH', '20190118', '20221231', 'factor_filled_150', 'pct_close_next_close', long_low=0, dt_index='240m', n_group=10, neu_style=None, calc_crowd=False)
result_df = C.run_code(nextpct_shift=1)
result_df
