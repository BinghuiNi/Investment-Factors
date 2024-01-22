import pandas as pd
import datetime
from fastcache import lru_cache
import dcube as dc
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import jieba.posseg as pseg
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

token = ''
pro = dc.pro_api(token)

stock = pd.read_csv(r'C:\Users\nibh\Desktop\FZ\daily_factor\PCTCHANGE.csv', index_col=0)
df_preclose=pd.read_csv(r'C:\Users\nibh\Desktop\FZ\daily_factor\ADJPRECLOSE.csv',index_col=0)
df_csi=pd.read_csv(r'C:\Users\nibh\Desktop\FZ\daily_factor\CSIndex.csv',index_col=0)
df_csi=df_csi.drop('HS300',axis=1)
df_csi.index = [datetime.datetime.strptime(i, '%Y-%m-%d').date() for i in df_csi.index]
df_preclose.index = [datetime.datetime.strptime(i, '%Y-%m-%d').date() for i in df_preclose.index]
start_date = '20110101'
end_date = '20231231'
@lru_cache(1)
def get_all_trade_days(start_date=start_date):
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

# 1. DATA COLLECTION (Corporate Performance Pre-announcements)
ls_annc = [pro.forecast(ts_code=i, start_date=start_date, end_date=end_date, fields='ts_code,ann_date') for i in stock.columns]
df_annc = pd.concat(ls_annc, ignore_index=True)
df_annc.index = df_annc['ts_code']
df_annc['ann_date'] = [datetime.datetime.strptime(i, '%Y%m%d').date() for i in df_annc['ann_date']]
# 1. DATA COLLECTION (Sell-side Research Reports)
date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
ls_report = []
for i in range(len(date_range)):
    start = date_range[i].strftime('%Y%m%d')
    end = (date_range[i] + pd.DateOffset(months=1) - pd.DateOffset(days=1)).strftime('%Y%m%d')
    ls_report.append(pro.query('ashare_earning_est',start_date=start, end_date=end,
                               fields='s_info_windcode,est_dt,report_name,report_summary'))
df_report = pd.concat(ls_report, ignore_index=True)
df_report['est_dt'] = [datetime.datetime.strptime(i, '%Y%m%d').date() for i in df_report['est_dt']]
df_report = df_report.sort_values(by=['s_info_windcode', 'est_dt'])

# 2. DATA INTEGRATION (Announcements and Reports within 5 days)
for i in range(len(df_annc)):
    stock = df_annc.iloc[i, 0]
    date = df_annc.iloc[i, 1]
    future_date = [date + datetime.timedelta(days=k) for k in range(1, 6)]
    mask = (df_report['s_info_windcode'] == stock) & (df_report['est_dt'].adf_anncly(lambda x: x in future_date))
    df_report.loc[mask, 'original'] = date
df_merge = pd.merge(df_annc, df_report, how='inner', left_on=['ts_code.1', 'ann_date'],
                    right_on=['s_info_windcode', 'original'])
df_merge = df_merge.drop(['s_info_windcode', 'original'], axis=1).sort_values(by=['ann_date', 'ts_code.1'])
df_merge = df_merge.reset_index(drop=True, inplace=True)
# 2. DATA INTEGRATION ( With Excess Returns)
for i in range(df_merge.shape[0]):
    before = get_near_trade_day(df_merge[:,'ann_date'][i], forward=True)
    after = get_near_trade_day(df_merge[:,'ann_date'][i], forward=False)
    csi_ret = (df_csi.loc[after] - df_csi.loc[before]) / df_csi.loc[before]
    df_merge.loc[i,'ret'] = float(
        (df_preclose.loc[after, df_merge[:,'ts_code.1'][i]] - df_preclose.loc[before, df_merge['ts_code.1'][i]]) /
        df_preclose.loc[before, df_merge[:,'ts_code.1'][i]]) - csi_ret
df_merge = df_merge.dropna().sort_values(by=['ann_date','ts_code.1'])
df_merge = df_merge[(df_merge['ann_date'] >= '2011-01-01') & (df_merge['ann_date'] <= '2021-12-31')]
df_merge['ann_date'] = [datetime.datetime.strptime(i, '%Y-%m-%d').date() for i in df_merge['ann_date']]

# 3. TEXT CLEANING
def filter_words(text):
    words=pseg.cut(text)
    pos_tags=['n', 'nt', 'v', 'vd', 'vn', 'an', 'ad']
    filtered_words=[word for word,pos in words if pos in pos_tags]
    return filtered_words
ls_titles = df_merge['report_name'].astype(str)
ls_summaries= df_merge['report_summary'].astype(str)
ls_filtered_titles = [filter_words(title) for title in tqdm(ls_titles)]
ls_filtered_summaries = [filter_words(summary) for summary in tqdm(ls_summaries)]
ls_title_text=[' '.join(title) for title in ls_filtered_titles]
ls_summary_text=[' '.join(summary) for summary in ls_filtered_summaries]
df_merge['report_name'] = ls_title_text
df_merge['report_summary'] = ls_summary_text

# 4. DATA SPLIT (SAMPLE_IN: 2 years; SAMPLE_OUT: 1 year)
def split_dataset(df, current_date):
    sample_in_start = datetime.date(current_date.year - 2, current_date.month, current_date.day)
    sample_in_end = current_date - datetime.timedelta(days=1)
    sample_out_start = current_date
    sample_out_end = datetime.date(current_date.year + 1, current_date.month, current_date.day)
    sample_in_data = df[(df['ann_date'] >= sample_in_start) & (df['ann_date'] <= sample_in_end)]
    sample_out_data = df[(df['ann_date'] >= sample_out_start) & (df['ann_date'] < sample_out_end)]
    return sample_in_data, sample_out_data
# 4. DATA SPLIT (ROLLING)
def generate_rolling_datasets(df, start_date, end_date):
    rolling_datasets = []
    while start_date <= end_date:
        sample_in_data, sample_out_data = split_dataset(df, start_date)
        rolling_datasets.append((sample_in_data, sample_out_data))
        start_date = datetime.date(start_date.year + 1, start_date.month, start_date.day)
    return rolling_datasets
rolling_datasets = generate_rolling_datasets(df_merge,
                                             start_date=datetime.date(2013, 1, 1),
                                             end_date=datetime.date(2021, 1, 1))

# 5. FEATURE ENGINEERING (Word Frequency Matrix and Categorical Returns)
def generate_features_labels(dataset):
    vectorized_title = CountVectorizer(max_features=100)
    vectorized_summary = CountVectorizer(max_features=500)
    X_title = vectorized_title.fit_transform(dataset['report_name'])
    X_summary = vectorized_summary.fit_transform(dataset['report_summary'])
    X = np.hstack((X_title.toarray(), X_summary.toarray()))
    X = np.log(X + 1)
    ret_percentiles = dataset['ret'].quantile([0.3, 0.7])
    Y = pd.cut(dataset['ret'],
               bins=[-float('inf'), ret_percentiles[0.3], ret_percentiles[0.7], float('inf')], labels=[-1, 0, 1])
    return X, Y
def generate_features_labels_rolling(rolling_datasets):
    ls_in_Xs, ls_in_Ys, ls_out_Xs, ls_outs = [], [], [], []
    for sample_in_data, sample_out_data in tqdm(rolling_datasets):
        sample_in_X, sample_in_Y = generate_features_labels(sample_in_data)
        sample_out_X, sample_out_Y = generate_features_labels(sample_out_data)
        ls_in_Xs.append(sample_in_X)
        ls_in_Ys.append(sample_in_Y)
        ls_out_Xs.append(sample_out_X)
        df_out = sample_out_data[['ann_date', 'ts_code.1']]
        ls_outs.append(df_out)
    return ls_in_Xs, ls_in_Ys, ls_out_Xs, ls_outs
ls_in_Xs, ls_in_Ys, ls_out_Xs, ls_outs = generate_features_labels_rolling(rolling_datasets)

# 6. XGBoost -- XGBClassifier Prediction
ls_stocks = list(df_preclose.columns)
ls_trade_days = df_preclose.index[
    (df_preclose.index >= datetime.date(2013, 1, 1)) & (df_preclose.index <= datetime.date(2021, 12, 31))]
def train_model(ls_in_Xs, ls_in_Ys, ls_out_Xs, ls_outs):
    ls_factors, ls_prob = [], []
    for in_X, in_Y, out_X, out_data in tqdm(zip(ls_in_Xs, ls_in_Ys, ls_out_Xs, ls_outs)):
        param_grid = {
            'learning_rate': [0.025, 0.05, 0.1],
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5]
        }
        for i in range(-1, 2): # OvR
            # lr_model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=20)
            xgb_model = XGBClassifier(objective='binary:logistic', random_state=20)
            # GRID SEARCH FOR TUNING HYPERPARAMETERS
            grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='roc_auc')
            # OvR
            Y_class = (in_Y==i).astype(int)
            grid_search.fit(in_X, Y_class)
            best_params = grid_search.best_params_
            final_model = XGBClassifier (learning_rate=best_params['learning_rate'],
                                         n_estimators=best_params['n_estimators'],
                                         max_depth=best_params['max_depth'],
                                         gamma=0.8,
                                         subsample=0.7,
                                         seed=1024)
            final_model.fit(in_X, Y_class)
            df_prob = pd.DataFrame(final_model.predict_proba(out_X))[1]
            ls_prob.append(df_prob)
        # FACTOR DEVELOPMENT
        df_probs = pd.concat(ls_prob, axis=1)
        log_odds = np.log(df_probs / (1 - df_probs))
        out_data = out_data.reset_index()
        out_data['SUE'] = log_odds.iloc[:, 2] - log_odds.iloc[:, 0]
        df_factor = out_data.groupby(['ts_code.1', 'ann_date'])['SUE'].mean().reset_index()
        df_factor = df_factor.pivot(index='ann_date', columns='ts_code.1', values='SUE')
        df_factor_fill = df_factor.reindex(columns=ls_stocks)
        ls_factors.append(df_factor_fill)
    return ls_factors

ls_factors = train_model(ls_in_Xs, ls_in_Ys, ls_out_Xs, ls_outs)
df_Factor = pd.concat(ls_factors, axis=0).reindex(index=ls_trade_days).dropna(axis=1, how='all')

# 7. MISSING VALUES IMPUTATION
df_bool = df_Factor.notna()
array = np.where(df_bool == True)
array_index = np.argsort(array[1])
array2 = (array[0][array_index], array[1][array_index])
length = df_Factor.shape[0]
def factor_decay_imputation(df_Factor, times):
    decay_rate = 0.95
    for i, j in tqdm(zip(*array2)):
        col_data = df_Factor.iloc[:, j]
        for date in range(i + 1, i + times):
            if date >= length:
                break
            elif pd.isna(col_data[date]) == False:
                break
            else:
                col_data[date] = col_data[date - 1] * decay_rate
    return df_Factor
factor_imput_150 = factor_decay_imputation(df_Factor, times=150)
factor_imput_240 = factor_decay_imputation(df_Factor, times=240)

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
count_non_null_values(factor_imput_150, times=150)
count_non_null_values(factor_imput_240, times=240)

%run Single_Factor_Analysis.py
C = SingleFactorAnalysis('000985.SH', start_date, end_date, 'factor_imput_240', 'pct_close_next_close', long_low=0, dt_index='240m', n_group=5, neu_style=None, calc_crowd=False)
result_df = C.run_code(nextpct_shift=1)
result_df
