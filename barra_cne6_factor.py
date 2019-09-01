# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 08:26:53 2019

@author: admin
"""

import os
import calendar
import warnings
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pandas.tseries.offsets as toffsets
from itertools import dropwhile, chain, product
from functools import reduce, wraps
from dask import dataframe as dd
from dask.multiprocessing import get
#from pyfinance.ols import PandasRollingOLS as rolling_ols
from pyfinance.utils import rolling_windows
from barra_template import Data
warnings.filterwarnings('ignore')

START_YEAR = 2009
END_YEAR = 2019
BENCHMARK = '000300.SH'
VERSION = 6

dat = Data()
work_dir = os.path.dirname(__file__)
SENTINEL = 1e10

__spec__ = None

def time_decorator(func):
    @wraps(func)
    def timer(*args, **kwargs):
        start = datetime.datetime.now()
        result = func(*args, **kwargs)
        end = datetime.datetime.now()
        print(f'“{func.__name__}” run time: {end - start}.')
        return result
    return timer

class lazyproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value

class parallelcal:
    @staticmethod
    def _regress(y, X, intercept=True, weight=1, verbose=True):
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.DataFrame(y)
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            X = pd.DataFrame(X)
        
        if intercept:
            cols = X.columns.tolist()
            X['const'] = 1
            X = X[['const'] + cols] 
        
        model = sm.WLS(y, X, weights=weight)
        result = model.fit()
        params = result.params

        if verbose:
            resid = y - pd.DataFrame(np.dot(X, params), index=y.index, 
                         columns=y.columns)
            if intercept:
                return params.iloc[1:], params.iloc[0], resid
            else:
                return params, None, resid
        else:
            if intercept:
                return params.iloc[1:]
            else:
                return params
    
    @staticmethod
    def weighted_std(series, weights):
        return np.sqrt(np.sum((series-np.mean(series)) ** 2 * weights))
    
    def weighted_func(self, func, series, weights):
        weights /= np.sum(weights)
        if func.__name__ == 'std':
            return self.weighted_std(series, weights)
        else:
            return func(series * weights)
    
    def nanfunc(self, series, func, sentinel=SENTINEL, weights=None):
        valid_idx = np.argwhere(series != sentinel)
        if weights is not None:
            return self.weighted_func(func, series[valid_idx], 
                                      weights=weights[valid_idx])
        else:
            return func(series[valid_idx])
    
    @staticmethod
    def _cal_cmra(series, months=12, days_per_month=21, version=6):
        z = sorted(series[-i * days_per_month:].sum() for i in range(1, months+1))
        if version == 6:
            return z[-1] - z[0]
        elif version == 5:
            return np.log(1 + z[-1]) - np.log(1 + z[0])
    
    def _cal_midcap(self, series):
        x = series.dropna().values
        y = x ** 3
        beta, alpha, _ = self._regress(y, x, intercept=True, weight=1, verbose=True)
        resid = series ** 3 - (alpha + beta[0] * series)
        return resid
    
    @staticmethod
    def _cal_liquidity(series, days_pm=21, sentinel=-SENTINEL):
        freq = len(series) // days_pm
        valid_idx = np.argwhere(series != SENTINEL)
        series = series[valid_idx]
        res = np.log(np.nansum(series) / freq)
        if np.isinf(res):
            return sentinel
        else:
            return res
    
    def _cal_growth_rate(self, series): 
        valid_idx = np.argwhere(pd.notna(series))
        y = series[valid_idx]
        x = np.arange(1, len(series)+1)[valid_idx]
        
        coef = self._regress(y, x, intercept=True, verbose=False)
        return coef.iloc[0] / y.mean()
    
    def _get_apply_rptdate(self, df, idate=None, delist_map=None):        
        code = df.name
        delist_date = delist_map[code]
        rptrealdates = idate.loc[code,:].tolist()
        
        if pd.isnull(delist_date): 
            res = [self.__append_date(rptrealdates, curdate, idate) for curdate in df.index]
        else:
            res = []
            for curdate in df.index:
                if curdate >= delist_date:
                    res.append(pd.NaT)
                else:
                    res.append(self.__append_date(rptrealdates, curdate, idate))
        return res
    
    @staticmethod
    def __append_date(rptrealdates, curdate, idate, base_time='1899-12-30 00:00:00'):
        base_time = pd.to_datetime(base_time)
        rptavaildates = sorted(d for d in rptrealdates if d < curdate and d != base_time)
        if rptavaildates:
            availdate1 = rptavaildates[-1]
            didx = rptrealdates.index(availdate1) 
            try:
                availdate2 = rptavaildates[-2]
            except IndexError:
                pass
            else:
               if availdate1 == availdate2:
                   didx += 1
            finally:
               return idate.columns[didx]
        else:
            return pd.NaT 
    
def get_fill_vals(nanidx, valid_vals):
    start, end = nanidx[0], nanidx[-1]
    before_val, after_val = valid_vals[start-1], valid_vals[end+1]
    diff = (after_val - before_val) / (1 + len(nanidx))
    fill_vals = [before_val + k * diff for k in range(1, len(nanidx) + 1)]
    return fill_vals

def linear_interpolate(series):
    vals = series.values
    valid_vals = list(dropwhile(lambda x: np.isnan(x), vals))
    idx = np.where(np.isnan(valid_vals))[0]
    start_idx = len(vals) - len(valid_vals)
    
    tmp = []
    for i, cur_num in enumerate(idx):
        try:
            next_num = idx[i+1]
        except IndexError:
            if cur_num < len(vals) - 1:
                try:
                    if tmp:
                        tmp.append(cur_num)
                        fill_vals = get_fill_vals(tmp, valid_vals)
                        for j in range(len(tmp)):
                            vals[start_idx + tmp[j]] = fill_vals[j]
                    else:
                        fill_val = 0.5 * (valid_vals[cur_num - 1] + valid_vals[cur_num + 1])
                        vals[start_idx + cur_num] = fill_val
                except IndexError:
                    break
                break
        else:
            if next_num - cur_num == 1:
                tmp.append(cur_num)
            else:
                if tmp:
                    tmp.append(cur_num)
                    fill_vals = get_fill_vals(tmp, valid_vals)
                    for j in range(len(tmp)):
                        vals[start_idx + tmp[j]] = fill_vals[j]
                    tmp = []
                else:
                    try:
                        fill_val = 0.5 * (valid_vals[cur_num - 1] + valid_vals[cur_num + 1])
                        vals[start_idx + cur_num] = fill_val
                    except IndexError:
                        break
    res = pd.Series(vals, index=series.index)
    return res

class CALFUNC(Data):    
    def __init__(self):
        super().__init__()
        self._parallel_funcs = parallelcal()
        
    @lazyproperty
    def tdays(self):
        return sorted(self.adjfactor.columns)

    def __getattr__(self, item):
        try:
            return getattr(self._parallel_funcs, item)
        except AttributeError:
            return super().__getattr__(item)
    
    def _cal_pctchange_in_month(self, series):
        date = series.name
        stocks = series.index
        if date.month > 1:
            lstyear = date.year
            lstmonth = date.month - 1
        else:
            lstyear = date.year - 1
            lstmonth = date.month - 1 + 12
        lstday = min(date.day, calendar.monthrange(lstyear, lstmonth)[1])
        lstdate = toffsets.datetime(lstyear, lstmonth, lstday)
        lstdateidx = self._get_date_idx(lstdate, self.tdays)
        lstdate = self.tdays[lstdateidx]
        try:
            res = self.hfq_close.loc[stocks, date] / self.hfq_close.loc[stocks, lstdate] - 1
        except KeyError:
            res = series.where(pd.isnull(series), np.nan)
        return res 
    
    def _cal_pctchange_nextmonth(self, series):
        date = series.name
        stocks = series.index
        td_idx = self._get_date_idx(date, self.tdays)
        nstart_idx, nend_idx = td_idx + 1, td_idx + 21
        try:
            nend_date = self.tdays[nend_idx]
        except IndexError:
            return np.array([np.nan] * len(series))
        else:
            nstart_date = self.tdays[nstart_idx]
            res = self.hfq_close.loc[stocks, nend_date] / self.hfq_close.loc[stocks, nstart_date] - 1
            return res
    
    def _get_price_last_month_end(self, type_='close'):
        price = getattr(self, type_,)
        if price is None:
            raise Exception(f'Unsupported price type {type_}!')
        date_range = price.columns.tolist()
        price_me = price.T.groupby(pd.Grouper(freq='m')).apply(lambda df:df.iloc[-1])
        dates_me = [d2 for d1, d2 in zip(date_range[1:], date_range[:-1]) if d1.month != d2.month]
        if len(dates_me) < price_me.shape[0]:
            price_me.index = dates_me + date_range[-1:]
        else:
            price_me.index = dates_me
        price_lme = price_me.reindex(date_range).fillna(method='ffill').shift(1)
        return price_lme
    
    def _get_pct_chg_m_daily(self):
        tdays = dropwhile(lambda date: date.year != START_YEAR - 6, self.tdays)
        res = pd.DataFrame(index=self.hfq_close.index, 
                           columns=list(tdays))
#        return self._pandas_parallelcal(res, self._cal_pctchange_in_month, 
#               args=(self._get_date_idx, self.tdays, self.hfq_close), axis=0).T
        return res.apply(self._cal_pctchange_in_month).T
    
    def _get_pct_chg_nm(self):
        tdays = dropwhile(lambda date: date.year != START_YEAR - 5, self.tdays)
        res = pd.DataFrame(index=self.hfq_close.index, 
                           columns=list(tdays))
        return res.apply(self._cal_pctchange_nextmonth).T
    
    @staticmethod
    def clear_vals(df):
        df.iloc[1:] = np.nan
        return df
            
    @staticmethod
    def fill_vals(df):
        return df.fillna(method='ffill')
    
    def clean_data(self, datdf, index=False, limit_days=False):
        if datdf.index.dtype != 'O':
            datdf = datdf.T
        data_cleaned = self.reindex(datdf, to='wind', if_index=index)
        if not index:
            valid_stks = [i for i in data_cleaned.index if i[0].isnumeric()]
            data_cleaned = data_cleaned.loc[valid_stks, :]
        if limit_days:
            tdays = self.get_trade_days(START_YEAR, END_YEAR)
            data_cleaned = data_cleaned.loc[:, tdays]
        return data_cleaned.T
    
    def _get_intact_rpt_dates(self, start_year=START_YEAR, end_year=END_YEAR):
        intact_rpt_dates = sorted(map(lambda x: pd.to_datetime(f'{x[0]}-{x[1]}'), 
            product(range(start_year, end_year+1), ('03-31', '06-30', '09-30', '12-31'))))
        cur_year = toffsets.datetime.now().year
        cur_month = toffsets.datetime.now().month
        if end_year == cur_year:
            if cur_month < 4:
                return intact_rpt_dates[:-4]
            elif 4 <= cur_month < 8:
                return intact_rpt_dates[:-3]
            elif 8 <= cur_month < 10:
                return intact_rpt_dates[:-2]
            else:
                return intact_rpt_dates[:-1]
        return intact_rpt_dates
        
    def _get_ttm_data(self, datdf):
        datdf = self.clean_data(datdf)
        rpt_dates = sorted(d for d in datdf.index if (d.month, d.day) in ((3, 31), (6, 30), (9, 30), (12, 31)))
        datdf = datdf.loc[rpt_dates,:]
        
        start_year, end_year = rpt_dates[0].year, rpt_dates[-1].year
        intact_rpt_dates = self._get_intact_rpt_dates(start_year, end_year)
        datdf = datdf.reindex(intact_rpt_dates)
        virtual_rpt_dates = np.argwhere(pd.isnull(datdf).sum(axis=1) == datdf.shape[1])
        datdf.iloc[virtual_rpt_dates.flatten()] = 0
        
        res = pd.DataFrame(columns=datdf.index, index=datdf.columns)
        for date in datdf.index[4:]:
            if date.month == 12:
                res[date] = datdf.loc[date]
                continue
            lst_rpt_y = pd.to_datetime(f'{date.year - 1}-12-31')
            lst_rpt_q = pd.to_datetime(f'{date.year - 1}-{date.month}-{date.day}')
            res[date] = datdf.loc[lst_rpt_y] + datdf.loc[date] - datdf.loc[lst_rpt_q]
        return res.T
        
    def _transfer_freq(self, datdf, method='lyr', from_='q', to_='d', start_year=2006):
        dat_cleaned = self.clean_data(datdf)
        if to_ == 'y':
            dat_cleaned = dat_cleaned.fillna(method='ffill')
            dat_cleaned = dat_cleaned.groupby(pd.Grouper(freq='y')).apply(lambda df: df.iloc[-1])
            return dat_cleaned
        
        if from_ == 'q':
            if method == 'mrq':
                curstart_year = start_year - 1
                dat_cleaned = dat_cleaned.loc[f'{curstart_year}':,]
                
                q_map = {'Q1':3, 'Q2':6, 'Q3':9}
                month_group = self.month_group.loc[:,['Q1','Q2','Q3']]
                dat_cleaned.reindex(month_group.index)
                dat_cleaned = pd.concat([dat_cleaned, month_group], axis=1)    
                
                res = pd.DataFrame()
                for gp in ['Q1', 'Q2', 'Q3']:
                    tmp = dat_cleaned.groupby(gp).apply(self.fill_vals).dropna(how='all', axis=0)
                    tmp = tmp.groupby(gp).apply(self.clear_vals).apply(self.fill_vals)
                    tmp = tmp.iloc[:, :-3]
                    
                    drop_rows_idx = [d for d in tmp.index if d.month == q_map[gp]]
                    tmp = tmp.drop(drop_rows_idx, axis=0) 
                    
                    res = pd.concat([res, tmp], axis=0)
                    
            elif method == 'lyr':
                curstart_year = start_year - 2
                dat_cleaned = dat_cleaned.loc[f'{curstart_year}':,]
                dat_cleaned = dat_cleaned.fillna(method='ffill')
                
                annual_rpt_date = [d for d in dat_cleaned.index if d.month == 12 and d.day == 31]
                annual_rpt_data = dat_cleaned.loc[annual_rpt_date,:]
                syear, eyear = annual_rpt_date[0].year, annual_rpt_date[-1].year+1
                month_group = self.month_group.loc[str(syear):str(eyear), ['Q4-1', 'Q4-2']]
                year_group = month_group['Q4-2']
                annual_rpt_data = pd.concat([annual_rpt_data, year_group], axis=1)
                
                dat_grouped = pd.concat([dat_cleaned.reindex(month_group.index), month_group['Q4-1']], axis=1)
                
                res = pd.DataFrame()
                for gp, df in dat_grouped.groupby('Q4-1'):
                    data_to_broadcast = annual_rpt_data.loc[annual_rpt_data['Q4-2']==gp].iloc[:,:-1]
                    tmp = pd.DataFrame(index=df.index, columns=df.columns[:-1])
                    if len(data_to_broadcast) == 0:   
                        tmp.loc[:,:] = np.nan
                    else:
                        tmp.loc[:,:] = np.repeat(data_to_broadcast.values, len(tmp), 0)
                    res = pd.concat([res, tmp])                
            res = res.sort_index()
            res = res.fillna(method='ffill')
        
        if to_ == 'd':
            res.index = self.month_map.loc[res.index].values.flatten()
            start_year, end_year = res.index[0].year, res.index[-1].year
            start_year = max(2004, start_year) 
            tdays = self.get_trade_days(start_year, end_year)
            res = res.reindex(tdays).fillna(method='ffill')            
        return res        

    def get_trade_days(self, start_year=START_YEAR, end_year=END_YEAR, tdays=None):
        if tdays is None:
            tdays = self.tdays    
        start_idx = self._get_date_idx(f'{start_year}-01-01', tdays)
        if end_year < toffsets.date.today().year:
            end_idx = self._get_date_idx(f'{end_year}-12-31', tdays)
        else:
            end_idx = -1
        return tdays[start_idx+1:end_idx]
    
    def _shift(self, datdf, shift=1):
        datdf = self.clean_data(datdf)
        datdf = datdf.shift(shift)
        return datdf
    
    def _align(self, df1, df2, *dfs):
        dfs_all = [self.clean_data(df) for df in chain([df1, df2], dfs)]
        if any(len(df.shape) == 1 or 1 in df.shape for df in dfs_all):
            dims = 1
        else:
            dims = 2
        mut_date_range = sorted(reduce(lambda x,y: x.intersection(y), (df.index for df in dfs_all)))
        mut_codes = sorted(reduce(lambda x,y: x.intersection(y), (df.columns for df in dfs_all)))
        if dims == 2:
            dfs_all = [df.loc[mut_date_range, mut_codes] for df in dfs_all]
        elif dims == 1:
            dfs_all = [df.loc[mut_date_range, :] for df in dfs_all]
        return dfs_all
    
    @staticmethod
    def __drop_invalid_and_fill_val(series, val=None, method=None):
        valid_idx = np.argwhere(series.notna()).flatten()
        try:
            series_valid = series.iloc[valid_idx[0]:]
        except IndexError:
            return series
        if val:
            series_valid = series_valid.fillna(val)
        elif method:
            series_valid = series_valid.fillna(method=method)
        else:
            median = np.nanmedian(series_valid)
            series_valid = series_valid.fillna(median)
        series = series.iloc[:valid_idx[0]].append(series_valid)
        return series
    
    def _fillna(self, datdf, value=None, method=None):
        datdf = self.clean_data(datdf)        
        datdf = datdf.apply(self.__drop_invalid_and_fill_val, 
                            args=(value, method))
        return datdf
    
    @staticmethod
    def _get_exp_weight(window, half_life):
        exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
        return exp_wt[::-1] / np.sum(exp_wt)
    
    @staticmethod
    @time_decorator
    def _pandas_parallelcal(dat, myfunc, ncores=6, args=None, axis=1, window=None):
        if axis == 0 and window is None:
            dat = dat.T
        dat = dd.from_pandas(dat, npartitions=ncores)
        if window:
            dat = dat.rolling(window=window)
            if args is None:
                res = dat.apply(myfunc)
            else:
                res = dat.apply(myfunc, args=args)
        else:
            res = dat.apply(myfunc, args=args, axis=1)
        return res.compute(get=get)
    
    @time_decorator
    def _get_growth_rate(self, ori_data, periods=5, freq='y'):
#        self = CALFUNC(); s = parallelcal(); freq='y'; periods=5;ori_data = self.totalassets
        ori_data = self.clean_data(ori_data)
        current_lyr_rptdates = self.applied_lyr_date_d
        if ori_data.index.dtype == 'O':
            ori_data = ori_data.T
        ori_data = ori_data.groupby(pd.Grouper(freq=freq)).apply(lambda df: df.iloc[-1])
        ori_data = self._pandas_parallelcal(ori_data, self._cal_growth_rate, window=5)
        
        current_lyr_rptdates = current_lyr_rptdates.loc[ori_data.columns, :] 
        current_lyr_rptdates = current_lyr_rptdates.stack().reset_index()
        current_lyr_rptdates.columns = ['code', 'date', 'rptdate']
        current_lyr_rptdates['rptdate'] = pd.to_datetime(current_lyr_rptdates['rptdate'])
        current_lyr_rptdates = current_lyr_rptdates.set_index(['code', 'rptdate'])
        
        ori_data = ori_data.T.stack()
        res = ori_data.loc[current_lyr_rptdates.index]
        res = pd.concat([current_lyr_rptdates, res], axis=1)
        res = res.reset_index()
        res.columns = ['code', 'rptdate', 'date', 'value']
        res = pd.pivot_table(res, values='value', index=['code'], columns=['date'])
        return res
    
    def _get_codes_listed(self, stocks, date):
        stk_basic_info = self.all_stocks_code
        stk_basic_info = stk_basic_info[stk_basic_info['wind_code'].isin(stocks)]
        listed_cond = stk_basic_info['ipo_date'] <= date
        return stk_basic_info[listed_cond].wind_code.tolist()
    
    def _get_codes_not_delisted(self, stocks, date):
        stk_basic_info = self.all_stocks_code
        stk_basic_info = stk_basic_info[stk_basic_info['wind_code'].isin(stocks)]
        not_delisted_cond = (stk_basic_info['delist_date'] >= date) | \
                            (pd.isnull(stk_basic_info['delist_date']))
        return stk_basic_info[not_delisted_cond].wind_code.tolist()
    
    def _get_benchmark_ret(self, code=BENCHMARK):        
        pct_chg_idx = self.clean_data(self.indexquote_changepct / 100, 
                                      index=True)
        idx_ret = self._get_ret(pct_chg_idx, [code])
        return idx_ret
    
    def _get_ret(self, pct_chg=None, codes=None):
        if pct_chg is None:
            pct_chg = self.clean_data(self.changepct / 100)
        if codes is None:
            codes = pct_chg.columns
        ret = pct_chg.loc[:, codes]
        return ret
    
    def _rolling(self, datdf, window, half_life=None, 
                 func_name='sum', weights=None):
        global SENTINEL
        datdf = self.clean_data(datdf)
        datdf = datdf.where(pd.notnull(datdf), SENTINEL)
        if datdf.index.dtype == 'O':
            datdf = datdf.T
        
        func = getattr(np, func_name, )
        if func is None:
            msg = f"""Search func:{func_name} from numpy failed, 
                   only numpy ufunc is supported currently, please retry."""
            raise AttributeError(msg)
        
        if half_life or (weights is not None):
            exp_wt = self._get_exp_weight(window, half_life) if half_life else weights
            args = func, SENTINEL, exp_wt
        else:
            args = func, SENTINEL
        
        try:
            res = self._pandas_parallelcal(datdf, self.nanfunc, args=args, 
                                           axis=0, window=window)
        except Exception:
            print('Calculating under single core mode...')
            res = self._rolling_apply(datdf, self.nanfunc, args=args, 
                                      axis=0, window=window)
        return res.T
    
    def _rolling_apply(self, datdf, func, args=None, axis=0, window=None):
        if window:
            res = datdf.rolling(window=window).apply(func, args=args)
        else:
            res = datdf.apply(func, args=args, axis=axis)
        return res
    
    def _rolling_regress(self, y, x, window=5, half_life=None, 
                         intercept=True, verbose=False, fill_na=0):
        fill_args = {'method': fill_na} if isinstance(fill_na, str) else {'value': fill_na}
        
        x, y = self._align(x, y)
        stocks = y.columns
        if half_life:
            weight = self._get_exp_weight(window, half_life)
        else:
            weight = 1
        
        start_idx = x.loc[pd.notnull(x).values.flatten()].index[0]
        x, y = x.loc[start_idx:], y.loc[start_idx:,:]
        rolling_ys = rolling_windows(y, window)
        rolling_xs = rolling_windows(x, window)

        beta = pd.DataFrame()
        alpha = pd.DataFrame()
        sigma = pd.DataFrame()
        for i, (rolling_x, rolling_y) in enumerate(zip(rolling_xs, rolling_ys)):
            rolling_y = pd.DataFrame(rolling_y, columns=y.columns,
                                     index=y.index[i:i+window])
            window_sdate, window_edate = rolling_y.index[0], rolling_y.index[-1]
            stks_to_regress = sorted(set(self._get_codes_listed(stocks, window_sdate)) & \
                              set(self._get_codes_not_delisted(stocks, window_edate)))
            rolling_y = rolling_y[stks_to_regress].fillna(**fill_args)
            try:
                b, a, resid = self._regress(rolling_y.values, rolling_x,
                                        intercept=True, weight=weight, verbose=True)
            except:
                print(i)
                raise
            vol = np.std(resid, axis=0)
            vol.index = a.index = b.columns = stks_to_regress
            b.index = [window_edate]
            vol.name = a.name = window_edate
            beta = pd.concat([beta, b], axis=0)
            alpha = pd.concat([alpha, a], axis=1)
            sigma = pd.concat([sigma, vol], axis=1)
        
        beta = beta.T
        beta = beta.reindex(y.index, axis=1).reindex(y.columns, axis=0)
        alpha = alpha.reindex(y.index, axis=1).reindex(y.columns, axis=0)
        sigma = sigma.reindex(y.index, axis=1).reindex(y.columns, axis=0)
        return beta, alpha, sigma

    def _capm_regress(self, window=504, half_life=252):
        y = self._get_ret(self.changepct / 100)
        x = self._get_benchmark_ret()
        beta, alpha, sigma = self._rolling_regress(y, x, window=window, 
                                                   half_life=half_life)
        return beta, alpha, sigma
    
    def _get_period_d(self, date, offset=None, freq=None, datelist=None):
        if isinstance(offset, (float, int)) and offset > 0:
            raise Exception("Must return a period before current date.")
        
        conds = {}
        freq = freq.upper()
        if freq == "M":
            conds.update(months=-offset)
        elif freq == "Q":
            conds.update(months=-3*offset)
        elif freq == "Y":
            conds.update(years=-offset)
        else:
            freq = freq.lower()
            conds.update(freq=-offset)
        
        sdate = pd.to_datetime(date) - pd.DateOffset(**conds) 
        
        if datelist is None:
            datelist = self.tdays
        sindex = self._get_date_idx(sdate, datelist, ensurein=True)
        eindex = self._get_date_idx(date, datelist, ensurein=True)
        return datelist[sindex:eindex+1]
    
    def _get_date_idx(self, date, datelist=None, ensurein=False):
        msg = """Date {} not in current tradedays list. If tradedays list has already been setted, \
              please reset tradedays list with longer periods or higher frequency."""
        date = pd.to_datetime(date)
        if datelist is None:
            datelist = self.tdays
        try:
            datelist = sorted(datelist)
            idx = datelist.index(date)
        except ValueError:
            if ensurein:
                raise IndexError(msg.format(str(date)[:10]))
            dlist = list(datelist)
            dlist.append(date)
            dlist.sort()
            idx = dlist.index(date) 
            if idx == len(dlist)-1 or idx == 0:
                raise IndexError(msg.format(str(date)[:10]))
            return idx - 1
        return idx

#1******Size
class Size(CALFUNC):
    @lazyproperty
    def LNCAP(self):  
        lncap = np.log(self.negotiablemv)
        return lncap
    #5 --
    @lazyproperty
    def MIDCAP(self):
        lncap = self.LNCAP
        midcap = self._pandas_parallelcal(lncap, self._cal_midcap, axis=0).T
        return midcap
    #5 -- 
#2******Volatility
class Volatility(CALFUNC):    
    @lazyproperty
    def BETA(self, version=VERSION):
        if 'BETA' in self.__dict__:
            return self.__dict__['BETA']
        if version == 6:
            beta, alpha, hsigma = self._capm_regress(window=504, half_life=252)
            self.__dict__['HSIGMA'] = hsigma
            self.__dict__['HALPHA'] = alpha
        elif version == 5:
            beta, alpha, hsigma = self._capm_regress(window=252, half_life=63)
            self.__dict__['HSIGMA'] = hsigma
        return beta
    #5 ** window = 252, hl = 63
    @lazyproperty
    def HSIGMA(self, version=VERSION):
        if 'HSIGMA' in self.__dict__:
            return self.__dict__['HSIGMA']
        if version == 6:
            beta, alpha, hsigma = self._capm_regress(window=504, half_life=252)
            self.__dict__['BETA'] = hsigma
            self.__dict__['HALPHA'] = alpha
        elif version == 5:
            beta, alpha, hsigma = self._capm_regress(window=252, half_life=63)
            self.__dict__['BETA'] = beta
        return hsigma
    #5 ** window = 252, hl = 63
    @lazyproperty
    def HALPHA(self):
        if 'HALPHA' in self.__dict__:
            return self.__dict__['HALPHA']
        beta, alpha, hsigma = self._capm_regress(window=504, half_life=252)
        self.__dict__['BETA'] = beta
        self.__dict__['HSIGMA'] = hsigma
        return alpha
    
    @lazyproperty
    def DASTD(self):
        dastd = self._rolling(self.changepct / 100, window=252, 
                              half_life=42, func_name='std')
        return dastd
    #5 --
    @lazyproperty
    def CMRA(self, version=VERSION):
        stock_ret = self._fillna(self.changepct / 100, 0)
        if version == 6:
            ret = np.log(1 + stock_ret)
        elif version == 5:
            index_ret = self._get_benchmark_ret()
            index_ret = np.log(1 + index_ret)
            index_ret, stock_ret = self._align(index_ret, stock_ret)
            ret = np.log(1 + stock_ret).sub(index_ret[BENCHMARK], axis=0)
        cmra = self._pandas_parallelcal(ret, self._cal_cmra, args=(12, 21, version), 
                                        window=252, axis=0).T
        return cmra
    #5 ** cmra = ln(1+zmax) - ln(1+zmin), z = sigma[ln(1+rt) - ln(1+r_hs300)]
#3******Liquidity
class Liquidity(CALFUNC):
    @lazyproperty
    def STOM(self):
        amt, mkt_cap_float = self._align(self.turnovervalue, self.negotiablemv)
        share_turnover = amt * 10000 / mkt_cap_float
        share_turnover = share_turnover.where(pd.notnull(share_turnover), SENTINEL)
        stom = self._pandas_parallelcal(share_turnover, self._cal_liquidity, 
                                        axis=0, window=21).T
        return stom
    #5 --     
    @lazyproperty
    def STOQ(self):
        amt, mkt_cap_float = self._align(self.turnovervalue, self.negotiablemv)
        share_turnover = amt * 10000 / mkt_cap_float
        share_turnover = share_turnover.where(pd.notnull(share_turnover), SENTINEL)
        stoq = self._pandas_parallelcal(share_turnover, self._cal_liquidity, 
                                        axis=0, window=63).T
        return stoq
    #5 --
    @lazyproperty
    def STOA(self):
        amt, mkt_cap_float = self._align(self.turnovervalue, self.negotiablemv)
        share_turnover = amt * 10000 / mkt_cap_float
        share_turnover = share_turnover.where(pd.notnull(share_turnover), SENTINEL)
        stoa = self._pandas_parallelcal(share_turnover, self._cal_liquidity, 
                                        axis=0, window=252).T
        return stoa
    #5 --
    @lazyproperty
    def ATVR(self):
        turnoverrate = self.turnoverrate / 100
        atvr = self._rolling(turnoverrate, window=252, half_life=63, func_name='sum')
        return atvr
    
#4******Momentum
class Momentum(CALFUNC):
    @lazyproperty
    def STREV(self):
        strev = self._rolling(self.changepct / 100, window=21, 
                              half_life=5, func_name='sum')
        return strev
        
    @lazyproperty
    def SEASON(self):
        nyears = 5
        pct_chg_m_d = self._get_pct_chg_m_daily()
        pct_chgs_shift = [pct_chg_m_d.shift(i*21*12 - 21) for i in range(1,nyears+1)]
        seasonality = sum(pct_chgs_shift) / nyears
        seasonality = seasonality.loc[f'{START_YEAR}':f'{END_YEAR}'].T
        return seasonality
        
    @lazyproperty
    def INDMOM(self):
        window = 6 * 21; half_life = 21
        logret = np.log(1 + self._fillna(self.changepct / 100, 0))
        rs = self._rolling(logret, window, half_life, 'sum')
        
        cap_sqrt = np.sqrt(self.negotiablemv)
        ind_citic_lv1 = self.firstind
        rs, cap_sqrt, ind_citic_lv1 = self._align(rs, cap_sqrt, ind_citic_lv1)
        
        dat = pd.DataFrame()
        for df in [rs, cap_sqrt, ind_citic_lv1]:
            df.index.name = 'time'
            df.columns.name = 'code'
            dat = pd.concat([dat, df.unstack()], axis=1)

        dat.columns = ['rs', 'weight', 'ind']
        dat = dat.reset_index()
        
        rs_ind = {(time, ind): (df['weight'] * df['rs']).sum() / df['weight'].sum()
                  for time, df_gp in dat.groupby(['time']) 
                  for ind, df in df_gp.groupby(['ind'])}
        
        def _get(key):
            nonlocal rs_ind
            try:
                return rs_ind[key]
            except:
                return np.nan
            
        dat['rs_ind'] = [_get((date, ind)) for date, ind in zip(dat['time'], dat['ind'])]
        dat['indmom'] = dat['rs_ind'] - dat['rs'] * dat['weight'] / dat['weight'].sum()
        indmom = pd.pivot_table(dat, values='indmom', index=['code'], columns=['time'])
        return indmom

    @lazyproperty
    def RSTR(self, version=VERSION):
        benchmark_ret = self._get_benchmark_ret()
        stock_ret = self.changepct / 100
        benchmark_ret, stock_ret = self._align(benchmark_ret, stock_ret)
        benchmark_ret = benchmark_ret[BENCHMARK]
        
        excess_ret = np.log((1 + stock_ret).divide((1 + benchmark_ret), axis=0))
        if version == 6:
            rstr = self._rolling(excess_ret, window=252, half_life=126, func_name='sum')
            rstr = rstr.rolling(window=11, min_periods=1).mean()
        elif version == 5:
            exp_wt = self._get_exp_weight(504+21, 126)[:504]
            rstr = self._rolling(excess_ret.shift(21), window=504, weights=exp_wt, 
                                 func_name='sum')
        return rstr
    #5 ** window=504, l=21, hl=126
#5******Quality
class Quality(CALFUNC):
    pass

class Leverage(Quality):
    @lazyproperty
    def MLEV(self, version=VERSION):
#            longdebttoequity, be = self._align(self.longdebttoequity, self.totalshareholderequity)
#            ld = be * longdebttoequity
        if version == 6:
            method = 'lyr' 
        elif version == 5:
            method = 'mrq'
        ld = self._transfer_freq(self.totalnoncurrentliability, 
                                 method=method, from_='q', to_='d')
        pe = self._transfer_freq(self.preferedequity, 
                                 method=method, from_='q', to_='d')
        me = self._shift(self.totalmv, shift=1)
        me, pe, ld = self._align(me, pe, ld)
        mlev = (me + pe + ld) / me
        return mlev.T
    #5 ** pe, ld ---- mrq
    @lazyproperty
    def BLEV(self, version=VERSION):
        if version == 6:
            method = 'lyr' 
        elif version == 5:
            method = 'mrq'
        ld = self._transfer_freq(self.totalnoncurrentliability, 
                                 method=method, from_='q', to_='d')
        pe = self._transfer_freq(self.preferedequity, 
                                 method=method, from_='q', to_='d')
        be = self._transfer_freq(self.totalshareholderequity, 
                                 method=method, from_='q', to_='d')
        be, pe, ld = self._align(be, pe, ld)
        blev = (be + pe + ld) / be
        return blev.T
    #5 ** oe, ld, be ---- mrq
    @lazyproperty
    def DTOA(self, version=VERSION):
        if version == 6:
            tl = self._transfer_freq(self.totalliability, 
                                     method='lyr', from_='q', to_='d')
            ta = self._transfer_freq(self.totalassets,
                                     method='lyr', from_='q', to_='d')
            tl, ta = self._align(tl, ta)
            dtoa = tl / ta
        elif version == 5:
            sewmi_to_ibd, sewmit_to_tl, tl = self._align(self.sewmitointerestbeardebt, 
                                           self.sewithoutmitotl, self.totalliability)
            ibd = tl * (sewmit_to_tl / sewmi_to_ibd)
            ta, td = self._align(self.totalassets, ibd)
            dtoa = td / ta
            dtoa = self._transfer_freq(dtoa, method='mrq', from_='q', to_='d')
        return dtoa.T
    #5 ** dtoa = td / ta; td -- long-term debt+current liabilities;td,ta ---- mrq
class EarningsVariablity(Quality):
    window = 5
    @lazyproperty
    def VSAL(self):
        sales_y = self._transfer_freq(self.operatingreenue, None, 
                                      from_='q', to_='y')
        std = sales_y.rolling(window=self.window).std() 
        avg = sales_y.rolling(window=self.window).mean()
        vsal = std / avg
        
        vsal = self._transfer_freq(vsal, method='lyr', from_='q', to_='d')
        return vsal.T
    
    @lazyproperty
    def VERN(self):
        earnings_y = self._transfer_freq(self.netprofit, None, 
                                         from_='q', to_='y')
        std = earnings_y.rolling(window=self.window).std() 
        avg = earnings_y.rolling(window=self.window).mean()
        vern = std / avg
        
        vern = self._transfer_freq(vern, method='lyr', from_='q', to_='d')
        return vern

    @lazyproperty
    def VFLO(self):
        cashflows_y = self._transfer_freq(self.cashequialentincrease, None, 
                                          from_='q', to_='y')
        std = cashflows_y.rolling(window=self.window).std()
        avg = cashflows_y.rolling(window=self.window).mean()
        vflo = std / avg
        
        vflo = self._transfer_freq(vflo, method='lyr', from_='q', to_='d')
        return vflo.T
    
#    @lazyproperty
#    def ETOPF_STD(self):
#        etopf = self.west_eps_ftm.T
#        etopf_std = etopf.rolling(window=240).std()
#        close = self.clean_data(self.close)
#        etopf_std, close = self._align(etopf_std, close)
#        etopf_std /= close
#        return etopf_std.T

class EarningsQuality(Quality):
    @lazyproperty
    def ABS(self):
        cetoda, ce = self._align(self.capitalexpendituretodm, self.capital_expenditure) #wind:资本支出/折旧加摊销，资本支出
        cetoda = cetoda.apply(linear_interpolate)
        da = ce / cetoda #此处需对cetoda插值填充处理
        #lc_mainindexdata:归属母公司股东的权益/带息债务(%), 归属母公司股东的权益/负债合计(%), 负债合计
        sewmi_to_ibd, sewmit_to_tl, tl = self._align(self.sewmitointerestbeardebt, 
                                           self.sewithoutmitotl, self.totalliability)
        ibd = tl * (sewmit_to_tl / sewmi_to_ibd)
        
        ta, cash, tl, td = self._align(self.totalassets, self.cashequialents, 
                                       self.totalliability, ibd)
        noa = (ta - cash) - (tl - td)
        
        noa, da = self._align(noa, da)
        accr_bs = noa - noa.shift(1) - da
        
        accr_bs, ta = self._align(accr_bs, ta)
        abs_ = - accr_bs / ta
        abs_ = self._transfer_freq(abs_, method='mrq', from_='q', to_='d')
        return abs_.T
        
    @lazyproperty
    def ACF(self):
        cetoda, ce = self._align(self.capitalexpendituretodm, self.capital_expenditure) #wind:资本支出/折旧加摊销，资本支出
        cetoda = cetoda.apply(linear_interpolate)
        da = ce / cetoda #此处需对cetoda插值填充处理
        ni, cfo, cfi, da = self._align(self.netprofit, self.netoperatecashflow, 
                                       self.netinvestcashflow, da)
        accr_cf = ni - (cfo + cfi) + da
        
        accr_cf, ta = self._align(accr_cf, self.totalassets)
        acf = - accr_cf / ta
        acf = self._transfer_freq(acf, method='mrq', from_='q', to_='d')
        return acf.T
            
class Profitability(Quality):
    @lazyproperty
    def ATO(self):
        sales = self._transfer_freq(self._get_ttm_data(self.operatingreenue),
                                    method='mrq', from_='q', to_='d')
        ta = self._transfer_freq(self.totalassets, method='mrq', 
                                 from_='q', to_='d')
        sales, ta = self._align(sales, ta)
        ato = sales / ta
        return ato.T
    
    @lazyproperty
    def GP(self):
        sales = self._transfer_freq(self.operatingreenue,
                                    method='lyr', from_='q', to_='d')
        cogs = self._transfer_freq(self.cogs_q,
                                   method='lyr', from_='q', to_='d')
        ta = self._transfer_freq(self.totalassets,
                                 method='lyr', from_='q', to_='d')
        sales, cogs, ta = self._align(sales, cogs, ta)
        gp = (sales - cogs) / ta
        return gp.T
        
    @lazyproperty
    def GPM(self):
        sales = self._transfer_freq(self.operatingreenue,
                                    method='lyr', from_='q', to_='d')
        cogs = self._transfer_freq(self.cogs_q,
                                   method='lyr', from_='q', to_='d')
        sales, cogs = self._align(sales, cogs)
        gpm = (sales - cogs) / sales
        return gpm.T
        
    @lazyproperty
    def ROA(self):
        earnings = self._transfer_freq(self._get_ttm_data(self.netprofit),
                                       method='mrq', from_='q', to_='d')
        ta = self._transfer_freq(self.totalassets,
                                 method='mrq', from_='q', to_='d')
        earnings, ta = self._align(earnings, ta)
        roa = earnings / ta
        return roa.T
    
class InvestmentQuality(Quality):
    window = 5
    @lazyproperty
    def AGRO(self):
        agro = self._get_growth_rate(self.totalassets, periods=self.window, 
                                     freq='y')
        return agro
    
    @lazyproperty
    def IGRO(self):
        igro = self._get_growth_rate(self.totalshares, periods=self.window, 
                                     freq='y')
        return igro
    
    @lazyproperty
    def CXGRO(self):
        cxgro = self._get_growth_rate(self.capital_expenditure, 
                                      periods=self.window, freq='y')
        return cxgro

#6*******Value
class Value(CALFUNC):
    @lazyproperty
    def BTOP(self):
        bv = self._transfer_freq(self.sewithoutmi, method='mrq', from_='q', to_='d')
        bv, mkv = self._align(bv, self.totalmv)
        btop = bv / (mkv * 10000)
        return btop.T
    #5 --
    #*****Earnings Yield
class EarningsYield(Value):
    @lazyproperty
    def ETOP(self):
        earings_ttm = self._transfer_freq(self._get_ttm_data(self.netprofit), 
                                          method='mrq', from_='q', to_='d')
        e_ttm, mkv = self._align(earings_ttm, self.totalmv)
        etop = e_ttm / (mkv * 10000)
        return etop.T
    #5 --
#        @lazyproperty
#        def ETOPF(self):
#            pass
    
    @lazyproperty
    def CETOP(self):
        cash_earnings = self._transfer_freq(self._get_ttm_data(self.netoperatecashflow),
                                            method='mrq', from_='q', to_='d')
        ce, mkv = self._align(cash_earnings, self.totalmv)
        cetop = ce / (mkv * 10000)
        return cetop.T
    #5 --
    @lazyproperty
    def EM(self):
        ebit = self._transfer_freq(self.ebit, method='lyr', from_='q', to_='d')
        
        sewmi_to_ibd, sewmit_to_tl, tl = self._align(self.sewmitointerestbeardebt, 
                                           self.sewithoutmitotl, self.totalliability)
        ibd = tl * (sewmit_to_tl / sewmi_to_ibd)
        ibd = self._transfer_freq(ibd, method='mrq', from_='q', to_='d')
        
        cash = self._transfer_freq(self.cashequialents, method='mrq', from_='q', to_='d')
        ebit, mkv, ibd, cash = self._align(ebit, self.totalmv, ibd, cash) 
        
        ev = mkv * 10000 + ibd - cash
        em = ebit / ev
        return em.T
    
class LongTermReversal(Value):
    @lazyproperty
    def LTRSTR(self):
        self = CALFUNC()
        benchmark_ret = self._get_benchmark_ret(BENCHMARK)
        stock_ret = self.changepct / 100
        benchmark_ret, stock_ret = self._align(benchmark_ret, stock_ret)
        benchmark_ret = benchmark_ret[BENCHMARK]
        
        excess_ret = np.log((1 + stock_ret).divide((1 + benchmark_ret), axis=0))
        ltrstr = self._rolling(excess_ret, window=1040, half_life=260, func_name='sum').T
        ltrstr = (-1) * ltrstr.shift(273).rolling(window=11).mean()
        return ltrstr.T
    
    @lazyproperty
    def LTHALPHA(self):
        _, alpha, _ = self._capm_regress(window=1040, half_life=260)
        lthalpha = (-1) * alpha.T.shift(273).rolling(window=11).mean()
        return lthalpha.T
        
#7*******Growth
class Growth(CALFUNC):
    window = 5
#    @lazyproperty
#    def EGRLF(self):
#        pass
    
    @lazyproperty
    def EGRO(self):        
        egro = self._get_growth_rate(self.eps, periods=self.window, 
                                     freq='y')
        return egro
    #5 --
    @lazyproperty
    def SGRO(self):
        total_shares, operatingrevenue = self._align(self.totalshares, self.operatingreenue)
        ops = operatingrevenue / total_shares
        sgro = self._get_growth_rate(ops, periods=self.window,
                                     freq='y')
        return sgro
    #5 -- 
#8*******Sentiment
#class Sentiment(CALFUNC):
#    @lazyproperty
#    def RRIBS(self):
#        pass
#    
#    @lazyproperty
#    def EPIBSC(self):
#        pass
#    
#    @lazyproperty
#    def EARNC(self):
#        pass

#9*******DividendYield
class DividendYield(CALFUNC):
    @lazyproperty
    def DTOP(self):
        dps = self._transfer_freq(self._get_ttm_data(self._fillna(self.dividendps, value=0)),
                                  method='mrq', from_='q', to_='d')
        price_lme = self._get_price_last_month_end('close')
        dps, price_lme = self._align(dps, price_lme)
        dtop = dps / price_lme
        return dtop.T
    
#    @lazyproperty
#    def DPIBS(self):
#        pass


if __name__ == '__main__':
    factor_styles = [name for name in globals().keys()
                     if name[0].isupper() and name[1].islower()
                     and name not in ('In', 'Out', 'Data')] 
    cne5 = ['BETA', 'HSIGMA', 'CMRA', 'RSTR', 'MLEV', 'BLEV', 'DTOA']
    for style in factor_styles:
        fstyle = globals()[style]()
        factors_names = [name for name in dir(fstyle) if name.isupper()]
        for factor in factors_names:
            if VERSION == 5:
                if factor not in cne5:
                    continue
                save_name = factor+'_5.csv'
            else:
                save_name = factor+'.csv'
            if os.path.exists(os.path.join(fstyle.save_path, save_name)):
                print('{} already exists.'.format(factor+'.csv'))
                continue
            if VERSION == 5:
                print(f'Calculating {factor}_5...')
            else:
                print(f'Calculating {factor}...')
            res = fstyle.clean_data(getattr(fstyle, factor), limit_days=True)
            res = res.T
            res = res.applymap(lambda s: float(s))
            res.index.name = 'code'
            cond = (res >= 0) | (~np.isinf(res)) 
            res = res.where(cond, -SENTINEL)
            fstyle.save(res, save_name)
            print('*'*80)
        

