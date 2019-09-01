# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 16:56:01 2018

@author: admin
"""
import os
import warnings
import numpy as np
import pandas as pd
import pandas.tseries.offsets as toffsets
from dask import dataframe as dd
warnings.filterwarnings('ignore')

# 原始因子文件所在根目录，可根据需要自行修改
WORK_PATH = os.path.dirname(__file__)
# 合成的barra因子存放根目录，根据需要自行修改
SAVE_PATH = os.path.dirname(__file__)

START_TIME = '2009-01-01'
END_TIME = '2019-4-30'

def ensure_time(x):
    try:
        return pd.to_datetime(x)
    except Exception:
        return x
        
class Data:
    global WORK_PATH, SAVE_PATH    
    ori_root = WORK_PATH
    save_root = SAVE_PATH
    pathmap = {}
    
    def __init__(self):
        self.basic_datapath = os.path.join(self.ori_root, "basic")
        self.market_datapath = os.path.join(self.ori_root, "market_quote")
        self.finance_datapath = os.path.join(self.ori_root, "finance")
        self.save_path = os.path.join(self.save_root, "factor_data")
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
            
        self.pathmap.update({self.__name_modify(name): (self.basic_datapath, name) for name in os.listdir(self.basic_datapath)})
        self.pathmap.update({self.__name_modify(name): (self.market_datapath, name) for name in os.listdir(self.market_datapath)})
        self.pathmap.update({self.__name_modify(name): (self.finance_datapath, name) for name in os.listdir(self.finance_datapath)})
        self.pathmap.update({self.__name_modify(name).upper(): (self.save_path, name) for name in os.listdir(self.save_path)})
        self._ori_factors = sorted(self.pathmap.keys())

    def __name_modify(self, f_name):
        if f_name.endswith('.csv') or f_name.endswith('.xlsx'):
            f_name = f_name.split('.')[0]
            if all(n.isnumeric() for n in f_name[-10:].split('-')):
                f_name = f_name[:-11]
        return f_name.lower()

    def _open_file(self, name, **kwargs):        
        path, file_name = self.pathmap[name]
        
        ext = file_name.split('.')[-1]
        if ext == 'csv':
            dat = self.__read_csv(path, file_name, name, **kwargs)
        elif ext == 'xlsx':
            dat = self.__read_excel(path, file_name, name, **kwargs)
        else:
            msg = f"不支持的文件存储类型：{ext}"
            raise TypeError(msg)
        return dat
        
    def __read_csv(self, path, rname, fname, **kwargs):
        read_conds = {
                'encoding':'gbk',
                'engine':'python', 
                'index_col':[0],
                **kwargs
                }
        dat = pd.read_csv(os.path.join(path, rname), **read_conds)
        dat.columns = pd.to_datetime(dat.columns)
        if fname in ('close', 'mkt_cap_float_d'):
            dat = dat.where(dat != 0, np.nan)
        if fname in ('stm_issuingdate', 'applied_rpt_date_M'): 
#                     'applied_rpt_date_d', 'applied_lyr_date_d'):
            dat = dat.where(dat != '0', pd.NaT)
            dat = dat.applymap(ensure_time)
        return dat
    
    def __read_excel(self, path, rname, fname, **kwargs):
        path = os.path.join(path, rname)
        
        if fname == 'all_stocks_code':
            kwargs['parse_dates'] = ['ipo_date', "delist_date"]
        elif fname == 'all_index_code':
            kwargs['parse_dates'] = ['PubDate', 'EndDate']
        elif fname == 'month_map':
            kwargs['parse_dates'] = ['trade_date', 'calendar_date']
        elif fname == 'month_group':
            kwargs['parse_dates'] = ['calendar_date']
        else:
            kwargs['index_col'] = [0]
        dat = pd.read_excel(path, encoding='gbk', **kwargs)
        
        if fname not in ('all_stocks_code', 'all_index_code', 
                         'month_map', 'month_group'):
            dat.columns = pd.to_datetime(dat.columns)
        
        if fname in ('cogs_q','ebitps_q','ev2_m','net_profit_ttm_q',
                     'netprofit_report_q', ):
            dat = dat.where(dat != 0, np.nan)
        
        if fname in ('month_map', 'month_group'):
            dat = dat.set_index(['calendar_date'])
        return dat
    
    def save(self, df, name, **kwargs):
        path = os.path.join(self.save_path, name)
        if name.endswith('csv'):
            df.to_csv(path, encoding='gbk', **kwargs)
        elif name.endswith('xlsx'):
            df.to_excel(path, encoding='gbk', **kwargs)
        print(f'Save {name} successfully.')
    
    def __getattr__(self, name, **kwargs):
        if name not in self.__dict__:
            name = self._get_fac_valid_name(name)
            res = self._open_file(name, **kwargs)
            self.__dict__[name] = res
        return self.__dict__[name]

    def _get_fac_valid_name(self, name):
        if name not in self._ori_factors:
            i = 0
            while True:
                try:
                    cur_fname = self._ori_factors[i]
                    if cur_fname.startswith(name):
                        name = cur_fname
                        break
                    i += 1
                except IndexError:
                    msg = f"请确认因子名称{name}是否正确"
                    raise Exception(msg) 
        return name
    
#    @staticmethod
#    def __lyr_date(date):
#        if date.month == 12:
#            return date
#        else:
#            try:
#                return pd.to_datetime(f'{date.year-1}-12-31')
#            except:
#                return pd.NaT
            
#    def _get_lyr_date(self):        
#        applied_rpt_date_d = self.applied_rpt_date_M
#        applied_lyr_date_d = applied_rpt_date_d.applymap(self.__lyr_date)
#        applied_lyr_date_d.to_csv(os.path.join(WORK_PATH, 'basic', 'applied_lyr_date_d.csv'),
#                                  encoding='gbk')
        
    def _generate_month_group(self, start_year=None, end_year=None):
        ori_syear, ori_eyear = self.month_map.index[0].year, self.month_map.index[-1].year
        if start_year is None and end_year is None:
            start_year, end_year = ori_syear, ori_eyear
            
        month_group = pd.DataFrame(index=self.month_map.index)
        
        group1 = [[None]*2+[i+1]*5+[None]*5 for i in range(len(range(ori_syear, ori_eyear+1)))]
        group1 = [i for group in group1 for i in group]
        
        group2 = [[None]*5+[i+1]+[None]+[i+1]*2+[None]*3 for i in range(len(range(ori_syear, ori_eyear+1)))]
        group2 = [i for group in group2 for i in group]
        
        group3 = [[i]*3+[None]*5+[i+1]*4 for i in range(len(range(ori_syear, ori_eyear+1)))]
        group3 = [i for group in group3 for i in group]
        
        month_group['Q1'] = group1
        month_group['Q2'] = group2
        month_group['Q3'] = group3
        
        month_group = pd.concat([month_group, self.month_map], axis=1)
        month_group = month_group.loc[start_year:end_year]
        
        month_group.to_excel(os.path.join(WORK_PATH, 'basic', 'month_group.xlsx'))
        
    def reindex(self, df, to='wind', if_index=False):
        dat = df.copy()
        if all('.' in code for code in dat.index) and to == 'wind':
            return dat
        if all(code.startswith('`') for code in dat.index) and to == 'juyuan':
            return dat
        
        if if_index:
            all_codes = getattr(self, 'all_index_code',)
        else:
            all_codes = getattr(self, 'all_stocks_code',)
        if to == 'wind':
            idx_code = 'juyuan_code'
        else:
            idx_code = 'wind_code'
        code_map = all_codes[['juyuan_code', 'wind_code']].set_index(idx_code) 
        new_idx = code_map.loc[df.index]
        new_idx_val = new_idx.values.flatten()
        dat.index = np.where(pd.isna(new_idx_val), new_idx.index, new_idx_val)
        return dat

if __name__ == "__main__":
    #测试代码
    dat = Data()   
    
    #test1 -- 读取因原始子文件的两种方式，推荐前者（因子名称需为小写）
    citic_level1_1 = dat.firstindustryname
    citic_level1_2 = getattr(dat, 'firstindustryname',)
    print(citic_level1_1 is citic_level1_2)
    
    #test2 -- 名称模糊查找，目前仅支持可以输入因子的简称，
    #即所输入的因子简称可与因子全称的前若干位字符相匹配
    citic_level1_3 = dat.firstind
    print(citic_level1_1 is citic_level1_3)
    
    #test3-索引转换-（证券代码在wind与聚源间互换）
    ta = dat.totalassets
    print(ta.index[:5])
    ta_1 = dat.reindex(ta, to='wind')
    print(ta_1.index[:5])
    
    close = dat.close
    close1 = dat.reindex(close, to='juyuan')
    print(close.index[:5])
    print(close1.index[:5])

