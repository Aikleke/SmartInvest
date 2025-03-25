import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')  # 添加这行代码

def get_drawdown(p):
    """
    计算净值回撤
    """
    T = len(p)
    hmax = p.cummax()
    dd = p / hmax - 1

    return dd


def cal_period_perf_indicator(adjnav):
    """
    计算区间业绩指标:输入必须是日频净值
    """

    if type(adjnav) == pd.DataFrame:
        res = pd.DataFrame(index=adjnav.columns, columns=['AnnRet', 'AnnVol', 'SR', 'MaxDD', 'Calmar'])
        for col in adjnav:
            res.loc[col] = cal_period_perf_indicator(adjnav[col])

        return res

    ret = adjnav.pct_change()
    # annret = np.nanmean(ret) * 242 # 单利
    annret = (adjnav[-1] / 1) ** (242/len(adjnav)) - 1 # 复利
    annvol = np.nanstd(ret) * np.sqrt(242)
    sr = annret / annvol
    dd = get_drawdown(adjnav)
    mdd = np.nanmin(dd)
    calmar = annret / -mdd

    return [annret, annvol, sr, mdd, calmar]

def datestr2dtdate(datestr, format='%Y-%m-%d'):
    # 日期格式转换：'yyyy-mm-dd'转为datetime.date
    return datetime.datetime.strptime(datestr, format).date()

def date_count_in_month(dates):
    # 计算日期序列中每个日期在所在月中的序数
    cur_count = 1
    counts = [cur_count]
    for i in range(1,len(dates)):
        if dates[i].month == dates[i-1].month:
            cur_count = cur_count + 1
        else:
            cur_count = 1
        counts.append(cur_count)
    return counts


# 获取交易日序列
def get_trading_dates(start_date=None, end_date=None):
    """
    读取指定起止日期之间的交易日序列
    """
    dates = pd.read_csv('trading_date.csv')['trade_date'].to_list()
    dates = [datestr2dtdate(e, '%Y/%m/%d') for e in dates]
    if start_date is not None:
        dates = [e for e in dates if e >= start_date]
    if end_date is not None:
        dates = [e for e in dates if e <= end_date]
    return dates


# 数据接口
def get_hist_data(index_ids=None, end_date=None):
    """
    读取指数历史数据到指定截止日
    Input:
        index_ids: list of str, 指数代码列表, like ['hs300', 'csi500']
        end_date: datetime.date, 截止日期
    Output:
        data: df(date*, index1, index2, ...), 多个指数的历史收盘价序列
    """
    # 从csv文件获取指数价格数据
    data = pd.read_csv('basic_data.csv').set_index('datetime')
    data.index = [datestr2dtdate(e) for e in data.index]
    print('基础数据起止日期：%s，%s' % (data.index[0], data.index[-1]))
    if index_ids is not None:
        data = data.loc[:, index_ids]
    if end_date is not None:
        data = data.loc[:end_date, :]
    return data


# 日历策略
def calendar_stgy(data, start_date, end_date, params):
    """
    开盘前调用，返回目标组合权重
    Input:
        data: df(date*, index1, index2, ...), basic data
        start_date, end_date: 'yyyy-mm-dd' or datetime.date
        params: dict, format {'index_id':'hs300', 't1':1, 't2':5}
    Output:
        target_wgt: df(trade_date*, index1, index2, ...) 目标权重
    """
    if type(start_date) is str:
        start_date = datestr2dtdate(start_date)
    if type(end_date) is str:
        end_date = datestr2dtdate(end_date)
    index_id = params['index_id']
    t1 = params['t1']
    t2 = params['t2']

    start_date0 = start_date - datetime.timedelta(31)
    dates0 = get_trading_dates(start_date0, end_date)
    dates0_rank = date_count_in_month(dates0)
    target_wgt = pd.DataFrame(data=0, index=dates0, columns=data.columns)
    target_wgt[index_id] = [1 if (e >= t1 and e <= t2) else 0 for e in dates0_rank]
    target_wgt = target_wgt.loc[start_date:end_date]
    return target_wgt


# 轮动策略
def rotation_stgy(data, start_date, end_date, params):
    """
    开盘前调用，返回目标组合权重
    Input:
        data: df(date*, index1, index2, ...), basic data
        start_date, end_date: 'yyyy-mm-dd' or datetime.date
        params: dict, format {'index_list':['N':20}
    Output:
        target_wgt: df(trade_date*, index1, index2, ...) 目标权重
    """
    if type(start_date) is str:
        start_date = datestr2dtdate(start_date)
    if type(end_date) is str:
        end_date = datestr2dtdate(end_date)
    N = params['N']

    start_date0 = start_date - datetime.timedelta(N) * 2
    dates0 = get_trading_dates(start_date0, end_date)
    data0 = data.reindex(index=dates0)
    N_day_ret = data0.shift(1) / data0.shift(N + 1) - 1  # 截止昨收的最近N个交易日涨幅
    target_wgt = pd.DataFrame(index=data0.index, columns=data0.columns)
    target_wgt['hs300'] = [1 if e > 0 else 0 if e <= 0 else np.nan for e in N_day_ret['hs300'] - N_day_ret['csi500']]
    target_wgt['csi500'] = 1 - target_wgt['hs300']
    target_wgt = target_wgt.loc[start_date:end_date].fillna(0)

    return target_wgt


# 轮动策略（可以空仓版）
def rotation_stgy1(data, start_date, end_date, params):
    """
    开盘前调用，返回目标组合权重
    Input:
        data: df(date*, index1, index2, ...), basic data
        start_date, end_date: 'yyyy-mm-dd' or datetime.date
        params: dict, format {'index_list':['N':20}
    Output:
        target_wgt: df(trade_date*, index1, index2, ...) 目标权重
    """
    if type(start_date) is str:
        start_date = datestr2dtdate(start_date)
    if type(end_date) is str:
        end_date = datestr2dtdate(end_date)
    N = params['N']

    start_date0 = start_date - datetime.timedelta(N) * 2
    dates0 = get_trading_dates(start_date0, end_date)
    data0 = data.reindex(index=dates0)
    N_day_ret = data0.shift(1) / data0.shift(N + 1) - 1  # 截止昨收的最近N个交易日涨幅
    target_wgt = pd.DataFrame(0, index=data0.index, columns=data0.columns)
    for i in range(1, len(target_wgt)):
        t = target_wgt.index[i]
        t0 = target_wgt.index[i - 1]
        if N_day_ret.loc[t0, 'hs300'] >= N_day_ret.loc[t0, 'csi500'] and N_day_ret.loc[t0, 'hs300'] > 0:
            target_wgt.loc[t, 'hs300'] = 1
        elif N_day_ret.loc[t0, 'hs300'] < N_day_ret.loc[t0, 'csi500'] and N_day_ret.loc[t0, 'csi500'] > 0:
            target_wgt.loc[t, 'csi500'] = 1
    target_wgt = target_wgt.loc[start_date:end_date].fillna(0)

    return target_wgt


## 实测程序

def fastTest(start_date, end_date):
    # 读取基础数据
    data = get_hist_data(end_date=end_date)

    # 调用策略模块生成目标组合权重
    target_wgt1 = calendar_stgy(data, start_date, end_date, params={'index_id':'hs300', 't1':1, 't2':5})
    target_wgt2 = calendar_stgy(data, start_date, end_date, params={'index_id':'csi1000', 't1':1, 't2':5})
    target_wgt3 = rotation_stgy(data, start_date, end_date, params={'N':20})
    target_wgt4 = rotation_stgy1(data, start_date, end_date, params={'N':20})
    target_wgt = 0*target_wgt1 + 0.5*target_wgt2 + 0*target_wgt3 + 0.5*target_wgt4 # 多策略目标组合整合

    # 产生每日持仓权重
    hold_wgt = target_wgt # 假设每天都可以准确地执行交易计划

    # 计算组合业绩

    # 获得这个数值比上一个的变化率
    asset_ret = data.pct_change().loc[start_date:end_date]
    # 计算累计收益率 
    # 1. 第一天净值 = 1 (1 + 5%) = 1.05
    # 2. 第二天净值 = 1.05 (1 + 3%) = 1.0815
    # 3. 第三天净值 = 1.0815 (1 - 2%) = 1.05987
    # 4. 第四天净值 = 1.05987 (1 + 4%) = 1.102265
    res = (1+asset_ret).cumprod()
    # 把持仓数据向下推一行，因为我们用昨天的持仓权重来计算今天的收益
    # 比如：如果某资产权重是0.6，当天收益率是2%，则贡献的收益是 0.6 2% = 1.2%
    res['account'] = (1+(hold_wgt.shift(2) * asset_ret).sum(axis=1)).cumprod()

    # 展示净值曲线图和业绩指标表
    res.loc[:,['hs300','csi500','account']].plot(figsize=(16,8), grid=True)
    plt.show()

    cal_period_perf_indicator(res.loc[:,['hs300','csi500','account']])





## 实盘程序

"""
    开盘前
"""
def safeTradeBefore(T, target_amount):
    # 读取基础数据：截止T-1日
    # 通过基础历史数据，形成决策指南
    data = get_hist_data(end_date=T-datetime.timedelta(days=1))

    # 生成目标组合权重
    target_wgt1 = calendar_stgy(data, start_date=T, end_date=T, params={'index_id':'csi1000', 't1':1, 't2':5})
    target_wgt2 = rotation_stgy1(data, start_date=T, end_date=T, params={'N':20})
    target_wgt = 0.5*target_wgt1 + 0.5*target_wgt2

    # 输出目标持仓市值
    target_mv = target_wgt * target_amount
    print('目标持仓市值：')
    print(target_mv)

# 设置回测参数
start_date = datetime.date(2004,12,31) # 回测起始日期
end_date = datetime.date(2021,7,31) # 回测截止日期
# 回测函数
fastTest(start_date, end_date)


T = datetime.date(2020,8,3) # 设置拟交易日期
target_amount = 100000 # 目标投资金额
# 实盘函数
safeTradeBefore(T, target_amount)

"""
    交易时段和盘后
"""
# 盘中：按目标持仓市值完成交易
# 盘后：导出持仓明细，写入账户持仓记录表




"""
    业绩分析：定期进行
"""
# 从账户持仓记录表读取持仓市值数据
hold_mv = pd.read_csv('账户持仓记录表.csv').set_index('date')
hold_mv.index = [datestr2dtdate(e) for e in hold_mv.index]
start_date = hold_mv.index[0]
end_date = hold_mv.index[-1]

# 化为权重
hold_wgt = hold_mv.iloc[:,0:4].copy()
for t in hold_wgt.index:
    hold_wgt.loc[t] = hold_wgt.loc[t] / hold_mv.loc[t,'amount']

# 计算净值
data = get_hist_data(end_date=end_date)
asset_ret = data.pct_change().loc[start_date:end_date]
res = (1+asset_ret).cumprod()
res['account'] = (1+(hold_wgt.shift(1) * asset_ret).sum(axis=1)).cumprod()

# 展示净值曲线图和业绩指标表
res.loc[:,['hs300','csi500','account']].plot(figsize=(16,8), grid=True)
plt.show()
cal_period_perf_indicator(res.loc[:,['hs300','csi500','account']])