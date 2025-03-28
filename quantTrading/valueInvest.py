import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import akshare as ak
from pathlib import Path

# 设置pandas显示选项，显示所有列
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', 1000)  # 加宽显示宽度
pd.set_option('display.expand_frame_repr', False)  # 禁止自动换行

# 定义数据根目录
DATA_ROOT = os.path.expanduser('~/investData')

# 确保数据根目录存在
os.makedirs(DATA_ROOT, exist_ok=True)

plt.switch_backend('TkAgg')  # 可以尝试其他后端

import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import concurrent.futures
import time
from datetime import datetime

from valueInvestGetData import (
    load_stock_history,
    load_financial_data,
    get_stock_pe_at_date,
    get_all_stocks_pe,
    download_all_financial_data,
    download_all_stocks_history,
    get_and_save_stock_codes
)

# 设置中文字体
try:
    # 对于MacOS
    font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
    plt.rcParams['font.family'] = ['PingFang HK']
except:
    # 对于其他系统
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_stock_price_and_pe(stock_code, start_date, end_date, financial_data=None):
    """
    分析股票价格和PE的关系
    
    参数:
    stock_code (str): 股票代码
    start_date (str): 开始日期，格式为'YYYYMMDD'
    end_date (str): 结束日期，格式为'YYYYMMDD'
    financial_data (pd.DataFrame): 财务数据，如果为None则从文件加载
    
    返回:
    tuple: (price_series, pe_series) 包含价格和PE的Series
    """
    # 1. 从本地文件获取股票每日收盘价
    stock_price_df = load_stock_history(stock_code, start_date, end_date)
    if stock_price_df is None:
        print(f"未找到股票 {stock_code} 的历史数据")
        return None, None

    # 2. 获取财务数据
    if financial_data is None:
        financial_data = load_financial_data(stock_code)
        if financial_data is None or financial_data.empty:
            print(f"未找到股票 {stock_code} 的财务数据")
            return None, None

    # 3. 数据处理
    # 处理日期格式
    stock_price_df['日期'] = pd.to_datetime(stock_price_df['日期'])
    stock_price_df.set_index('日期', inplace=True)
    
    # 处理财务数据
    financial_data['基本每股收益'] = pd.to_numeric(financial_data['基本每股收益'], errors='coerce')
    
    # 确保日期索引格式化正确
    stock_price_df.index = pd.to_datetime(stock_price_df.index).date
    financial_data.index = pd.to_datetime(financial_data.index).date

    # 将每股收益数据填充到每个交易日
    eps_daily = financial_data['基本每股收益'].reindex(stock_price_df.index, method='ffill')

    # 4. 计算PE
    pe_daily = stock_price_df['收盘'] / eps_daily

    # 5. 创建可视化图表
    plt.style.use('default')  # 使用默认样式而不是seaborn
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # 设置背景色
    fig.patch.set_facecolor('#f8f9fa')
    ax1.set_facecolor('#f8f9fa')

    # 定义颜色
    price_color = '#2ecc71'  # 清新的绿色
    pe_color = '#e74c3c'    # 优雅的红色

    # 绘制股价走势（左轴）
    ax1.plot(stock_price_df.index, stock_price_df['收盘'], color=price_color, label='股价', linewidth=2)
    ax1.set_xlabel('日期', fontsize=12, color='#2c3e50', labelpad=10)
    ax1.set_ylabel('股价(元)', color=price_color, fontsize=12, labelpad=10)
    ax1.tick_params(axis='y', labelcolor=price_color)
    ax1.grid(False)  # 删除网格线

    # 创建右轴并绘制PE走势
    ax2 = ax1.twinx()
    ax2.plot(pe_daily.index, pe_daily, color=pe_color, label='PE', linewidth=2)
    ax2.set_ylabel('PE比率', color=pe_color, fontsize=12, labelpad=10)
    ax2.tick_params(axis='y', labelcolor=pe_color)

    # 设置标题
    plt.title(f'{stock_code} 股价与PE走势对比', fontsize=14, color='#2c3e50', pad=20)

    # 设置x轴日期格式
    ax1.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, color='#2c3e50')

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
                       frameon=True, facecolor='white', edgecolor='#bdc3c7',
                       shadow=True, fontsize=10)
    legend.get_frame().set_alpha(0.9)

    # 设置边框样式
    for spine in ax1.spines.values():
        spine.set_color('#bdc3c7')
        spine.set_linewidth(0.5)
        spine.set_alpha(0.5)

    plt.tight_layout()
    plt.show()

    # 6. 输出相关系数
    correlation = stock_price_df['收盘'].corr(pe_daily)
    print(f"\n股价与PE的相关系数: {correlation:.4f}")

    return stock_price_df['收盘'], pe_daily

def analyze_low_pe_stocks_performance(pe_file_date, end_date=None, pe_lower_bound=None, pe_upper_bound=None,
                                    pb_lower_bound=None, pb_upper_bound=None, dsi_lower_bound=None, dsi_upper_bound=None,
                                    roe_lower_bound=None, roe_upper_bound=None, 
                                    core_roe_lower_bound=None, core_roe_upper_bound=None,
                                    market_cap_threshold=200):
    """
    分析指定PE和PB范围内的股票历史收益
    
    参数:
    pe_file_date (str): PE数据文件的日期，格式为'YYYYMMDD'
    pe_lower_bound (float): PE下界，可选
    pe_upper_bound (float): PE上界，可选
    pb_lower_bound (float): PB下界，可选
    pb_upper_bound (float): PB上界，可选
    dsi_lower_bound (float): 存货周转天数下界，可选
    dsi_upper_bound (float): 存货周转天数上界，可选
    roe_lower_bound (float): ROE下界，可选
    roe_upper_bound (float): ROE上界，可选
    core_roe_lower_bound (float): 扣非ROE下界，可选
    core_roe_upper_bound (float): 扣非ROE上界，可选
    market_cap_threshold (float): 市值阈值（单位：亿元），默认200亿
    end_date (str): 计算收益的结束日期，格式为'YYYYMMDD'，默认为None（使用当前日期）
    
    返回:
    tuple: (positive_df, negative_df) 包含正收益和负收益股票的DataFrame
    """
    try:
        # 读取PE数据
        pe_file = os.path.join(DATA_ROOT, 'pe_data', f'pe_{pe_file_date}.csv')
        if not os.path.exists(pe_file):
            print(f"未找到PE数据文件: {pe_file}")
            print("开始计算PE和PB值...")
            pe_df = get_all_stocks_pe(pe_file_date, max_workers=50)  # 增加线程数
            if pe_df is None:
                print("计算PE和PB值失败")
                return None, None
            # 保存PE数据
            os.makedirs(os.path.dirname(pe_file), exist_ok=True)
            pe_df.to_csv(pe_file, index=False, encoding='utf-8-sig')
            print(f"PE和PB数据已保存到: {pe_file}")
        else:
            pe_df = pd.read_csv(pe_file)
        
        # 确保股票代码是6位字符串格式
        pe_df['stock_code'] = pe_df['stock_code'].astype(str).str.zfill(6)
        
        # 从保存的文件中读取市值数据
        print("读取市值数据...")
        stock_info = pd.read_csv(os.path.join(DATA_ROOT, 'all_stocks.csv'))
        stock_info['stock_code'] = stock_info['stock_code'].astype(str).str.zfill(6)
        
        # 合并PE数据和市值数据，使用suffixes参数避免列名冲突
        pe_df = pd.merge(pe_df, stock_info[['stock_code', 'market_cap']], on='stock_code', how='left', suffixes=('', '_new'))
        
        # 如果存在重复的market_cap列，保留一个
        if 'market_cap_new' in pe_df.columns:
            pe_df['market_cap'] = pe_df['market_cap_new']
            pe_df = pe_df.drop('market_cap_new', axis=1)
        
        # 筛选市值大于阈值的股票
        pe_df = pe_df[pe_df['market_cap'] >= market_cap_threshold * 100000000]  # 转换为元
        print(f"筛选出 {len(pe_df)} 只市值大于 {market_cap_threshold} 亿的股票")
        
        # 根据PE范围筛选股票
        if pe_lower_bound is not None:
            pe_df = pe_df[pe_df['pe'] >= pe_lower_bound]
        if pe_upper_bound is not None:
            pe_df = pe_df[pe_df['pe'] <= pe_upper_bound]
            
        # 根据PB范围筛选股票
        if pb_lower_bound is not None:
            pe_df = pe_df[pe_df['pb'] >= pb_lower_bound]
        if pb_upper_bound is not None:
            pe_df = pe_df[pe_df['pb'] <= pb_upper_bound]

        # 筛选出DSI符合要求的股票
        if dsi_lower_bound is not None:
            pe_df = pe_df[pe_df['dsi'] >= dsi_lower_bound]
        if dsi_upper_bound is not None:
            pe_df = pe_df[pe_df['dsi'] <= dsi_upper_bound]

        # 筛选出ROE符合要求的股票
        if roe_lower_bound is not None:
            pe_df = pe_df[pe_df['roe'] >= roe_lower_bound]
        if roe_upper_bound is not None:
            pe_df = pe_df[pe_df['roe'] <= roe_upper_bound]

        # 筛选出扣非ROE符合要求的股票
        if core_roe_lower_bound is not None:
            pe_df = pe_df[pe_df['core_roe'] >= core_roe_lower_bound]
        if core_roe_upper_bound is not None:
            pe_df = pe_df[pe_df['core_roe'] <= core_roe_upper_bound]
            
        if pe_df.empty:
            print(f"未找到符合条件的股票")
            return None, None
            
        print(f"\n找到 {len(pe_df)} 只符合条件的股票")
        print(f"PE范围: [{pe_lower_bound}, {pe_upper_bound}]")
        print(f"PB范围: [{pb_lower_bound}, {pb_upper_bound}]")
        print(f"DSI范围: [{dsi_lower_bound}, {dsi_upper_bound}]")
        print(f"ROE范围: [{roe_lower_bound}, {roe_upper_bound}]")
        print(f"扣非ROE范围: [{core_roe_lower_bound}, {core_roe_upper_bound}]")
        print(f"市值范围: > {market_cap_threshold} 亿")
        
        # 设置结束日期
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        print(f"计算区间: {pe_file_date} 至 {end_date}")
        
        # 创建结果列表
        results = []
        
        def process_stock(row):
            try:
                stock_code = row['stock_code']
                stock_name = row['stock_name']
                pe = row['pe']
                pb = row['pb']
                dsi = row['dsi']
                roe = row['roe']
                core_roe = row['core_roe']
                market_cap = row['market_cap'] / 100000000  # 转换为亿元
                
                # 从本地文件获取历史数据
                hist_data = load_stock_history(stock_code, pe_file_date, end_date)
                if hist_data is None or hist_data.empty:
                    print(f"未找到股票 {stock_code} 的历史数据")
                    return None
                    
                # 计算收益率
                start_price = hist_data['收盘'].iloc[0]
                end_price = hist_data['收盘'].iloc[-1]
                return_rate = (end_price - start_price) / start_price * 100
                
                return {
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'pe': pe,
                    'pb': pb,
                    'dsi': dsi,
                    'roe': roe,
                    'core_roe': core_roe,
                    'market_cap': market_cap,
                    'return_rate': return_rate,
                    'start_price': start_price,
                    'end_price': end_price,
                    'days': len(hist_data)
                }
                
            except Exception as e:
                print(f"处理股票 {stock_code} 时发生错误: {str(e)}")
                return None
        
        # 使用线程池并行处理
        print("\n开始计算历史收益...")
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:  # 增加线程数
            future_to_stock = {
                executor.submit(process_stock, row): row['stock_code']
                for _, row in pe_df.iterrows()
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_stock):
                stock_code = future_to_stock[future]
                completed += 1
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    if completed % 10 == 0:
                        print(f"已处理 {completed}/{len(pe_df)} 只股票...")
                except Exception as e:
                    print(f"处理股票 {stock_code} 时发生错误: {str(e)}")
        
        # 创建结果DataFrame
        if results:
            result_df = pd.DataFrame(results)
            
            # 分离正收益和负收益股票
            positive_df = result_df[result_df['return_rate'] > 0].copy()
            negative_df = result_df[result_df['return_rate'] <= 0].copy()
            
            # 分别排序
            positive_df = positive_df.sort_values('return_rate', ascending=False)
            negative_df = negative_df.sort_values('return_rate', ascending=True)
            
            # 计算统计信息
            print("\n分析结果:")
            print(f"正收益股票数量: {len(positive_df)}")
            print(f"负收益股票数量: {len(negative_df)}")
            print(f"\n正收益股票统计:")
            print(f"平均收益率: {positive_df['return_rate'].mean():.2f}%")
            print(f"最大收益率: {positive_df['return_rate'].max():.2f}%")
            print(f"最小收益率: {positive_df['return_rate'].min():.2f}%")
            print(f"收益率中位数: {positive_df['return_rate'].median():.2f}%")
            print(f"\n负收益股票统计:")
            print(f"平均收益率: {negative_df['return_rate'].mean():.2f}%")
            print(f"最大亏损率: {negative_df['return_rate'].max():.2f}%")
            print(f"最小亏损率: {negative_df['return_rate'].min():.2f}%")
            print(f"亏损率中位数: {negative_df['return_rate'].median():.2f}%")
            
            # 保存结果
            positive_file = os.path.join(DATA_ROOT, 'pe_data', f'pe_positive_{pe_file_date}.csv')
            negative_file = os.path.join(DATA_ROOT, 'pe_data', f'pe_negative_{pe_file_date}.csv')
            positive_df.to_csv(positive_file, index=False, encoding='utf-8-sig')
            negative_df.to_csv(negative_file, index=False, encoding='utf-8-sig')
            print(f"\n分析结果已保存到:")
            print(f"正收益股票: {positive_file}")
            print(f"负收益股票: {negative_file}")
            
            return positive_df, negative_df
        else:
            print("未获取到任何有效的收益数据")
            return None, None
            
    except Exception as e:
        print(f"分析股票时发生错误: {str(e)}")
        return None, None

def analyze_multiple_stocks_price_and_pe(stock_codes, start_date, end_date):
    """
    批量分析多只股票的价格和PE关系
    
    参数:
    stock_codes (list): 股票代码列表
    start_date (str): 开始日期，格式为'YYYYMMDD'
    end_date (str): 结束日期，格式为'YYYYMMDD'
    """
    # 1. 一次性加载所有股票的财务数据
    print("加载财务数据...")
    all_financial_data = load_financial_data()  # 不传入stock_code，加载所有数据
    if all_financial_data is None:
        print("加载财务数据失败")
        return
    
    # 2. 分析每只股票
    for stock_code in stock_codes:
        try:
            # 从所有财务数据中筛选出当前股票的财务数据
            stock_financial_data = all_financial_data[all_financial_data['stock_code'] == stock_code].copy()
            if not stock_financial_data.empty:
                stock_financial_data.set_index('报告期', inplace=True)
                analyze_stock_price_and_pe(stock_code, start_date, end_date, stock_financial_data)
            else:
                print(f"未找到股票 {stock_code} 的财务数据")
        except Exception as e:
            print(f"分析股票 {stock_code} 时发生错误: {str(e)}")

# 根据策略  进行选股
def smartSelect():
    # ** 分析指定PE范围内的股票历史收益**
    pe_file_date = "20100324"
    end_date = "20200326"
    positive_df, negative_df = analyze_low_pe_stocks_performance(
        pe_file_date,
        pe_lower_bound=0,  # PE下界
        pe_upper_bound=20,  # PE上界
        pb_lower_bound=0.5,  # PB下界
        pb_upper_bound=2.0,  # PB上界
        dsi_lower_bound=0,  # DSI下界
        dsi_upper_bound=100,  # DSI上界
        core_roe_lower_bound=0.2,
        core_roe_upper_bound=1,
        end_date=end_date
    )

    if positive_df is not None and negative_df is not None:
        print("收益率>0的股票数量："+str(positive_df.shape[0])+"\n收益率最高的10只股票:")
        print(positive_df.head(10))
        print("收益率<0的股票数量："+str(negative_df.shape[0])+"\n收益率最低的10只股票:")
        print(negative_df.head(10))

def backtest_stock(stock_code, start_date, end_date):
    """
    对单只股票进行回测，计算收益率、最大回撤并绘制图表
    
    参数:
    stock_code (str): 股票代码
    start_date (str): 开始日期，格式为'YYYYMMDD'
    end_date (str): 结束日期，格式为'YYYYMMDD'
    
    返回:
    tuple: (收益率, 最大回撤, 年化收益率)
    """
    # 1. 从本地文件获取股票每日收盘价
    stock_price_df = load_stock_history(stock_code, start_date, end_date)
    if stock_price_df is None or stock_price_df.empty:
        print(f"未找到股票 {stock_code} 的历史数据")
        return None
    
    # 确保日期为datetime格式
    stock_price_df['日期'] = pd.to_datetime(stock_price_df['日期'])
    
    # 计算收益率和累计收益率
    initial_price = stock_price_df['收盘'].iloc[0]
    stock_price_df['收益率'] = stock_price_df['收盘'] / initial_price - 1
    
    # 计算最大回撤
    stock_price_df['累计最大值'] = stock_price_df['收盘'].cummax()
    stock_price_df['回撤'] = (stock_price_df['收盘'] - stock_price_df['累计最大值']) / stock_price_df['累计最大值']
    max_drawdown = stock_price_df['回撤'].min() * 100
    
    # 计算总收益率
    total_return = stock_price_df['收益率'].iloc[-1] * 100
    
    # 计算年化收益率
    days = (stock_price_df['日期'].iloc[-1] - stock_price_df['日期'].iloc[0]).days
    annualized_return = (1 + total_return/100) ** (365.0/days) - 1
    annualized_return *= 100
    
    # 2. 绘制股价散点图
    plt.figure(figsize=(14, 8))
    plt.scatter(stock_price_df['日期'], stock_price_df['收盘'], s=10, color='#1f77b4', alpha=0.7)
    plt.plot(stock_price_df['日期'], stock_price_df['收盘'], color='#1f77b4', alpha=0.5)
    
    # 标记起点和终点
    plt.scatter(stock_price_df['日期'].iloc[0], stock_price_df['收盘'].iloc[0], 
                color='green', s=100, label='起始点', zorder=5, edgecolors='black')
    plt.scatter(stock_price_df['日期'].iloc[-1], stock_price_df['收盘'].iloc[-1], 
                color='red', s=100, label='终止点', zorder=5, edgecolors='black')
    
    # 标记最高点和最低点
    max_idx = stock_price_df['收盘'].idxmax() if hasattr(stock_price_df['收盘'], 'idxmax') else stock_price_df['收盘'].values.argmax()
    min_idx = stock_price_df['收盘'].idxmin() if hasattr(stock_price_df['收盘'], 'idxmin') else stock_price_df['收盘'].values.argmin()
    
    # 使用数值索引访问最高点和最低点
    plt.scatter(stock_price_df['日期'].iloc[max_idx], stock_price_df['收盘'].iloc[max_idx], 
                color='yellow', s=100, label='最高点', zorder=5, edgecolors='black')
    plt.scatter(stock_price_df['日期'].iloc[min_idx], stock_price_df['收盘'].iloc[min_idx], 
                color='purple', s=100, label='最低点', zorder=5, edgecolors='black')
    
    # 获取股票名称
    stock_info = pd.read_csv(os.path.join(DATA_ROOT, 'all_stocks.csv'))
    stock_info['stock_code'] = stock_info['stock_code'].astype(str).str.zfill(6)
    stock_name = stock_info[stock_info['stock_code'] == stock_code]['stock_name'].values[0] if not stock_info[stock_info['stock_code'] == stock_code].empty else '未知'
    
    # 设置图表标题和标签
    plt.title(f'{stock_code} {stock_name} 股价走势 ({start_date} - {end_date})', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('价格(元)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 添加文本标注：总收益率和最大回撤
    text_info = (
        f"总收益率: {total_return:.2f}%\n"
        f"年化收益率: {annualized_return:.2f}%\n"
        f"最大回撤: {max_drawdown:.2f}%\n"
        f"起始价格: {stock_price_df['收盘'].iloc[0]:.2f}元\n"
        f"结束价格: {stock_price_df['收盘'].iloc[-1]:.2f}元"
    )
    
    plt.figtext(0.15, 0.15, text_info, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'), fontsize=12)
    
    # 格式化x轴日期
    plt.gcf().autofmt_xdate()
    
    # 显示图表
    plt.tight_layout()
    plt.show()
    
    # 3. 打印收益和回撤数据
    print(f"\n--- {stock_code} {stock_name} 回测结果 ---")
    print(f"回测区间: {start_date} 至 {end_date}")
    print(f"交易天数: {len(stock_price_df)}天")
    print(f"开始价格: {stock_price_df['收盘'].iloc[0]:.2f}元")
    print(f"结束价格: {stock_price_df['收盘'].iloc[-1]:.2f}元")
    print(f"最高价格: {stock_price_df['收盘'].max():.2f}元")
    print(f"最低价格: {stock_price_df['收盘'].min():.2f}元")
    print(f"总收益率: {total_return:.2f}%")
    print(f"年化收益率: {annualized_return:.2f}%")
    print(f"最大回撤: {max_drawdown:.2f}%")
    
    return total_return, max_drawdown, annualized_return

# 使用示例
if __name__ == "__main__":
    # 1.获取所有股票代码和当前市值
    #get_and_save_stock_codes()

    # # 首先下载所有股票的财务数据（只需要运行一次）
    # print("开始下载财务数据...")
    # download_all_financial_data(max_workers=3)

    # # 然后再获取PE值
    # target_date = "20050105"
    # result_df = get_all_stocks_pe(target_date, max_workers=5)
    #
    # if result_df is not None:
    #     print("\nPE值最低的10只股票:")
    #     print(result_df.head(10))
    # # 保存pe数据
    # # 与target_date联动
    # result_df.to_csv(f'stock_pe_{target_date}.csv', index=False, encoding='utf-8-sig')

    # # 分析多只股票的价格和PE关系
    # stock_codes = ["000001", "600000", "600036"]  # 示例股票代码
    # start_date = "20200101"
    # end_date = "20231231"
    # analyze_multiple_stocks_price_and_pe(stock_codes, start_date, end_date)

    #smartSelect()

    backtest_stock("000001", "20150104","20230104")
    
    # 回测示例
    # backtest_stock("600519", "20100101", "20231231")

    # 下载所有股票的历史数据
    #start_date = "20000101"  # 从2000年开始
    # download_all_stocks_history(start_date, max_workers=5)
    #download_all_stocks_history(start_date, max_workers=3,adjust_type=None)
    
    # # 加载单个股票的历史数据示例
    # stock_code = "000001"
    # hist_data = load_stock_history(stock_code)
    # if hist_data is not None:
    #     print(f"\n股票 {stock_code} 的历史数据:")
    #     print(hist_data.head())




            


