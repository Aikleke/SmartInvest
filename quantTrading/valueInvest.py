import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.switch_backend('TkAgg')  # 可以尝试其他后端

import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# 设置中文字体
try:
    # 对于MacOS
    font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
    plt.rcParams['font.family'] = ['PingFang HK']
except:
    # 对于其他系统
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_stock_price_and_pe(stock_code, start_date, end_date):
    """
    分析股票价格和PE的关系
    """
    # 1. 获取股票每日收盘价
    stock_price_df = ak.stock_zh_a_hist(
        symbol=stock_code,
        period="daily",
        start_date=start_date,
        end_date=end_date
    )

    # 2. 获取每股收益数据
    financial_data = ak.stock_financial_abstract_ths(
        symbol=stock_code,
        indicator="按报告期"
    )

    # 3. 数据处理
    # 处理日期格式
    stock_price_df['日期'] = pd.to_datetime(stock_price_df['日期'])
    stock_price_df.set_index('日期', inplace=True)
    
    # 处理财务数据 - 修改这部分
    financial_data['基本每股收益'] = pd.to_numeric(financial_data['基本每股收益'], errors='coerce')
    financial_data.index = pd.to_datetime(financial_data['报告期'])  # 确保日期格式正确
    
    # 确保日期索引格式化正确
    stock_price_df.index = pd.to_datetime(stock_price_df.index).date
    financial_data.index = pd.to_datetime(financial_data.index).date

    # 添加调试信息
    print("股价数据日期范围:", stock_price_df.index.min(), "至", stock_price_df.index.max())
    print("财务数据日期范围:", financial_data.index.min(), "至", financial_data.index.max())
    
    # 检查日期格式
    print("股价数据日期类型:", type(stock_price_df.index[0]))
    print("财务数据日期类型:", type(financial_data.index[0]))

    # 将每股收益数据填充到每个交易日
    eps_daily = financial_data['基本每股收益'].reindex(stock_price_df.index, method='ffill')

    # 4. 计算PE
    pe_daily = stock_price_df['收盘'] / eps_daily

    # 5. 创建可视化图表
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))

    # 绘制股价走势
    ax1.plot(stock_price_df.index, stock_price_df['收盘'], color='blue')
    ax1.set_title(f'{stock_code} 股价走势')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('股价(元)')
    # 设置x轴日期格式
    ax1.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    ax1.grid(True)

    # 绘制PE走势
    ax2.plot(pe_daily.index, pe_daily, color='red')
    ax2.set_title(f'{stock_code} PE走势')
    ax2.set_xlabel('日期')
    ax2.set_ylabel('PE比率')
    ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    ax2.grid(True)

    # 绘制股价和PE的散点图
    ax3.scatter(stock_price_df['收盘'], pe_daily, alpha=0.5)
    ax3.set_title('股价与PE的关系散点图')
    ax3.set_xlabel('股价(元)')
    ax3.set_ylabel('PE比率')
    ax3.grid(True)

    # 添加趋势线
    z = np.polyfit(stock_price_df['收盘'], pe_daily, 1)
    p = np.poly1d(z)
    ax3.plot(stock_price_df['收盘'], p(stock_price_df['收盘']), "r--", alpha=0.8)

    plt.tight_layout()
    plt.show()

    # 6. 输出相关系数
    correlation = stock_price_df['收盘'].corr(pe_daily)
    print(f"\n股价与PE的相关系数: {correlation:.4f}")

    return stock_price_df['收盘'], pe_daily

# 使用示例
stock_code = "601288"  # 可以更换为其他股票代码
start_date = "20200101"
end_date = "20250528"

prices, pes = analyze_stock_price_and_pe(stock_code, start_date, end_date)
