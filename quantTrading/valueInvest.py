import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
plt.switch_backend('TkAgg')  # 可以尝试其他后端

import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import concurrent.futures
import time
from datetime import datetime

# 设置中文字体
try:
    # 对于MacOS
    font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
    plt.rcParams['font.family'] = ['PingFang HK']
except:
    # 对于其他系统
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_all_stock_codes():
    """
    获取所有A股股票代码和名称
    返回: DataFrame，包含股票代码和名称
    """
    try:
        # 获取实时行情数据
        stock_df = ak.stock_zh_a_spot_em()
        
        # 提取需要的列
        result_df = stock_df[['代码', '名称']]
        
        # 重命名列
        result_df.columns = ['stock_code', 'stock_name']
        
        # 确保股票代码是字符串格式
        result_df['stock_code'] = result_df['stock_code'].astype(str).str.zfill(6)
        
        # 按股票代码排序
        result_df = result_df.sort_values('stock_code')
        
        return result_df
    except Exception as e:
        print(f"获取股票代码时发生错误: {str(e)}")
        return None

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

def get_stock_pe_at_date(stock_code, target_date):
    """
    获取指定时间点的股票PE值，使用本地财务数据
    """
    try:
        
        # 获取股票每日收盘价
        stock_price_df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=target_date,
            end_date=target_date
        )
        
        if stock_price_df.empty:
            print(f"未找到股票 {stock_code} 在 {target_date} 的交易数据")
            return None
            
        # 从本地加载财务数据
        financial_data = load_financial_data(stock_code)
        if financial_data is None or financial_data.empty:
            print(f"未找到股票 {stock_code} 的本地财务数据")
            return None
            
        # 处理财务数据
        financial_data['基本每股收益'] = pd.to_numeric(financial_data['基本每股收益'], errors='coerce')
        
        # 只保留年报数据
        annual_reports = financial_data[financial_data['报告期'].month == 12]
        
        if annual_reports.empty:
            print(f"未找到股票 {stock_code} 的年报数据")
            return None
            
        # 获取目标日期之前的最近一期年报数据
        target_date_dt = pd.to_datetime(target_date)
        latest_annual = annual_reports[annual_reports.index <= target_date_dt].iloc[-1]
        
        # 计算PE
        pe = stock_price_df['收盘'].iloc[0] / latest_annual['基本每股收益']
        
        return pe
        
    except Exception as e:
        print(f"获取股票 {stock_code} 在 {target_date} 的PE值时发生错误: {str(e)}")
        return None

def get_and_save_stock_codes():
    """
    获取所有A股股票代码和名称，并保存到CSV文件
    """
    try:
        # 获取所有股票代码和名称
        all_stocks = get_all_stock_codes()

        # 保存到CSV文件
        all_stocks.to_csv('all_stocks.csv', index=False, encoding='utf-8-sig')
        print("\n股票代码已保存到 all_stocks.csv")
        
        return all_stocks
    except Exception as e:
        print(f"获取股票代码时发生错误: {str(e)}")  

def load_stock_codes_from_csv():
    """
    从本地CSV文件加载股票代码数据
    如果文件不存在，则重新获取并保存
    """
    try:
        # 尝试从CSV文件读取
        if os.path.exists('all_stocks.csv'):
            df = pd.read_csv('all_stocks.csv')
            print(f"从本地文件加载了 {len(df)} 只股票数据")
            return df
        else:
            print("本地文件不存在，重新获取股票数据...")
            return get_all_stock_codes()
    except Exception as e:
        print(f"读取本地文件时发生错误: {str(e)}")
        print("重新获取股票数据...")
        return get_all_stock_codes()

def get_all_stocks_pe(target_date, max_workers=10):
    """
    并行获取所有股票的PE值
    
    参数:
    target_date (str): 目标日期，格式为'YYYYMMDD'
    max_workers (int): 最大线程数，默认10
    
    返回:
    DataFrame: 包含股票代码、名称和PE值的DataFrame
    """
    # 从本地文件加载股票代码
    all_stocks = load_stock_codes_from_csv()
    if all_stocks is None:
        return None
    
    # 创建结果列表
    results = []
    
    def process_stock(stock_code):
        """处理单个股票的函数"""
        try:
            stock_code = str(stock_code).zfill(6)
            pe_value = get_stock_pe_at_date(stock_code, target_date)
            if pe_value is not None:
                return {
                    'stock_code': stock_code,
                    'pe': pe_value
                }
        except Exception as e:
            print(f"处理股票 {stock_code} 时发生错误: {str(e)}")
        return None
    
    # 使用线程池并行处理
    print(f"\n开始获取 {len(all_stocks)} 只股票的PE值...")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_stock = {
            executor.submit(process_stock, row['stock_code']): row['stock_code']
            for _, row in all_stocks.iterrows()
        }
        
        # 处理完成的任务
        completed = 0
        for future in concurrent.futures.as_completed(future_to_stock):
            stock_code = future_to_stock[future]
            completed += 1
            try:
                result = future.result()
                if result:
                    results.append(result)
                # 打印进度
                if completed % 50 == 0:  # 每处理50只股票打印一次进度
                    print(f"已处理 {completed}/{len(all_stocks)} 只股票...")
            except Exception as e:
                print(f"处理股票 {stock_code} 时发生错误: {str(e)}")
    
    # 计算耗时
    end_time = time.time()
    print(f"\n处理完成！总耗时: {end_time - start_time:.2f} 秒")
    
    # 创建结果DataFrame
    if results:
        result_df = pd.DataFrame(results)
        # 合并原始股票信息
        result_df = pd.merge(all_stocks, result_df, on='stock_code', how='left')
        # 按PE值排序
        result_df = result_df.sort_values('pe')
        
        # 保存结果到CSV
        filename = f'stock_pe_{target_date}.csv'
        result_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到 {filename}")
        
        return result_df
    else:
        print("未获取到任何有效的PE值")
        return None

def save_financial_data_batch(stock_codes, max_workers=5):
    """
    批量获取并保存所有股票的财务数据到单个文件
    """
    try:
        # 确保存储目录存在
        if not os.path.exists('financial_data'):
            os.makedirs('financial_data')
            
        filepath = 'financial_data/all_stocks_financial.csv'
        
        # 如果文件已存在且不超过7天，直接返回
        if os.path.exists(filepath):
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if (datetime.now() - file_time).days < 7:
                print("使用缓存的财务数据（未超过7天）")
                return True
        
        all_financial_data = []
        
        def process_stock(stock_code):
            try:
                stock_code = str(stock_code).zfill(6)
                time.sleep(0.5)  # 添加延时避免被封
                financial_data = ak.stock_financial_abstract_ths(
                    symbol=stock_code,
                    indicator="按年度"
                )
                
                if financial_data is not None and not financial_data.empty:
                    # 添加股票代码列
                    financial_data['stock_code'] = stock_code
                    return financial_data
                    
            except Exception as e:
                print(f"获取股票 {stock_code} 财务数据时发生错误: {str(e)}")
            return None
        
        print(f"\n开始下载 {len(stock_codes)} 只股票的财务数据...")
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_stock = {
                executor.submit(process_stock, code): code
                for code in stock_codes
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_stock):
                stock_code = future_to_stock[future]
                completed += 1
                try:
                    df = future.result()
                    if df is not None:
                        all_financial_data.append(df)
                    if completed % 10 == 0:
                        print(f"已处理 {completed}/{len(stock_codes)} 只股票...")
                except Exception as e:
                    print(f"处理股票 {stock_code} 时发生错误: {str(e)}")
        
        # 合并所有数据
        if all_financial_data:
            combined_df = pd.concat(all_financial_data, ignore_index=True)
            # 确保股票代码格式一致
            combined_df['stock_code'] = combined_df['stock_code'].astype(str).str.zfill(6)
            # 保存到单个CSV文件
            combined_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"\n已保存所有股票的财务数据到 {filepath}")
            
            end_time = time.time()
            print(f"下载完成！总耗时: {end_time - start_time:.2f} 秒")
            return True
            
        return False
            
    except Exception as e:
        print(f"保存财务数据时发生错误: {str(e)}")
        return False

def load_financial_data(stock_code=None):
    """
    从本地加载股票财务数据
    如果不指定stock_code，则返回所有数据
    """
    try:
        filepath = 'financial_data/all_stocks_financial.csv'
        if os.path.exists(filepath):
            all_financial_data = pd.read_csv(filepath)
            
            # 确保股票代码为字符串格式，并补齐6位
            all_financial_data['stock_code'] = all_financial_data['stock_code'].astype(str).str.zfill(6)
            all_financial_data['报告期'] = pd.to_datetime(all_financial_data['报告期'])
            
            if stock_code:
                # 确保输入的stock_code也是6位字符串格式
                stock_code = str(stock_code).zfill(6)
                # 返回指定股票的数据
                stock_data = all_financial_data[all_financial_data['stock_code'] == stock_code].copy()
                stock_data.set_index('报告期', inplace=True)
                return stock_data
            else:
                # 返回所有数据
                return all_financial_data
        return None
    except Exception as e:
        print(f"加载财务数据时发生错误: {str(e)}")
        return None

def download_all_financial_data(max_workers=5):
    """
    下载所有股票的财务数据
    """
    all_stocks = load_stock_codes_from_csv()
    if all_stocks is None:
        return False
    
    stock_codes = all_stocks['stock_code'].tolist()
    return save_financial_data_batch(stock_codes, max_workers)

# 使用示例
if __name__ == "__main__":
    # 首先下载所有股票的财务数据（只需要运行一次）
    print("开始下载财务数据...")
    download_all_financial_data(max_workers=3)  # 使用较小的并发数
    
    # 然后再获取PE值
    target_date = "20240301"
    result_df = get_all_stocks_pe(target_date, max_workers=5)
    
    if result_df is not None:
        print("\nPE值最低的10只股票:")
        print(result_df.head(10))
