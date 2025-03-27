import akshare as ak
import pandas as pd
import os
import concurrent.futures
import time
from datetime import datetime
from pathlib import Path

# 定义数据根目录
DATA_ROOT = os.path.expanduser('~/investData')

# 确保数据根目录存在
os.makedirs(DATA_ROOT, exist_ok=True)

def get_all_stock_codes():
    """
    获取所有A股股票代码和名称
    
    返回:
    DataFrame: 包含股票代码和名称的DataFrame
    """
    try:
        # 获取A股股票列表
        stock_info = ak.stock_zh_a_spot_em()
        
        # 只保留需要的列
        stock_info = stock_info[['代码', '名称', '总市值']]
        
        # 重命名列
        stock_info.columns = ['stock_code', 'stock_name', 'market_cap']
        
        # 确保股票代码是6位字符串格式
        stock_info['stock_code'] = stock_info['stock_code'].astype(str).str.zfill(6)
        
        return stock_info
        
    except Exception as e:
        print(f"获取股票代码时发生错误: {str(e)}")
        return None

def get_and_save_stock_codes():
    """
    获取所有A股股票代码和名称，并保存到CSV文件
    
    返回:
    DataFrame: 包含股票代码和名称的DataFrame
    """
    try:
        # 获取股票代码和名称
        stock_info = get_all_stock_codes()
        if stock_info is None:
            return None
            
        # 保存到CSV文件
        filepath = os.path.join(DATA_ROOT, 'all_stocks.csv')
        stock_info.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"股票代码和市值数据已保存到 {filepath}")
        
        return stock_info
        
    except Exception as e:
        print(f"保存股票代码时发生错误: {str(e)}")
        return None

def load_stock_codes_from_csv():
    """
    从本地CSV文件加载股票代码数据
    如果文件不存在，则重新获取并保存
    """
    try:
        filepath = os.path.join(DATA_ROOT, 'all_stocks.csv')
        # 尝试从CSV文件读取
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"从本地文件加载了 {len(df)} 只股票数据")
            return df
        else:
            print("本地文件不存在，重新获取股票数据...")
            return get_all_stock_codes()
    except Exception as e:
        print(f"读取本地文件时发生错误: {str(e)}")
        print("重新获取股票数据...")
        return get_all_stock_codes()

def get_stock_pe_at_date(stock_code, target_date):
    """
    获取指定时间点的股票PE值，使用本地财务数据
    """
    try:
        # 从本地文件获取股票每日收盘价（使用不复权数据计算PE）
        stock_price_df = load_stock_history(stock_code, target_date, target_date, adjust_type=None)
        if stock_price_df is None or stock_price_df.empty:
            print(f"未找到股票 {stock_code} 在 {target_date} 的交易数据")
            return None
            
        # 从本地加载财务数据
        financial_data = load_financial_data(stock_code)
        if financial_data is None or financial_data.empty:
            print(f"未找到股票 {stock_code} 的本地财务数据")
            return None
            
        # 处理财务数据
        financial_data['基本每股收益'] = pd.to_numeric(financial_data['基本每股收益'], errors='coerce')
        
        # 获取目标日期的年份
        target_year = pd.to_datetime(target_date).year
        
        # 获取目标日期之前的最近一期年报数据
        valid_reports = financial_data[financial_data.index.year < target_year]
        
        if valid_reports.empty:
            print(f"未找到股票 {stock_code} 在 {target_year} 年之前的年报数据")
            return None
            
        latest_annual = valid_reports.iloc[-1]
        
        # 检查每股收益是否有效
        if pd.isna(latest_annual['基本每股收益']) or latest_annual['基本每股收益'] == 0:
            print(f"股票 {stock_code} 的每股收益数据无效")
            return None
        
        # 计算PE
        pe = stock_price_df['收盘'].iloc[0] / latest_annual['基本每股收益']
        
        # 检查PE值是否合理
        if pd.isna(pe) or pe < 0 or pe > 1000000:  # 设置一个合理的PE值范围
            print(f"股票 {stock_code} 的PE值异常: {pe}")
            return None
            
        return pe
        
    except Exception as e:
        print(f"获取股票 {stock_code} 在 {target_date} 的PE值时发生错误: {str(e)}")
        return None

def get_all_stocks_pe(target_date, max_workers=50):
    """
    并行获取所有股票的PE值
    
    参数:
    target_date (str): 目标日期，格式为'YYYYMMDD'
    max_workers (int): 最大线程数，默认50
    
    返回:
    DataFrame: 包含股票代码、名称和PE值的DataFrame
    """
    # 从本地文件加载股票代码
    all_stocks = load_stock_codes_from_csv()
    if all_stocks is None:
        return None
    
    # 确保stock_code列是字符串类型
    all_stocks['stock_code'] = all_stocks['stock_code'].astype(str).str.zfill(6)
    
    # 一次性加载所有财务数据
    print("加载所有股票的财务数据...")
    all_financial_data = load_financial_data()
    if all_financial_data is None:
        print("加载财务数据失败")
        return None
    
    # 创建结果列表
    results = []
    
    def process_stock(stock_code):
        """处理单个股票的函数"""
        try:
            stock_code = str(stock_code).zfill(6)
            
            # 从本地文件获取股票每日收盘价（使用不复权数据计算PE）
            stock_price_df = load_stock_history(stock_code, target_date, target_date, adjust_type=None)
            if stock_price_df is None or stock_price_df.empty:
                print(f"未找到股票 {stock_code} 在 {target_date} 的交易数据")
                return None
                
            # 从已加载的财务数据中筛选当前股票的数据
            stock_financial_data = all_financial_data[all_financial_data['stock_code'] == stock_code].copy()
            if stock_financial_data.empty:
                print(f"未找到股票 {stock_code} 的财务数据")
                return None
                
            # 处理财务数据
            stock_financial_data['基本每股收益'] = pd.to_numeric(stock_financial_data['基本每股收益'], errors='coerce')
            
            # 获取目标日期的年份
            target_year = pd.to_datetime(target_date).year
            
            # 获取目标日期之前的最近一期年报数据
            valid_reports = stock_financial_data[pd.to_datetime(stock_financial_data['报告期']).dt.year < target_year]
            
            if valid_reports.empty:
                print(f"未找到股票 {stock_code} 在 {target_year} 年之前的年报数据")
                return None
                
            latest_annual = valid_reports.iloc[-1]
            
            # 检查每股收益是否有效
            if pd.isna(latest_annual['基本每股收益']) or latest_annual['基本每股收益'] == 0:
                print(f"股票 {stock_code} 的每股收益数据无效")
                return None
            
            # 计算PE
            pe = stock_price_df['收盘'].iloc[0] / latest_annual['基本每股收益']
            
            # 检查PE值是否合理
            if pd.isna(pe) or pe < 0 or pe > 1000:  # 设置一个合理的PE值范围
                print(f"股票 {stock_code} 的PE值异常: {pe}")
                return None
                
            return {
                'stock_code': stock_code,
                'pe': pe
            }
            
        except Exception as e:
            print(f"处理股票 {stock_code} 时发生错误: {str(e)}")
            return None
    
    # 使用线程池并行处理
    print(f"\n开始获取 {len(all_stocks)} 只股票的PE值...")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_stock = {
            executor.submit(process_stock, row['stock_code']): row['stock_code']
            for _, row in all_stocks.iterrows()
        }
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_stock):
            stock_code = future_to_stock[future]
            completed += 1
            try:
                result = future.result()
                if result:
                    results.append(result)
                if completed % 50 == 0:
                    print(f"已处理 {completed}/{len(all_stocks)} 只股票...")
            except Exception as e:
                print(f"处理股票 {stock_code} 时发生错误: {str(e)}")
    
    end_time = time.time()
    print(f"\n处理完成！总耗时: {end_time - start_time:.2f} 秒")
    
    # 创建结果DataFrame
    if results:
        result_df = pd.DataFrame(results)
        
        # 确保两个DataFrame的stock_code列都是字符串类型
        result_df['stock_code'] = result_df['stock_code'].astype(str).str.zfill(6)
        all_stocks['stock_code'] = all_stocks['stock_code'].astype(str).str.zfill(6)
        
        # 合并数据
        result_df = pd.merge(all_stocks, result_df, on='stock_code', how='left')
        
        # 添加日期列
        result_df['date'] = target_date
        
        # 按PE值排序
        result_df = result_df.sort_values('pe')
        
        # 保存结果到CSV
        filename = os.path.join(DATA_ROOT, f'pe_data/pe_{target_date}.csv')
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 保存数据
        result_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到 {filename}")
        
        # 同时保存一个最新数据的副本
        latest_filepath = os.path.join(DATA_ROOT, 'pe_data/pe_latest.csv')
        result_df.to_csv(latest_filepath, index=False, encoding='utf-8-sig')
        
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
        financial_dir = os.path.join(DATA_ROOT, 'financial_data')
        if not os.path.exists(financial_dir):
            os.makedirs(financial_dir)
            
        filepath = os.path.join(financial_dir, 'all_stocks_financial.csv')
        
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
        filepath = os.path.join(DATA_ROOT, 'financial_data', 'all_stocks_financial.csv')
        if os.path.exists(filepath):
            all_financial_data = pd.read_csv(filepath)
            
            # 确保股票代码为字符串格式，并补齐6位
            all_financial_data['stock_code'] = all_financial_data['stock_code'].astype(str).str.zfill(6)
            
            # 处理报告期，确保是年份格式
            all_financial_data['报告期'] = pd.to_datetime(all_financial_data['报告期'].astype(str) + '-12-31')
            
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

def download_all_stocks_history(start_date, end_date=None, max_workers=5, retry_times=3, adjust_type='hfq'):
    """
    下载所有股票的历史数据并保存到单独的文件中
    
    参数:
    start_date (str): 开始日期，格式为'YYYYMMDD'
    end_date (str): 结束日期，格式为'YYYYMMDD'，默认为当前日期
    max_workers (int): 最大线程数，默认5
    retry_times (int): 失败重试次数，默认3次
    adjust_type (str): 复权类型，可选值：
        - 'hfq': 后复权（默认）
        - 'qfq': 前复权
        - None: 不复权
    
    返回:
    bool: 是否成功完成下载
    """
    try:
        # 如果没有指定结束日期，使用当前日期
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
            
        # 根据复权类型选择存储目录
        adjust_map = {
            'hfq': 'stock_history_hfq',
            'qfq': 'stock_history_qfq',
            None: 'stock_history'
        }
        history_dir = os.path.join(DATA_ROOT, adjust_map.get(adjust_type, 'stock_history'))
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
            
        # 创建临时目录用于存储下载进度
        temp_dir = os.path.join(history_dir, 'temp')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # 获取所有股票代码
        all_stocks = load_stock_codes_from_csv()
        if all_stocks is None:
            return False
            
        # 确保股票代码是字符串格式
        all_stocks['stock_code'] = all_stocks['stock_code'].astype(str).str.zfill(6)
        
        # 加载下载进度
        progress_file = os.path.join(temp_dir, 'download_progress.csv')
        if os.path.exists(progress_file):
            progress_df = pd.read_csv(progress_file)
            completed_stocks = set(progress_df['stock_code'].astype(str).str.zfill(6))
            print(f"发现已下载 {len(completed_stocks)} 只股票的数据")
        else:
            completed_stocks = set()
            progress_df = pd.DataFrame(columns=['stock_code', 'status', 'last_update'])
        
        def process_stock(stock_code):
            try:
                stock_code = str(stock_code).zfill(6)
                
                # 检查是否已经成功下载
                if stock_code in completed_stocks:
                    return True
                
                filepath = os.path.join(history_dir, f'{stock_code}.csv')
                temp_filepath = os.path.join(temp_dir, f'{stock_code}.csv')
                
                # 尝试下载数据
                for attempt in range(retry_times):
                    try:
                        # 准备API调用参数
                        api_params = {
                            'symbol': stock_code,
                            'period': "daily",
                            'start_date': start_date,
                            'end_date': end_date
                        }
                        
                        # 只有当adjust_type不为None时才添加adjust参数
                        if adjust_type is not None:
                            api_params['adjust'] = adjust_type
                        
                        # 获取历史数据
                        hist_data = ak.stock_zh_a_hist(**api_params)
                        
                        if hist_data is not None and not hist_data.empty:
                            # 数据完整性检查
                            if len(hist_data) < 10:  # 如果数据太少，可能有问题
                                raise Exception("数据量不足")
                            
                            # 保存到临时文件
                            hist_data.to_csv(temp_filepath, index=False, encoding='utf-8-sig')
                            # 移动到正式目录
                            os.rename(temp_filepath, filepath)
                            
                            # 更新进度
                            progress_df.loc[len(progress_df)] = {
                                'stock_code': stock_code,
                                'status': 'success',
                                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            progress_df.to_csv(progress_file, index=False, encoding='utf-8-sig')
                            completed_stocks.add(stock_code)
                            return True
                            
                    except Exception as e:
                        if attempt < retry_times - 1:
                            print(f"股票 {stock_code} 第 {attempt + 1} 次下载失败: {str(e)}")
                            time.sleep(2)  # 失败后等待2秒再重试
                            continue
                        else:
                            print(f"股票 {stock_code} 下载失败: {str(e)}")
                            # 更新进度为失败
                            progress_df.loc[len(progress_df)] = {
                                'stock_code': stock_code,
                                'status': 'failed',
                                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            progress_df.to_csv(progress_file, index=False, encoding='utf-8-sig')
                            return False
                            
            except Exception as e:
                print(f"处理股票 {stock_code} 时发生错误: {str(e)}")
                return False
        
        print(f"\n开始下载 {len(all_stocks)} 只股票的历史数据...")
        print(f"时间范围: {start_date} 至 {end_date}")
        print(f"复权类型: {adjust_type if adjust_type else '不复权'}")
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_stock = {
                executor.submit(process_stock, row['stock_code']): row['stock_code']
                for _, row in all_stocks.iterrows()
            }
            
            completed = 0
            success_count = 0
            for future in concurrent.futures.as_completed(future_to_stock):
                stock_code = future_to_stock[future]
                completed += 1
                try:
                    if future.result():
                        success_count += 1
                    if completed % 10 == 0:
                        print(f"已处理 {completed}/{len(all_stocks)} 只股票...")
                except Exception as e:
                    print(f"处理股票 {stock_code} 时发生错误: {str(e)}")
        
        end_time = time.time()
        print(f"\n下载完成！")
        print(f"总耗时: {end_time - start_time:.2f} 秒")
        print(f"成功下载: {success_count}/{len(all_stocks)} 只股票")
        print(f"数据保存在: {os.path.abspath(history_dir)}")
        
        # 清理临时目录
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"下载历史数据时发生错误: {str(e)}")
        return False

def load_stock_history(stock_code, start_date=None, end_date=None, adjust_type='hfq'):
    """
    从本地文件加载股票历史数据
    
    参数:
    stock_code (str): 股票代码
    start_date (str): 开始日期，格式为'YYYYMMDD'，可选
    end_date (str): 结束日期，格式为'YYYYMMDD'，可选
    adjust_type (str): 复权类型，可选值：
        - 'hfq': 后复权（默认）
        - 'qfq': 前复权
        - None: 不复权
    
    返回:
    DataFrame: 股票历史数据，如果文件不存在则返回None
    """
    try:
        # 确保股票代码是6位字符串格式
        stock_code = str(stock_code).zfill(6)
        
        # 根据复权类型选择存储目录
        adjust_map = {
            'hfq': 'stock_history_hfq',
            'qfq': 'stock_history_qfq',
            None: 'stock_history'
        }
        history_dir = os.path.join(DATA_ROOT, adjust_map.get(adjust_type, 'stock_history'))
        filepath = os.path.join(history_dir, f'{stock_code}.csv')
        
        if not os.path.exists(filepath):
            print(f"未找到股票 {stock_code} 的历史数据文件")
            return None
            
        # 读取数据
        hist_data = pd.read_csv(filepath)
        
        # 转换日期列
        hist_data['日期'] = pd.to_datetime(hist_data['日期'])
        
        # 如果指定了日期范围，进行过滤
        if start_date:
            start_date = pd.to_datetime(start_date)
            hist_data = hist_data[hist_data['日期'] >= start_date]
            
        if end_date:
            end_date = pd.to_datetime(end_date)
            hist_data = hist_data[hist_data['日期'] <= end_date]
            
        return hist_data
        
    except Exception as e:
        print(f"加载股票 {stock_code} 历史数据时发生错误: {str(e)}")
        return None
