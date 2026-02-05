"""
股票价格异动监控程序
功能：监控股票价格突破20日均线，并发送邮件通知
"""

import akshare as ak
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import schedule
import json
import os
import logging
from longport.openapi import QuoteContext, Config, Period, AdjustType, Market
from datetime import date, datetime




class StockPriceMonitor:

    def _setup_logger(self):
        """
        设置日志记录器
        :return: logger 对象
        """
        logger = logging.getLogger('StockPriceMonitor')
        logger.setLevel(logging.INFO)

        # 如果已有处理器，直接返回
        if logger.handlers:
            return logger

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        return logger

    """股票价格监控类"""

    def __init__(self, config_file='monitor_config.json'):
        """
        初始化监控器

        Args:
            config_file: 配置文件路径
        """
        # 初始化日志记录器
        self.logger = self._setup_logger()
        self.config = self._load_config(config_file)
        self.watch_stocks = self.config.get('watch_stocks', [])
        self.alert_history = {}  # 记录已发送的告警，避免重复发送
        # 初始化 longport 上下文 - 用于获取实时行情
        self.quote_ctx = self._init_longport_context()

    def _init_longport_context(self):
        """
        初始化 longport QuoteContext
        """
        try:
            longport_config = self.config.get('longport', {})
            config = Config(
                app_key=longport_config['app_key'],
                app_secret=longport_config['app_secret'],
                access_token=longport_config['access_token']
            )
            return QuoteContext(config)
        except Exception as e:
            self.logger.error(f"初始化 longport 失败: {str(e)}")
            return None

    def _load_config(self, config_file):
        """
        加载配置文件

        Args:
            config_file: 配置文件路径

        Returns:
            配置字典
        """
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 默认配置
            default_config = {
                'watch_stocks': ['00001.HK', '02669.HK', '01113.HK'],  # 示例股票代码
                'email': {
                    'smtp_server': 'smtp.qq.com',
                    'smtp_port': 465,
                    'sender': 'leekle@foxmail.com',
                    'password': 'mzagwwwwhhhgbbjf',  # QQ邮箱授权码
                    'receivers': ['809549783@qq.com']
                },
                'check_interval': 5,  # 检查间隔（分钟）
                'ma_period': 20  # 均线周期
            }
            # 保存默认配置
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=4)
            print(f"已创建默认配置文件：{config_file}，请修改后使用")
            return default_config

    def get_n_day_channel(self, stock_code, days=20):
        """
        获取 20 日通道（上轨、中线、下轨）

        Args:
            stock_code: 股票代码

        Returns:
            {'high': 上轨, 'ma': 中线, 'low': 下轨} 或 None
        """
        bars = self.get_kline_data(stock_code, days=days)
        if not bars or len(bars) == 0:
            return None

        closes = [float(bar.close) for bar in bars]
        highs = [float(bar.high) for bar in bars]
        lows = [float(bar.low) for bar in bars]

        # 上轨：n 日最高价
        upper_band = max(highs)

        # 中线：n 日均线
        middle_band = sum(closes) / len(closes)

        # 下轨：n 日最低价
        lower_band = min(lows)

        return {
            'high': upper_band,
            'ma': middle_band,
            'low': lower_band
        }

    def get_kline_data(self, stock_code, days=20):
        """
        获取近 n 个交易日的 K 线数据

        Args:
            stock_code: 股票代码 (如 000001, SH600000)
            days: 交易日数，默认 20 天

        Returns:
            K 线数据列表，失败返回 None
        """
        try:

            # 按偏移查询，获取最近 n 个交易日
            bars = self.quote_ctx.history_candlesticks_by_offset (stock_code,
                                                                 Period.Day,  # 日 K 线
                                                                 AdjustType.NoAdjust,
                                                                 False,
                                                                 count=days,
                                                                  time = datetime(2026, 2, 2))

            return bars

        except Exception as e:
            self.logger.error(f"获取 {stock_code} K 线数据失败: {str(e)}")
            return None


    def check_breakthrough(self, stock_code, days):
        """
        检查股票是否突破20日均线

        Args:
            stock_code: 股票代码

        Returns:
            突破信息字典，如果未突破则返回None
        """

        # 获取股票数据
        channel = self.get_n_day_channel(stock_code, days)

        # 获取实时价格
        current_price = self.get_realtime_price(stock_code)
        if current_price is None:
            print(f"  无法获取实时价格，使用最新收盘价")
            return None

        # 检查是否有突破
        breakthrough_type = None

        self.logger.info(f"stock:{stock_code},curPrice: {current_price}, 20 high:{channel['high']} ")
        self.logger.info(f"stock:{stock_code},curPrice: {current_price}, 20 low:{channel['low']} ")

        # 向上突破：前一日在均线下方，今日在均线上方
        if current_price > channel['high']:
            breakthrough_type = 'up'

        if current_price < channel['low']:
            breakthrough_type = 'down'

        if breakthrough_type:
            static_info = self.quote_ctx.static_info([stock_code])
            return {
                'stock_code': stock_code,
                'type': breakthrough_type,
                'current_price': current_price,
                'ma_value': channel['high'],
                'change_pct': ((current_price - channel['high']) / channel['high']) * 100,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'stock_name': static_info[0].name_cn
            }

        return None

    def send_email(self, subject, content):
        """
        发送邮件通知

        Args:
            subject: 邮件主题
            content: 邮件内容
        """
        try:
            email_config = self.config.get('email', {})

            # 创建邮件对象
            msg = MIMEMultipart()
            msg['From'] = email_config['sender']
            msg['To'] = ','.join(email_config['receivers'])
            msg['Subject'] = subject

            # 添加邮件内容
            msg.attach(MIMEText(content, 'html', 'utf-8'))

            # 连接SMTP服务器并发送
            server = smtplib.SMTP_SSL(email_config['smtp_server'],
                                      email_config['smtp_port'])
            server.login(email_config['sender'], email_config['password'])
            server.sendmail(email_config['sender'],
                            email_config['receivers'],
                            msg.as_string())
            server.quit()

            print(f"邮件发送成功: {subject}")

        except Exception as e:
            print(f"邮件发送失败: {str(e)}")

    def get_stock_info_xq(self, code):
        """
        通过雪球API获取股票实时行情数据
        :param code: 股票代码 (如 SH600000)
        :return: 包含现价、均线等信息的字典，失败返回None
        """
        try:
            import akshare as ak
            df = ak.stock_individual_spot_xq(symbol=code)

            # 将DataFrame转换为字典便于查询
            data_dict = dict(zip(df['item'], df['value']))

            current_price = float(data_dict.get('现价', 0))
            stock_name = data_dict.get('名称', '未知')

            return {
                'code': code,
                'name': stock_name,
                'current_price': current_price,
                'high': float(data_dict.get('最高', 0)),
                'low': float(data_dict.get('最低', 0)),
                'open': float(data_dict.get('今开', 0)),
                'volume': float(data_dict.get('成交量', 0)),
                'timestamp': data_dict.get('时间', '')
            }
        except Exception as e:
            self.logger.error(f"获取雪球数据失败 - {code}: {str(e)}")
            return None

    def get_stock_name(self, code):
        """
        通过股票代码获取股票名称（使用雪球数据）
        :param code: 股票代码
        :return: 股票名称
        """
        try:
            info = self.get_stock_info_xq(code)
            if info:
                return info.get('name', '未知')
            return '未知'
        except Exception as e:
            self.logger.error(f"获取股票名称失败 - {code}: {str(e)}")
            return '未知'

    def _init_longport_context(self):
        """
        初始化 longport QuoteContext，用于获取实时行情
        :return: QuoteContext 对象，初始化失败返回 None
        """
        try:
            # 从配置文件读取 longport 凭证
            longport_config = self.config.get('longport', {})

            if not longport_config or not all(k in longport_config for k in ['app_key', 'app_secret', 'access_token']):
                self.logger.warning("longport 配置不完整，将使用备用数据源")
                return None

            config = Config(
                app_key=longport_config.get('app_key'),
                app_secret=longport_config.get('app_secret'),
                access_token=longport_config.get('access_token')
            )

            ctx = QuoteContext(config)
            self.logger.info("longport QuoteContext 初始化成功")
            return ctx

        except Exception as e:
            self.logger.error(f"初始化 longport QuoteContext 失败: {str(e)}")
            return None

    def get_realtime_price(self, stock_code):
        """
        通过 longport API 获取实时价格
        """
        try:

            quotes = self.quote_ctx.quote([stock_code])
            current_price = float(quotes[0].last_done)
            return current_price
        except Exception as e:
            self.logger.error(f"获取 {stock_code} 价格失败: {str(e)}")
            return None

    def format_alert_email(self, alert_info, days):
        """
        格式化告警邮件内容

        Args:
            alert_info: 告警信息字典

        Returns:
            邮件主题和内容
        """
        stock_code = alert_info['stock_code']
        stock_name = alert_info['stock_name']
        breakthrough_type = '向上突破' if alert_info['type'] == 'up' else '向下跌破'
        k_line_type = '通道上轨' if alert_info['type'] == 'up' else '通道下轨'

        subject = f"【股票异动】{stock_name} {stock_code} {breakthrough_type}{days}{k_line_type}"

        content = f"""
        <html>
        <body>
            <h2>股票价格异动通知</h2>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr>
                    <td><b>股票名称</b></td>
                    <td>{stock_name}</td>
                </tr>
            
                <tr>
                    <td><b>股票代码</b></td>
                    <td>{stock_code}</td>
                </tr>
                <tr>
                    <td><b>异动类型</b></td>
                    <td style="color: {'red' if alert_info['type'] == 'up' else 'green'};">
                        <b>{breakthrough_type}{days}{k_line_type}</b>
                    </td>
                </tr>
                <tr>
                    <td><b>日期</b></td>
                    <td>{alert_info['date']}</td>
                </tr>
                <tr>
                    <td><b>当前价格</b></td>
                    <td>{alert_info['current_price']:.2f}</td>
                </tr>
                <tr>
                    <td><b>20日均线</b></td>
                    <td>{alert_info['ma_value']:.2f}</td>
                </tr>
                <tr>
                    <td><b>涨跌幅</b></td>
                    <td style="color: {'red' if alert_info['change_pct'] > 0 else 'green'};">
                        {alert_info['change_pct']:.2f}%
                    </td>
                </tr>
            </table>
            <p style="margin-top: 20px; color: #666;">
                发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </body>
        </html>
        """

        return subject, content

    def monitor_once(self):
        """执行一次监控"""
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始监控...")

        self.monitor_once_days(10)
        self.monitor_once_days(20)
        self.monitor_once_days(55)

    def monitor_once_days(self,days):
        for stock_code in self.watch_stocks:

            print(f"检查股票: {stock_code}")
            alert_info = self.check_breakthrough(stock_code, days)

            if alert_info:
                print(f"发现异动: {stock_code} {alert_info['type']}")
                subject, content = self.format_alert_email(alert_info, 20)
                self.send_email(subject, content)
            else:
                print(f"  无异动")

    def is_within_time_range(self, begin_time, end_time):
        """
        判断当前时间是否在指定时间范围内

        Args:
            begin_time: 开始时间，格式 "HH:MM:SS" 如 "9:30:0"
            end_time: 结束时间，格式 "HH:MM:SS" 如 "16:0:0"

        Returns:
            True 在范围内，False 不在范围内
        """
        from datetime import datetime

        now = datetime.now()
        current_time = now.strftime('%H:%M:%S')

        # 时间字符串转成可比较的形式
        begin = begin_time.replace(':', '')
        end = end_time.replace(':', '')
        current = current_time.replace(':', '')

        # 比较大小
        return begin <= current < end

    def start_monitor(self):

        resp = self.quote_ctx.trading_session()
        trade_sessions = []
        for e in resp:
            if(e.market == "HK"):
                trade_sessions = e.trade_sessions

        isOpen = False
        for session in trade_sessions:
            if self.is_within_time_range(session.begin_time, session.end_time):
                isOpen = True
                break
        # if not isOpen:
        #     print(f"{datetime.now()} 非交易时间")
        #     return

        """启动定时监控"""
        check_interval = self.config.get('check_interval', 5)

        print(f"股票价格监控程序启动")
        print(f"监控股票: {', '.join(self.watch_stocks)}")
        print(f"检查间隔: {check_interval}分钟")
        print(f"均线周期: {self.config.get('ma_period', 20)}日")
        print("-" * 50)

        # 立即执行一次
        self.monitor_once()

        # 设置定时任务
        schedule.every(check_interval).minutes.do(self.monitor_once)

        # 持续运行
        while True:
            schedule.run_pending()
            time.sleep(30)  # 每30秒检查一次是否有待执行的任务


def main():
    """主函数"""
    # 创建监控器实例
    monitor = StockPriceMonitor()

    # 启动监控
    monitor.start_monitor()


if __name__ == '__main__':
    main()