#!/usr/bin/env python3
"""
批量LLM股票分析工具
支持批量分析多个股票，生成智能洞察和投资建议
"""

import json
import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import argparse
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
try:
    from tradingagents.utils.logging_manager import get_logger
    logger = get_logger('batch_stock_llm_analyzer')
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    logger = logging.getLogger('batch_stock_llm_analyzer')

# LLM相关导入
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class StockAnalysisConfig:
    """股票分析配置"""
    symbols: List[str]
    output_dir: str
    llm_config: Dict[str, Any]
    analysis_options: Dict[str, Any] = None
    email_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.analysis_options is None:
            self.analysis_options = {}
        if self.email_config is None:
            self.email_config = {}


@dataclass
class StockAnalysisResult:
    """单个股票分析结果"""
    symbol: str
    market_type: str
    analysis_time: str
    data_period: Dict[str, str]
    price_stats: Dict[str, Any]
    llm_insights: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class BatchAnalysisResult:
    """批量分析结果"""
    timestamp: str
    total_symbols: int
    successful_analyses: int
    failed_analyses: int
    results: List[StockAnalysisResult]
    summary: Dict[str, Any]
    duration: float


class TradingAgentsAnalyzer:
    """使用TradingAgents现有分析师团队进行分析"""
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.config = llm_config
        self.graph = None
        self._analysis_count = 0  # 分析计数器
        self._last_cleanup = time.time()  # 上次清理时间
        self._initialize_graph()
    
    def _initialize_graph(self):
        """初始化TradingAgents图"""
        try:
            from tradingagents.graph.trading_graph import TradingAgentsGraph
            from tradingagents.default_config import DEFAULT_CONFIG
            
            # 创建配置
            config = DEFAULT_CONFIG.copy()
            
            # 更新LLM配置
            api_key = self.config.get('api_key', '').strip()
            if not api_key:
                # 从环境变量获取API密钥
                api_key_env = self.config.get('api_key_env', 'DEEPSEEK_API_KEY')
                api_key = os.getenv(api_key_env)
                if api_key:
                    logger.info(f"✅ 从环境变量 {api_key_env} 获取API密钥")
            
            if api_key :
                config['api_key'] = api_key
                logger.info(f"✅ API密钥已设置")
            else:
                logger.error(f"❌ 未找到有效的API密钥")
                logger.error(f"   请检查以下配置:")
                logger.error(f"   1. 配置文件中的 api_key 字段")
                logger.error(f"   2. 环境变量 {self.config.get('api_key_env', 'DEEPSEEK_API_KEY')}")
                logger.error(f"   3. 确保API密钥不是示例值 'sk-example'")
                return
            
            if self.config.get('base_url'):
                config['base_url'] = self.config['base_url']
            if self.config.get('models', {}).get('analysis_model'):
                model_name = self.config['models']['analysis_model']
                config['deep_think_llm'] = model_name
                config['quick_think_llm'] = model_name
            
            # 获取TradingAgents配置
            tradingagents_config = self.config.get('tradingagents', {})
            
            # 设置在线工具
            config['online_tools'] = tradingagents_config.get('online_tools', True)
            config['max_debate_rounds'] = tradingagents_config.get('max_debate_rounds', 2)
            config['max_risk_discuss_rounds'] = tradingagents_config.get('max_risk_discuss_rounds', 1)
            config['llm_provider'] = self.config.get('llm_provider', 'openai')
            if config['llm_provider'] == 'custom_openai':
                config['custom_openai_base_url'] = self.config.get('custom_openai_base_url', 'http://localhost:28000/v1')
                config['custom_openai_api_key'] = self.config.get('custom_openai_api_key', 'sk-example')
            # 根据研究深度调整配置
            research_depth = tradingagents_config.get('research_depth', 3)
            self._apply_research_depth_config(config, research_depth)
            
            # 对于Google AI，需要将API key设置到环境变量中
            if config['llm_provider'].lower() == 'google' and api_key:
                # 如果环境变量中没有设置，则使用配置文件中的API key
                if not os.getenv('GOOGLE_API_KEY'):
                    os.environ['GOOGLE_API_KEY'] = api_key
                    logger.info(f"✅ 已将Google API密钥设置到环境变量 GOOGLE_API_KEY")
            
            # 获取分析师选择
            selected_analysts = self._get_enabled_analysts(tradingagents_config)
            debug_mode = tradingagents_config.get('debug_mode', False)
            
            # 初始化图
            self.graph = TradingAgentsGraph(
                selected_analysts=selected_analysts,
                debug=debug_mode,
                config=config
            )
            
            logger.info(f"✅ TradingAgents图初始化成功")
            logger.info(f"  - 使用分析师: {', '.join(selected_analysts)}")
            logger.info(f"  - 研究深度: {research_depth}级")
            logger.info(f"  - 在线工具: {config.get('online_tools', False)}")
            logger.info(f"  - 调试模式: {debug_mode}")
            logger.info(f"  - 辩论轮次: {config.get('max_debate_rounds', 2)}")
            logger.info(f"  - 风险讨论轮次: {config.get('max_risk_discuss_rounds', 1)}")
            
        except Exception as e:
            logger.error(f"❌ TradingAgents图初始化失败: {e}")
            self.graph = None
    
    def analyze_stock(self, symbol: str, market_type: str, price_data: List[Dict], 
                     price_stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """使用TradingAgents分析单个股票"""
        if not self.graph:
            return None
        
        try:
            # 使用当前日期作为分析日期
            analysis_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"🤖 使用TradingAgents分析: {symbol} ({analysis_date})")
            
            # 运行TradingAgents分析
            state, decision = self.graph.propagate(symbol, analysis_date)
            
            # 提取分析结果
            insights = self._extract_insights_from_state(state, decision)
            
            # 更新分析计数
            self._analysis_count += 1
            
            # 定期清理内存
            self._periodic_cleanup()
            
            return {
                'insights': insights,
                'model_used': 'TradingAgents',
                'tokens_used': 0,  # TradingAgents内部管理token
                'timestamp': datetime.now().isoformat(),
                'raw_state': self._make_serializable(state),
                'raw_decision': self._make_serializable(decision)
            }
            
        except Exception as e:
            logger.error(f"❌ TradingAgents分析失败 {symbol}: {e}")
            return None
    
    def _periodic_cleanup(self):
        """定期清理内存"""
        current_time = time.time()
        # 使用配置的清理间隔
        cleanup_interval = getattr(self, 'memory_cleanup_interval', 10)
        if (self._analysis_count % cleanup_interval == 0 or 
            current_time - self._last_cleanup > 300):
            
            logger.debug(f"🧹 执行内存清理 (分析次数: {self._analysis_count})")
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 更新清理时间
            self._last_cleanup = current_time
    
    def _extract_insights_from_state(self, state: Dict, decision: Dict) -> str:
        """从TradingAgents状态中提取洞察"""
        insights = []
        
        # 添加最终决策
        if decision:
            insights.append("## 🎯 最终投资决策")
            insights.append(f"**推荐动作**: {decision.get('action', 'N/A')}")
            insights.append(f"**置信度**: {decision.get('confidence', 0):.2f}")
            insights.append(f"**推理**: {decision.get('reasoning', 'N/A')}")
            insights.append("")
        
        # 添加分析师报告
        if state:
            # 市场分析师报告
            if state.get('market_report'):
                insights.append("## 📈 市场分析师报告")
                insights.append(state['market_report'])
                insights.append("")
            
            # 基本面分析师报告
            if state.get('fundamentals_report'):
                insights.append("## 📊 基本面分析师报告")
                insights.append(state['fundamentals_report'])
                insights.append("")
            
            # 新闻分析师报告
            if state.get('news_report'):
                insights.append("## 📰 新闻分析师报告")
                insights.append(state['news_report'])
                insights.append("")
            
            # 研究经理总结
            if state.get('research_summary'):
                insights.append("## 👔 研究经理总结")
                insights.append(state['research_summary'])
                insights.append("")
            
            # 交易员决策
            if state.get('trader_decision'):
                insights.append("## 💼 交易员决策")
                insights.append(state['trader_decision'])
                insights.append("")
            
            # 风险经理评估
            if state.get('risk_assessment'):
                insights.append("## ⚠️ 风险经理评估")
                insights.append(state['risk_assessment'])
                insights.append("")
        
        return "\n".join(insights) if insights else "无分析结果"
    
    def _make_serializable(self, obj):
        """将对象转换为可JSON序列化的格式"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # 处理自定义对象
            if hasattr(obj, 'content') and hasattr(obj, 'type'):
                # 处理消息对象
                return {
                    'type': getattr(obj, 'type', 'unknown'),
                    'content': getattr(obj, 'content', str(obj))
                }
            else:
                # 处理其他对象
                return str(obj)
        elif hasattr(obj, 'isoformat'):
            # 处理datetime对象
            return obj.isoformat()
        else:
            # 处理基本类型
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    def _apply_research_depth_config(self, config: Dict, research_depth: int):
        """根据研究深度调整配置"""
        if research_depth == 1:
            # 1级 - 快速分析
            config['max_debate_rounds'] = 1
            config['max_risk_discuss_rounds'] = 1
            config['online_tools'] = False  # 使用缓存数据
        elif research_depth == 2:
            # 2级 - 基础分析
            config['max_debate_rounds'] = 1
            config['max_risk_discuss_rounds'] = 1
            config['online_tools'] = True
        elif research_depth == 3:
            # 3级 - 标准分析（默认）
            config['max_debate_rounds'] = 2
            config['max_risk_discuss_rounds'] = 1
            config['online_tools'] = True
        elif research_depth == 4:
            # 4级 - 深度分析
            config['max_debate_rounds'] = 3
            config['max_risk_discuss_rounds'] = 2
            config['online_tools'] = True
        elif research_depth == 5:
            # 5级 - 全面分析
            config['max_debate_rounds'] = 4
            config['max_risk_discuss_rounds'] = 3
            config['online_tools'] = True
    
    def _get_enabled_analysts(self, tradingagents_config: Dict) -> List[str]:
        """获取启用的分析师列表"""
        # 首先检查新的analyst_teams配置
        analyst_teams = tradingagents_config.get('analyst_teams', {})
        if analyst_teams:
            enabled_analysts = []
            for analyst_name, analyst_config in analyst_teams.items():
                if analyst_config.get('enabled', False):
                    # 转换分析师名称格式
                    if analyst_name == 'market_analyst':
                        enabled_analysts.append('market')
                    elif analyst_name == 'fundamentals_analyst':
                        enabled_analysts.append('fundamentals')
                    elif analyst_name == 'news_analyst':
                        enabled_analysts.append('news')
                    elif analyst_name == 'social_media_analyst':
                        enabled_analysts.append('social')
            return enabled_analysts if enabled_analysts else ["market", "fundamentals", "news"]
        
        # 回退到旧的selected_analysts配置
        return tradingagents_config.get('selected_analysts', ["market", "fundamentals", "news"])


class EmailSender:
    """邮件发送工具类"""
    
    def __init__(self, email_config: Dict[str, Any]):
        """
        初始化邮件发送器
        
        Args:
            email_config: 邮件配置字典，包含:
                - enabled: 是否启用邮件发送
                - smtp_server: SMTP服务器地址
                - smtp_port: SMTP端口
                - smtp_username: SMTP用户名
                - smtp_password: SMTP密码（或环境变量名）
                - smtp_password_env: SMTP密码环境变量名
                - from_email: 发件人邮箱
                - to_emails: 收件人邮箱列表
                - use_tls: 是否使用TLS
                - use_ssl: 是否使用SSL
        """
        self.enabled = email_config.get('enabled', False)
        if not self.enabled:
            logger.info("📧 邮件发送功能未启用")
            return
        
        self.smtp_server = email_config.get('smtp_server', '')
        self.smtp_port = email_config.get('smtp_port', 587)
        self.smtp_username = email_config.get('smtp_username', '')
        
        # 获取密码（优先从环境变量，如果环境变量不存在则使用配置中的密码）
        smtp_password_env = email_config.get('smtp_password_env', '')
        if smtp_password_env:
            # 尝试从环境变量获取
            env_password = os.getenv(smtp_password_env, '')
            if env_password:
                self.smtp_password = env_password
                logger.debug(f"📧 从环境变量 {smtp_password_env} 获取密码")
            else:
                # 环境变量不存在，使用配置中的密码
                self.smtp_password = email_config.get('smtp_password', '')
                if self.smtp_password:
                    logger.debug(f"📧 环境变量 {smtp_password_env} 不存在，使用配置中的密码")
        else:
            self.smtp_password = email_config.get('smtp_password', '')
        
        self.from_email = email_config.get('from_email', '')
        self.to_emails = email_config.get('to_emails', [])
        self.use_tls = email_config.get('use_tls', True)
        self.use_ssl = email_config.get('use_ssl', False)
        
        # 验证配置
        if not self.smtp_server:
            logger.warning("⚠️ SMTP服务器未配置，邮件发送功能将被禁用")
            self.enabled = False
            return
        
        if not self.smtp_username:
            logger.warning("⚠️ SMTP用户名未配置，邮件发送功能将被禁用")
            self.enabled = False
            return
        
        if not self.smtp_password:
            logger.warning("⚠️ SMTP密码未配置，邮件发送功能将被禁用")
            logger.warning("   提示: 163邮箱需要使用授权码，不是普通密码")
            logger.warning("   请检查配置中的 smtp_password 或设置环境变量")
            self.enabled = False
            return
        
        if not self.from_email or not self.to_emails:
            logger.warning("⚠️ 发件人或收件人未配置，邮件发送功能将被禁用")
            self.enabled = False
            return
        
        logger.info(f"✅ 邮件发送器初始化成功")
        logger.info(f"  - SMTP服务器: {self.smtp_server}:{self.smtp_port}")
        logger.info(f"  - 发件人: {self.from_email}")
        logger.info(f"  - 收件人: {', '.join(self.to_emails)}")
    
    def send_email(self, subject: str, body: str, attachments: List[Path] = None) -> bool:
        """
        发送邮件（支持多端口重试）
        
        Args:
            subject: 邮件主题
            body: 邮件正文
            attachments: 附件文件路径列表
        
        Returns:
            是否发送成功
        """
        if not self.enabled:
            logger.debug("📧 邮件发送功能未启用，跳过发送")
            return False
        
        # 创建邮件消息
        msg = MIMEMultipart()
        msg['From'] = self.from_email
        msg['To'] = ', '.join(self.to_emails)
        msg['Subject'] = subject
        
        # 添加正文
        msg.attach(MIMEText(body, 'html', 'utf-8'))
        
        # 添加附件
        if attachments:
            for attachment_path in attachments:
                if attachment_path.exists():
                    try:
                        with open(attachment_path, 'rb') as f:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(f.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename= {attachment_path.name}'
                            )
                            msg.attach(part)
                            logger.info(f"📎 已添加附件: {attachment_path.name}")
                    except Exception as e:
                        logger.warning(f"⚠️ 添加附件失败 {attachment_path}: {e}")
        
        # 163邮箱的备用配置（按优先级排序）
        if "163.com" in self.smtp_server:
            configs = [
                {"port": 465, "use_ssl": True, "use_tls": False, "name": "SSL(465)"},
                {"port": 25, "use_ssl": False, "use_tls": False, "name": "无加密(25)"},
                {"port": 587, "use_ssl": False, "use_tls": True, "name": "TLS(587)"},
            ]
        else:
            # 其他邮箱使用原始配置
            configs = [{
                "port": self.smtp_port,
                "use_ssl": self.use_ssl,
                "use_tls": self.use_tls,
                "name": f"原始配置({self.smtp_port})"
            }]
        
        # 尝试不同的配置
        for config in configs:
            try:
                logger.info(f"📧 正在发送邮件到 {', '.join(self.to_emails)}...")
                logger.info(f"📧 尝试配置: {config['name']} - {self.smtp_server}:{config['port']}")
                
                if config['use_ssl']:
                    logger.debug(f"📧 使用SSL连接...")
                    server = smtplib.SMTP_SSL(self.smtp_server, config['port'], timeout=30)
                else:
                    logger.debug(f"📧 使用普通连接...")
                    server = smtplib.SMTP(self.smtp_server, config['port'], timeout=30)
                
                if config['use_tls'] and not config['use_ssl']:
                    logger.debug(f"📧 启用TLS...")
                    server.starttls()
                
                logger.debug(f"📧 尝试登录...")
                server.login(self.smtp_username, self.smtp_password)
                logger.debug(f"📧 登录成功，发送邮件...")
                
                server.send_message(msg)
                server.quit()
                
                logger.info(f"✅ 邮件发送成功 (使用配置: {config['name']})")
                return True
                
            except smtplib.SMTPAuthenticationError as e:
                logger.error(f"❌ 认证失败 (配置: {config['name']}): {e}")
                logger.error(f"   请检查:")
                logger.error(f"   1. SMTP用户名是否正确: {self.smtp_username}")
                logger.error(f"   2. SMTP密码/授权码是否正确")
                logger.error(f"   3. 163邮箱需要使用授权码，不是普通密码")
                logger.error(f"   4. 是否已开启SMTP服务")
                # 认证错误不需要尝试其他配置
                return False
            except (smtplib.SMTPException, ConnectionError, OSError) as e:
                error_msg = str(e)
                logger.warning(f"⚠️ 连接失败 (配置: {config['name']}): {error_msg}")
                # 继续尝试下一个配置
                continue
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"⚠️ 发送失败 (配置: {config['name']}): {error_msg}")
                # 继续尝试下一个配置
                continue
        
        # 所有配置都失败
        logger.error(f"❌ 所有SMTP配置都失败，无法发送邮件")
        logger.error(f"   已尝试的配置:")
        for config in configs:
            logger.error(f"   - {config['name']}: {self.smtp_server}:{config['port']}")
        logger.error(f"   建议:")
        logger.error(f"   1. 检查网络连接")
        logger.error(f"   2. 检查防火墙设置")
        logger.error(f"   3. 确认163邮箱已开启SMTP服务")
        logger.error(f"   4. 尝试手动测试SMTP连接")
        return False
    
    def send_analysis_results(self, batch_result: BatchAnalysisResult, 
                             summary_file: Path = None, 
                             json_file: Path = None) -> bool:
        """
        发送分析结果邮件（内容直接展示在邮件正文中，不发送附件）
        
        Args:
            batch_result: 批量分析结果
            summary_file: 汇总报告文件路径（用于读取内容）
            json_file: JSON数据文件路径（用于读取内容）
        
        Returns:
            是否发送成功
        """
        if not self.enabled:
            return False
        
        # 生成邮件主题
        analysis_date = datetime.fromisoformat(batch_result.timestamp).strftime('%Y-%m-%d')
        subject = f"股票分析报告 - {analysis_date} ({batch_result.successful_analyses}/{batch_result.total_symbols} 成功)"
        
        # 生成邮件正文
        success_rate = batch_result.successful_analyses / batch_result.total_symbols * 100 if batch_result.total_symbols > 0 else 0
        
        # 读取汇总报告内容（如果存在）
        summary_content = ""
        if summary_file and summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_content = f.read()
            except Exception as e:
                logger.warning(f"⚠️ 读取汇总报告失败: {e}")
        
        # 生成HTML邮件正文
        body = self._generate_email_body(
            batch_result, analysis_date, success_rate, summary_content
        )
        
        # 发送邮件（不发送附件）
        return self.send_email(subject, body, attachments=None)
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        """将Markdown文本转换为HTML"""
        if not markdown_text:
            return ""
        
        html = markdown_text
        
        # 转换标题
        html = html.replace('# ', '<h1>').replace('\n# ', '</h1>\n<h1>')
        html = html.replace('## ', '<h2>').replace('\n## ', '</h2>\n<h2>')
        html = html.replace('### ', '<h3>').replace('\n### ', '</h3>\n<h3>')
        html = html.replace('#### ', '<h4>').replace('\n#### ', '</h4>\n<h4>')
        
        # 转换粗体
        html = html.replace('**', '<strong>').replace('**', '</strong>')
        
        # 转换列表
        lines = html.split('\n')
        in_list = False
        result_lines = []
        
        for line in lines:
            if line.strip().startswith('- ') or line.strip().startswith('* '):
                if not in_list:
                    result_lines.append('<ul>')
                    in_list = True
                content = line.strip()[2:]
                result_lines.append(f'<li>{content}</li>')
            elif line.strip().startswith('|') and '|' in line[1:]:
                # 表格行
                if not in_list:
                    result_lines.append('</ul>' if in_list else '')
                    in_list = False
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if '---' in line or '---' in ''.join(cells):
                    result_lines.append('<tr>' + ''.join([f'<th>{cell}</th>' for cell in cells]) + '</tr>')
                else:
                    result_lines.append('<tr>' + ''.join([f'<td>{cell}</td>' for cell in cells]) + '</tr>')
            else:
                if in_list:
                    result_lines.append('</ul>')
                    in_list = False
                if line.strip():
                    result_lines.append(f'<p>{line}</p>')
                else:
                    result_lines.append('<br>')
        
        if in_list:
            result_lines.append('</ul>')
        
        return '\n'.join(result_lines)
    
    def _generate_email_body(self, batch_result: BatchAnalysisResult, 
                             analysis_date: str, success_rate: float,
                             summary_content: str = "") -> str:
        """生成美观的HTML邮件正文"""
        
        # 生成股票详细分析内容
        stock_details_html = ""
        for result in batch_result.results:
            if result.error:
                continue
            
            # 获取分析时间
            try:
                analysis_time = datetime.fromisoformat(result.analysis_time).strftime('%Y-%m-%d %H:%M:%S')
            except:
                analysis_time = result.analysis_time
            
            stock_html = f"""
            <div class="stock-detail">
                <h3>📊 {result.symbol} ({result.market_type})</h3>
                <div class="stock-info">
                    <p><strong>分析时间:</strong> {analysis_time}</p>
"""
            
            # 添加价格统计
            if result.price_stats:
                stock_html += "<table class='price-stats'>\n"
                for key, value in result.price_stats.items():
                    if isinstance(value, (int, float)):
                        if 'price' in key.lower():
                            stock_html += f"<tr><td><strong>{key}</strong></td><td>{value:.2f}</td></tr>\n"
                        else:
                            stock_html += f"<tr><td><strong>{key}</strong></td><td>{value:.4f}</td></tr>\n"
                    else:
                        stock_html += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>\n"
                stock_html += "</table>\n"
            
            # 添加TradingAgents分析结果
            if result.llm_insights and result.llm_insights.get('insights'):
                insights = result.llm_insights.get('insights', '')
                # 将Markdown格式的insights转换为HTML
                insights_html = self._markdown_to_html(insights)
                stock_html += f"""
                <div class="llm-analysis">
                    <h4>🤖 TradingAgents 智能分析</h4>
                    <div class="analysis-content">
                        {insights_html}
                    </div>
                </div>
"""
            
            stock_html += """
                </div>
            </div>
"""
            stock_details_html += stock_html
        
        # 生成完整HTML
        body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px 20px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 28px; }}
        .header p {{ margin: 10px 0 0 0; font-size: 16px; opacity: 0.9; }}
        .content {{ padding: 20px; max-width: 900px; margin: 0 auto; }}
        .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #667eea; }}
        .stats {{ display: flex; justify-content: space-around; margin: 20px 0; flex-wrap: wrap; }}
        .stat-box {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; flex: 1; margin: 5px; min-width: 120px; }}
        .stat-value {{ font-size: 32px; font-weight: bold; margin-bottom: 5px; }}
        .stat-label {{ font-size: 14px; opacity: 0.9; }}
        .stock-detail {{ background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stock-detail h3 {{ color: #667eea; margin-top: 0; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        .stock-info {{ margin-top: 15px; }}
        .price-stats {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        .price-stats td {{ padding: 8px 12px; border-bottom: 1px solid #e0e0e0; }}
        .price-stats td:first-child {{ color: #666; width: 40%; }}
        .price-stats td:last-child {{ color: #333; font-weight: 500; }}
        .llm-analysis {{ background-color: #f8f9fa; padding: 15px; border-radius: 6px; margin-top: 15px; }}
        .llm-analysis h4 {{ color: #667eea; margin-top: 0; }}
        .analysis-content {{ color: #555; }}
        .analysis-content h1, .analysis-content h2, .analysis-content h3, .analysis-content h4 {{ color: #667eea; margin-top: 15px; }}
        .analysis-content ul {{ padding-left: 20px; }}
        .analysis-content li {{ margin: 5px 0; }}
        .analysis-content table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        .analysis-content table th, .analysis-content table td {{ padding: 8px; border: 1px solid #ddd; text-align: left; }}
        .analysis-content table th {{ background-color: #f0f0f0; font-weight: bold; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; background-color: #f8f9fa; margin-top: 30px; }}
        .error-box {{ background-color: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 6px; margin: 10px 0; }}
        .error-box strong {{ color: #856404; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 批量股票分析报告</h1>
        <p>分析日期: {analysis_date}</p>
    </div>
    
    <div class="content">
        <div class="summary">
            <h2>📈 分析概览</h2>
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">{batch_result.total_symbols}</div>
                    <div class="stat-label">总股票数</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{batch_result.successful_analyses}</div>
                    <div class="stat-label">成功分析</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{batch_result.failed_analyses}</div>
                    <div class="stat-label">失败分析</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{success_rate:.1f}%</div>
                    <div class="stat-label">成功率</div>
                </div>
            </div>
            <p><strong>分析耗时:</strong> {batch_result.duration:.2f} 秒</p>
"""
        
        # 添加市场分布
        if batch_result.summary.get('market_distribution'):
            body += """
            <h3>📊 市场分布</h3>
            <ul>
"""
            for market, count in batch_result.summary.get('market_distribution', {}).items():
                body += f"                <li><strong>{market}:</strong> {count} 只股票</li>\n"
            body += "            </ul>\n"
        
        body += """
        </div>
        
        <div class="summary">
            <h2>📋 详细分析结果</h2>
"""
        
        # 添加汇总报告内容（如果存在）
        if summary_content:
            summary_html = self._markdown_to_html(summary_content)
            body += f"""
            <div class="summary-content">
                {summary_html}
            </div>
"""
        
        # 添加每个股票的详细分析
        if stock_details_html:
            body += stock_details_html
        
        # 添加失败的分析
        failed_stocks = [r for r in batch_result.results if r.error]
        if failed_stocks:
            body += """
            <div class="summary">
                <h2>❌ 失败分析</h2>
"""
            for result in failed_stocks:
                body += f"""
                <div class="error-box">
                    <strong>{result.symbol}</strong> ({result.market_type}): {result.error}
                </div>
"""
            body += "            </div>\n"
        
        body += """
        </div>
    </div>
    
    <div class="footer">
        <p>此邮件由 TradingAgents-CN 自动生成</p>
        <p>生成时间: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        <p>分析结果已保存在服务器，可通过系统查看完整报告</p>
    </div>
</body>
</html>
"""
        return body


class BatchStockLLMAnalyzer:
    """批量股票LLM分析器"""
    
    def __init__(self, config: StockAnalysisConfig):
        self.config = config
        self.llm_analyzer = TradingAgentsAnalyzer(config.llm_config)
        self.results = []
        
        # 批量处理配置
        self.batch_settings = config.analysis_options.get('batch_settings', {})
        self.max_concurrent = self.batch_settings.get('max_concurrent', 2)
        self.delay_between_requests = self.batch_settings.get('delay_between_requests', 3)
        self.retry_failed = self.batch_settings.get('retry_failed', True)
        self.max_retries = self.batch_settings.get('max_retries', 3)
        self.memory_cleanup_interval = self.batch_settings.get('memory_cleanup_interval', 10)
        self.batch_delay_multiplier = self.batch_settings.get('batch_delay_multiplier', 2)
        self.api_rate_limit_detection = self.batch_settings.get('api_rate_limit_detection', True)
        self.adaptive_delay = self.batch_settings.get('adaptive_delay', True)
        self.stop_on_quota_exceeded = self.batch_settings.get('stop_on_quota_exceeded', True)
        
        # 初始化邮件发送器
        # 邮件配置从email_config字段获取
        email_config = config.email_config or {}
        self.email_sender = EmailSender(email_config)
        
        # 确保输出目录存在
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def run_batch_analysis(self) -> BatchAnalysisResult:
        """运行批量分析"""
        start_time = time.time()
        logger.info(f"🚀 开始批量LLM股票分析: {len(self.config.symbols)} 只股票")
        logger.info(f"📋 批量处理配置: 最大并发={self.max_concurrent}, 请求间隔={self.delay_between_requests}s")
        
        successful = 0
        failed = 0
        
        # 分批处理股票，控制并发数量
        batch_size = min(self.max_concurrent, len(self.config.symbols))
        symbol_batches = [self.config.symbols[i:i + batch_size] 
                         for i in range(0, len(self.config.symbols), batch_size)]
        
        for batch_idx, symbol_batch in enumerate(symbol_batches, 1):
            logger.info(f"📦 处理批次 {batch_idx}/{len(symbol_batches)}: {len(symbol_batch)} 只股票")
            
            for i, symbol in enumerate(symbol_batch, 1):
                global_idx = (batch_idx - 1) * batch_size + i
                logger.info(f"📊 分析股票 {global_idx}/{len(self.config.symbols)}: {symbol}")
                
                # 重试机制
                result = self._analyze_with_retry(symbol, global_idx)
                
                if result.error:
                    failed += 1
                    logger.error(f"❌ 分析失败 {symbol}: {result.error}")
                    
                    # 检查是否是配额超限错误，如果是则停止处理
                    if self.stop_on_quota_exceeded and self._is_quota_exceeded_error(result.error):
                        logger.error(f"🛑 配额已超限，停止批量处理")
                        logger.error(f"   已处理: {successful} 成功, {failed} 失败")
                        logger.error(f"   剩余股票: {len(self.config.symbols) - global_idx} 只")
                        # 为剩余股票创建占位结果
                        for remaining_symbol in self.config.symbols[global_idx:]:
                            self.results.append(StockAnalysisResult(
                                symbol=remaining_symbol,
                                market_type="unknown",
                                analysis_time=datetime.now().isoformat(),
                                data_period={},
                                price_stats={},
                                error="配额超限，未处理"
                            ))
                        break
                else:
                    successful += 1
                    logger.info(f"✅ 分析完成 {symbol}")
                
                self.results.append(result)
                
                # 添加延迟避免API限制
                if i < len(symbol_batch):
                    logger.debug(f"⏳ 等待 {self.delay_between_requests}s 避免API限制...")
                    time.sleep(self.delay_between_requests)
            
            # 批次间额外延迟
            if batch_idx < len(symbol_batches):
                batch_delay = self.delay_between_requests * self.batch_delay_multiplier
                logger.info(f"⏳ 批次间等待 {batch_delay}s...")
                time.sleep(batch_delay)
        
        duration = time.time() - start_time
        
        # 生成汇总分析
        summary = self._generate_summary()
        
        batch_result = BatchAnalysisResult(
            timestamp=datetime.now().isoformat(),
            total_symbols=len(self.config.symbols),
            successful_analyses=successful,
            failed_analyses=failed,
            results=self.results,
            summary=summary,
            duration=duration
        )
        
        # 保存结果
        self._save_results(batch_result)
        
        logger.info(f"🎉 批量分析完成!")
        logger.info(f"📊 成功: {successful}, 失败: {failed}, 耗时: {duration:.2f}s")
        
        return batch_result
    
    def _analyze_with_retry(self, symbol: str, global_idx: int) -> StockAnalysisResult:
        """带重试机制的股票分析"""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"🔄 重试分析 {symbol} (尝试 {attempt + 1}/{self.max_retries + 1})")
                    # 重试前等待更长时间
                    wait_time = self.delay_between_requests * (2 ** attempt)
                    logger.info(f"⏳ 重试前等待 {wait_time}s...")
                    time.sleep(wait_time)
                
                result = self._analyze_single_stock(symbol)
                
                # 如果成功，返回结果
                if not result.error:
                    if attempt > 0:
                        logger.info(f"✅ 重试成功 {symbol}")
                    return result
                
                # 记录错误但继续重试
                last_error = result.error
                logger.warning(f"⚠️ 分析失败 {symbol} (尝试 {attempt + 1}): {result.error}")
                
            except Exception as e:
                last_error = str(e)
                error_str = str(e)
                logger.warning(f"⚠️ 分析异常 {symbol} (尝试 {attempt + 1}): {e}")
                
                # 检查是否是配额超限错误（需要停止处理）
                if self._is_quota_exceeded_error(error_str):
                    logger.error(f"🚫 检测到配额超限错误！已达到Google API每日200次请求限制")
                    logger.error(f"   错误信息: {error_str}")
                    logger.error(f"   解决方案:")
                    logger.error(f"   1. 等待24小时后配额重置")
                    logger.error(f"   2. 升级到Google AI付费计划以获得更高配额")
                    logger.error(f"   3. 减少批量分析的股票数量")
                    logger.error(f"   访问配额监控: https://ai.dev/usage?tab=rate-limit")
                    # 返回特殊错误标记，让主循环知道需要停止
                    return StockAnalysisResult(
                        symbol=symbol,
                        market_type="unknown",
                        analysis_time=datetime.now().isoformat(),
                        data_period={},
                        price_stats={},
                        error=f"配额超限: {error_str}"
                    )
                
                # 检查是否是API限制错误（可以重试）
                if self._is_rate_limit_error(error_str):
                    logger.warning(f"🚫 检测到API限制错误，将延长等待时间")
                    if attempt < self.max_retries:
                        # API限制时等待更长时间
                        wait_time = self.delay_between_requests * 5 * (2 ** attempt)
                        logger.info(f"⏳ API限制等待 {wait_time}s...")
                        time.sleep(wait_time)
                elif attempt < self.max_retries:
                    # 其他错误等待较短时间
                    time.sleep(self.delay_between_requests)
        
        # 所有重试都失败
        logger.error(f"❌ 分析最终失败 {symbol} (已重试 {self.max_retries} 次): {last_error}")
        return StockAnalysisResult(
            symbol=symbol,
            market_type="unknown",
            analysis_time=datetime.now().isoformat(),
            data_period={},
            price_stats={},
            error=f"重试{self.max_retries}次后仍失败: {last_error}"
        )
    
    def _is_rate_limit_error(self, error_msg: str) -> bool:
        """检查是否是API限制错误"""
        rate_limit_indicators = [
            "rate limit", "rate_limit", "too many requests", "429",
            "quota exceeded", "quota_exceeded", "throttled",
            "api limit", "api_limit", "request limit"
        ]
        error_lower = error_msg.lower()
        return any(indicator in error_lower for indicator in rate_limit_indicators)
    
    def _is_quota_exceeded_error(self, error_msg: str) -> bool:
        """检查是否是配额超限错误（需要停止处理）"""
        quota_exceeded_indicators = [
            "quota exceeded", "quota_exceeded", "exceeded your current quota",
            "limit: 200", "free_tier_requests"
        ]
        error_lower = error_msg.lower()
        return any(indicator in error_lower for indicator in quota_exceeded_indicators)
    
    def _analyze_single_stock(self, symbol: str) -> StockAnalysisResult:
        """分析单个股票"""
        try:
            # 检测市场类型
            market_type = self._detect_market_type(symbol)
            
            # 使用TradingAgents进行完整分析
            llm_insights = None
            if self.llm_analyzer.graph:
                logger.info(f"🤖 使用TradingAgents分析: {symbol}")
                llm_insights = self.llm_analyzer.analyze_stock(
                    symbol, market_type, [], {}  # TradingAgents会自己获取数据
                )
            
            # 从TradingAgents结果中提取数据统计
            data_period = {}
            price_stats = {}
            
            if llm_insights and llm_insights.get('raw_state'):
                state = llm_insights['raw_state']
                # 尝试从状态中提取数据信息
                if 'data_period' in state:
                    data_period = state['data_period']
                if 'price_stats' in state:
                    price_stats = state['price_stats']
            
            return StockAnalysisResult(
                symbol=symbol,
                market_type=market_type,
                analysis_time=datetime.now().isoformat(),
                data_period=data_period,
                price_stats=price_stats,
                llm_insights=llm_insights
            )
            
        except Exception as e:
            return StockAnalysisResult(
                symbol=symbol,
                market_type="unknown",
                analysis_time=datetime.now().isoformat(),
                data_period={},
                price_stats={},
                error=str(e)
            )
    
    def _detect_market_type(self, symbol: str) -> str:
        """检测股票市场类型"""
        import re
        if re.match(r'^[A-Z]{1,5}$', symbol.upper()):
            return "美股"
        elif re.match(r'^\d{6}$', symbol):
            return "A股"
        elif re.match(r'^\d{4,5}(\.HK)?$', symbol.upper()):
            return "港股"
        else:
            return "美股"  # 默认美股
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成汇总分析"""
        successful_results = [r for r in self.results if not r.error]
        
        if not successful_results:
            return {"error": "没有成功的分析结果"}
        
        # 统计信息
        market_distribution = {}
        avg_volatility = 0
        price_ranges = []
        
        for result in successful_results:
            # 市场分布
            market = result.market_type
            market_distribution[market] = market_distribution.get(market, 0) + 1
            
            # 波动率统计
            volatility = result.price_stats.get('price_volatility', 0)
            avg_volatility += volatility
            
            # 价格区间
            price_range = result.price_stats.get('price_range', {})
            if price_range:
                price_ranges.append({
                    'symbol': result.symbol,
                    'min': price_range.get('min', 0),
                    'max': price_range.get('max', 0)
                })
        
        avg_volatility /= len(successful_results)
        
        # 排序价格区间
        price_ranges.sort(key=lambda x: x['max'], reverse=True)
        
        return {
            'market_distribution': market_distribution,
            'average_volatility': avg_volatility,
            'top_price_ranges': price_ranges[:5],  # 前5个最高价格区间
            'total_analyzed': len(successful_results),
            'analysis_success_rate': len(successful_results) / len(self.results) * 100
        }
    
    def _save_results(self, batch_result: BatchAnalysisResult):
        """保存分析结果"""
        # 创建按日期分组的文件夹结构
        analysis_date = datetime.now().strftime('%Y-%m-%d')
        date_folder = Path(self.config.output_dir) / analysis_date
        date_folder.mkdir(parents=True, exist_ok=True)
        
        # 保存批量分析的JSON结果
        json_file = date_folder / f"batch_analysis_{datetime.now().strftime('%H%M%S')}.json"
        serializable_result = self._make_serializable(asdict(batch_result))
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
        
        # 为每个股票创建单独的Markdown文件
        individual_files = []
        for result in batch_result.results:
            if not result.error:
                # 创建股票分析文件夹
                stock_folder = date_folder / f"{result.symbol}_{result.market_type}"
                stock_folder.mkdir(exist_ok=True)
                
                # 生成单个股票的Markdown报告
                markdown_content = self._generate_individual_stock_markdown(result)
                markdown_file = stock_folder / f"{result.symbol}_analysis.md"
                
                with open(markdown_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                individual_files.append(markdown_file)
                logger.info(f"📄 股票分析已保存: {markdown_file}")
        
        # 生成批量分析汇总报告
        summary_file = date_folder / f"batch_summary_{datetime.now().strftime('%H%M%S')}.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_batch_summary_markdown(batch_result))
        
        logger.info(f"📄 批量分析结果已保存到: {date_folder}")
        logger.info(f"  - 汇总报告: {summary_file}")
        logger.info(f"  - 个别股票文件: {len(individual_files)} 个")
        logger.info(f"  - JSON数据: {json_file}")
        
        # 发送邮件通知
        if self.email_sender.enabled:
            logger.info(f"📧 准备发送邮件通知...")
            email_success = self.email_sender.send_analysis_results(
                batch_result=batch_result,
                summary_file=summary_file,
                json_file=json_file
            )
            if email_success:
                logger.info(f"✅ 邮件通知已发送")
            else:
                logger.warning(f"⚠️ 邮件发送失败，但分析结果已保存")
    
    def _generate_text_report(self, batch_result: BatchAnalysisResult) -> str:
        """生成文本报告"""
        report = f"""
# 批量LLM股票分析报告

## 分析概览
- 分析时间: {batch_result.timestamp}
- 总股票数: {batch_result.total_symbols}
- 成功分析: {batch_result.successful_analyses}
- 失败分析: {batch_result.failed_analyses}
- 成功率: {batch_result.successful_analyses / batch_result.total_symbols * 100:.1f}%
- 分析耗时: {batch_result.duration:.2f} 秒

## 市场分布
"""
        
        for market, count in batch_result.summary.get('market_distribution', {}).items():
            report += f"- {market}: {count} 只股票\n"
        
        report += f"""
## 市场统计
- 平均波动率: {batch_result.summary.get('average_volatility', 0):.4f}
- 分析成功率: {batch_result.summary.get('analysis_success_rate', 0):.1f}%

## 详细分析结果
"""
        
        for result in batch_result.results:
            report += f"""
### {result.symbol} ({result.market_type})
- 分析时间: {result.analysis_time}
- 数据期间: {result.data_period.get('start', 'N/A')} 至 {result.data_period.get('end', 'N/A')}
- 平均价格: {result.price_stats.get('avg_price', 0):.2f}
- 价格波动率: {result.price_stats.get('price_volatility', 0):.4f}
- 价格区间: {result.price_stats.get('price_range', {}).get('min', 0):.2f} - {result.price_stats.get('price_range', {}).get('max', 0):.2f}
"""
            
            if result.error:
                report += f"- ❌ 错误: {result.error}\n"
            elif result.llm_insights:
                report += f"""
#### 🤖 TradingAgents智能分析
- 分析引擎: {result.llm_insights.get('model_used', 'TradingAgents')}
- 分析时间: {result.llm_insights.get('timestamp', 'unknown')}
- 市场类型: {result.market_type}

**专业分析结果:**
{result.llm_insights.get('insights', '无分析数据')}
"""
            else:
                report += "- ⚠️ 未生成TradingAgents分析\n"
        
        return report
    
    def _generate_individual_stock_markdown(self, result: StockAnalysisResult) -> str:
        """生成单个股票的Markdown报告"""
        analysis_time = datetime.fromisoformat(result.analysis_time)
        
        markdown = f"""# {result.symbol} 股票分析报告

## 📊 基本信息

| 项目 | 详情 |
|------|------|
| **股票代码** | {result.symbol} |
| **市场类型** | {result.market_type} |
| **分析时间** | {analysis_time.strftime('%Y-%m-%d %H:%M:%S')} |
| **数据期间** | {result.data_period.get('start', 'N/A')} 至 {result.data_period.get('end', 'N/A')} |

## 📈 价格统计

"""
        
        # 添加价格统计信息
        if result.price_stats:
            markdown += "| 指标 | 数值 |\n|------|------|\n"
            for key, value in result.price_stats.items():
                if isinstance(value, (int, float)):
                    if 'price' in key.lower() or 'price' in key:
                        markdown += f"| **{key}** | {value:.2f} |\n"
                    else:
                        markdown += f"| **{key}** | {value:.4f} |\n"
                else:
                    markdown += f"| **{key}** | {value} |\n"
        
        # 添加TradingAgents分析结果
        if result.llm_insights and result.llm_insights.get('insights'):
            markdown += f"""
## 🤖 TradingAgents 智能分析

### 分析引擎信息
- **模型**: {result.llm_insights.get('model_used', 'TradingAgents')}
- **分析时间**: {result.llm_insights.get('timestamp', 'N/A')}
- **Token使用**: {result.llm_insights.get('tokens_used', 0)}

### 专业分析结果

{result.llm_insights.get('insights', '无分析数据')}

"""
        else:
            markdown += """
## ⚠️ 分析状态

**TradingAgents分析未完成或失败**

可能原因：
- API调用失败
- 网络连接问题
- 股票数据获取失败
- 模型服务不可用

"""
        
        # 添加错误信息（如果有）
        if result.error:
            markdown += f"""
## ❌ 错误信息

```
{result.error}
```

## 🔧 故障排除建议

1. **检查网络连接**: 确保网络连接稳定
2. **验证股票代码**: 确认股票代码格式正确
3. **检查API配置**: 验证API密钥和端点配置
4. **重试分析**: 稍后重新运行分析

"""
        
        # 添加页脚
        markdown += f"""
---

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  
*分析引擎: TradingAgents-CN*  
*股票代码: {result.symbol}*
"""
        
        return markdown
    
    def _generate_batch_summary_markdown(self, batch_result: BatchAnalysisResult) -> str:
        """生成批量分析汇总的Markdown报告"""
        analysis_time = datetime.fromisoformat(batch_result.timestamp)
        success_rate = batch_result.successful_analyses / batch_result.total_symbols * 100
        
        markdown = f"""# 批量股票分析汇总报告

## 📊 分析概览

| 项目 | 数值 |
|------|------|
| **分析日期** | {analysis_time.strftime('%Y-%m-%d')} |
| **分析时间** | {analysis_time.strftime('%H:%M:%S')} |
| **总股票数** | {batch_result.total_symbols} |
| **成功分析** | {batch_result.successful_analyses} |
| **失败分析** | {batch_result.failed_analyses} |
| **成功率** | {success_rate:.1f}% |
| **分析耗时** | {batch_result.duration:.2f} 秒 |

## 📈 市场分布

"""
        
        # 添加市场分布
        if batch_result.summary.get('market_distribution'):
            markdown += "| 市场类型 | 股票数量 |\n|----------|----------|\n"
            for market, count in batch_result.summary['market_distribution'].items():
                markdown += f"| **{market}** | {count} |\n"
        
        # 添加统计信息
        markdown += f"""
## 📊 统计信息

- **平均波动率**: {batch_result.summary.get('average_volatility', 0):.4f}
- **分析成功率**: {batch_result.summary.get('analysis_success_rate', 0):.1f}%

## 📋 分析结果列表

"""
        
        # 添加每个股票的分析结果
        for i, result in enumerate(batch_result.results, 1):
            status = "✅ 成功" if not result.error else "❌ 失败"
            markdown += f"{i}. **{result.symbol}** ({result.market_type}) - {status}\n"
            if result.error:
                markdown += f"   - 错误: {result.error}\n"
            markdown += "\n"
        
        # 添加价格区间信息
        if batch_result.summary.get('top_price_ranges'):
            markdown += """
## 💰 价格区间排行

| 股票代码 | 最低价 | 最高价 |
|----------|--------|--------|
"""
            for price_range in batch_result.summary['top_price_ranges']:
                markdown += f"| {price_range['symbol']} | {price_range['min']:.2f} | {price_range['max']:.2f} |\n"
        
        # 添加页脚
        markdown += f"""
---

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  
*分析引擎: TradingAgents-CN*  
*批量分析ID: {batch_result.timestamp}*
"""
        
        return markdown
    
    def _make_serializable(self, obj):
        """将对象转换为可JSON序列化的格式"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # 处理自定义对象
            if hasattr(obj, 'content') and hasattr(obj, 'type'):
                # 处理消息对象
                return {
                    'type': getattr(obj, 'type', 'unknown'),
                    'content': getattr(obj, 'content', str(obj))
                }
            else:
                # 处理其他对象
                return str(obj)
        elif hasattr(obj, 'isoformat'):
            # 处理datetime对象
            return obj.isoformat()
        else:
            # 处理基本类型
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)


def load_config(config_file: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='批量LLM股票分析工具',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('symbols', nargs='*', help='股票代码列表，支持美股、A股、港股。如果不提供，将使用配置文件中的默认股票列表')
    parser.add_argument('--output', '-o', default='./reports/batch_stock_analysis', 
                       help='输出目录 (默认: ./reports/batch_stock_analysis)')
    parser.add_argument('--config', '-c', default='scripts/batch_stock_config.json',
                       help='配置文件路径 (默认: scripts/batch_stock_config.json)')
    parser.add_argument('--stock-list', '-l', help='使用配置文件中的预设股票组合名称')
    parser.add_argument('--list-stocks', action='store_true', help='显示可用的股票组合列表')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        config_data = load_config(args.config)
        llm_config = config_data.get('llm_config', {})
        stock_lists = config_data.get('stock_lists', {})
        
        # 处理股票列表
        symbols = []
        
        # 如果用户要求显示股票组合列表
        if args.list_stocks:
            print("📊 可用的股票组合:")
            print("=" * 50)
            for list_name, list_config in stock_lists.items():
                if isinstance(list_config, dict) and 'symbols' in list_config:
                    print(f"📈 {list_name}: {list_config.get('description', '无描述')}")
                    print(f"   股票: {', '.join(list_config['symbols'])}")
                    print(f"   市场: {', '.join(list_config.get('market_types', []))}")
                    print()
            return
        
        # 确定要分析的股票列表
        if args.stock_list:
            # 使用指定的股票组合
            if args.stock_list in stock_lists:
                stock_config = stock_lists[args.stock_list]
                symbols = stock_config.get('symbols', [])
                print(f"📊 使用股票组合: {args.stock_list}")
                print(f"   描述: {stock_config.get('description', '无描述')}")
                print(f"   股票: {', '.join(symbols)}")
            else:
                print(f"❌ 错误: 股票组合 '{args.stock_list}' 不存在")
                print("可用组合:", ', '.join(stock_lists.keys()))
                return
        elif args.symbols:
            # 使用命令行提供的股票代码
            symbols = args.symbols
        else:
            # 使用默认股票组合
            default_list = config_data.get('batch_analysis', {}).get('default_stock_list', 'default')
            if default_list in stock_lists:
                stock_config = stock_lists[default_list]
                symbols = stock_config.get('symbols', [])
                print(f"📊 使用默认股票组合: {default_list}")
                print(f"   描述: {stock_config.get('description', '无描述')}")
                print(f"   股票: {', '.join(symbols)}")
            else:
                print("❌ 错误: 未提供股票代码且默认股票组合不存在")
                print("请使用 --list-stocks 查看可用组合，或直接提供股票代码")
                return
        
        if not symbols:
            print("❌ 错误: 没有要分析的股票")
            return
        
        # 显示LLM配置信息
        if args.verbose:
            logger.info(f"📋 LLM配置信息:")
            logger.info(f"  - Base URL: {llm_config.get('base_url', '未设置')}")
            logger.info(f"  - API Key: {'已设置' if llm_config.get('api_key') or os.getenv(llm_config.get('api_key_env', '')) else '未设置'}")
            logger.info(f"  - 分析模型: {llm_config.get('models', {}).get('analysis_model', '未设置')}")
        
        # 创建分析配置
        # 从配置文件读取analysis_options和batch_settings
        analysis_options = config_data.get('analysis_options', {})
        batch_settings = config_data.get('batch_settings', {})
        # 将batch_settings合并到analysis_options中
        if batch_settings:
            analysis_options = {**analysis_options, 'batch_settings': batch_settings}
        
        # 从顶层获取email配置
        email_config = config_data.get('email', {})
        
        analysis_config = StockAnalysisConfig(
            symbols=symbols,
            output_dir=args.output,
            llm_config=llm_config,
            analysis_options=analysis_options,
            email_config=email_config
        )
        
        # 运行批量分析
        analyzer = BatchStockLLMAnalyzer(analysis_config)
        result = analyzer.run_batch_analysis()
        
        print(f"\n🎉 批量LLM股票分析完成!")
        print(f"📊 成功分析: {result.successful_analyses}/{result.total_symbols} 只股票")
        print(f"⏱️ 耗时: {result.duration:.2f}s")
        print(f"📄 结果保存在: {args.output}")
        
    except Exception as e:
        logger.error(f"❌ 批量分析失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
