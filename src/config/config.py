import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

class Config:
    """项目配置类"""
    
    # API配置
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '1500'))
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.3'))
    
    # 路径配置
    DATA_DIR = PROJECT_ROOT / 'data'
    DATA_PATH = DATA_DIR / 'drugage.csv'
    PROCESSED_DATA_PATH = DATA_DIR / 'processed_drugage.csv'
    CACHE_DIR = DATA_DIR / 'cache'
    
    # 应用配置
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Streamlit配置
    PAGE_TITLE = "DrugAge智能助手"
    PAGE_ICON = "🧬"
    LAYOUT = "wide"
    
    # DrugAge数据列映射
    COLUMN_MAPPING = {
        'compound': ['compound', 'drug_name', 'drug', 'substance', 'name'],
        'organism': ['organism', 'species', 'model_organism', 'model'],
        'lifespan_effect': ['mean_lifespan_change', 'mean', 'effect', 'lifespan_change', 
                           'lifespan_effect', 'percent_change'],
        'max_effect': ['max_lifespan_change', 'max_effect', 'maximum', 'max'],
        'gender': ['gender', 'sex'],
        'dosage': ['dosage', 'dose', 'concentration', 'amount'],
        'strain': ['strain', 'genetic_background', 'background'],
        'reference': ['reference', 'pubmed_id', 'study', 'pmid']
    }
    
    # 支持的生物模型
    SUPPORTED_ORGANISMS = [
        'Mouse', 'Rat', 'Caenorhabditis elegans', 'Drosophila melanogaster',
        'Saccharomyces cerevisiae', 'Mus musculus', 'Rattus norvegicus',
        'C. elegans', 'Drosophila', 'Yeast'
    ]
    
    # 查询类型关键词
    QUERY_KEYWORDS = {
        'drug_search': ['search', 'find', 'information about', 'tell me about', 'what is'],
        'effect_analysis': ['effect', 'impact', 'influence', 'lifespan', 'longevity', 'extend'],
        'comparison': ['compare', 'versus', 'vs', 'difference', 'better', 'which'],
        'ranking': ['best', 'top', 'most effective', 'highest', 'rank', 'list'],
        'organism_specific': ['mouse', 'mice', 'rat', 'worm', 'fly', 'yeast', 'in', 'on'],
        'mechanism': ['how', 'why', 'mechanism', 'pathway', 'target', 'work']
    }
    
    # GPT系统提示模板
    SYSTEM_PROMPTS = {
        'general': """你是DrugAge智能助手，专门帮助研究人员查询和分析与延长寿命相关的化合物数据。

你的职责：
1. 准确解释DrugAge数据库中的科学数据
2. 帮助研究人员理解不同化合物对寿命的影响
3. 提供基于证据的信息，并指出数据的局限性
4. 使用清晰、科学但易懂的语言回答问题
5. 当不确定时，诚实说明并建议查看原始研究

重要注意事项：
- 始终基于提供的数据回答问题
- 不要提供医疗建议
- 强调这些是实验室研究结果，不是临床建议
- 鼓励用户查看原始文献以获得完整信息""",
        
        'drug_analysis': """你正在分析DrugAge数据库中的特定药物数据。
重点关注：
- 寿命延长效果和统计显著性
- 不同模型生物上的表现
- 剂量信息和实验条件
- 与其他类似化合物的比较
- 研究的局限性和注意事项""",
        
        'comparison': """你正在比较多个药物的延寿效果。
请提供：
- 并排效果比较
- 特定生物模型中的表现
- 机制差异（如已知）
- 研究质量评估
- 进一步研究的建议"""
    }
    
    # 可视化配置
    PLOT_CONFIG = {
        'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'figure_size': (10, 6),
        'dpi': 100,
        'style': 'plotly_white'
    }
    
    @classmethod
    def get_data_columns(cls, data_df) -> Dict[str, str]:
        """动态映射数据列名"""
        column_map = {}
        available_columns = [col.lower().strip() for col in data_df.columns]
        
        for standard_name, possible_names in cls.COLUMN_MAPPING.items():
            for possible_name in possible_names:
                if possible_name.lower() in available_columns:
                    actual_column = data_df.columns[available_columns.index(possible_name.lower())]
                    column_map[standard_name] = actual_column
                    break
        
        return column_map
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """验证配置的有效性"""
        errors = []
        
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        
        # 创建必要的目录
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.CACHE_DIR.mkdir(exist_ok=True)
        
        return errors
    
    @classmethod
    def get_streamlit_config(cls) -> Dict:
        """获取Streamlit配置"""
        return {
            'page_title': cls.PAGE_TITLE,
            'page_icon': cls.PAGE_ICON,
            'layout': cls.LAYOUT,
            'initial_sidebar_state': 'expanded'
        }

# 全局配置实例
config = Config()