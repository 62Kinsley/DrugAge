import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import pickle
import sys

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config.config import config

logger = logging.getLogger(__name__)

class DrugAgeDataProcessor:
    """DrugAge数据的智能处理器"""
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        初始化数据处理器
        
        Args:
            csv_path: CSV文件路径，默认使用配置中的路径
        """
        self.csv_path = Path(csv_path) if csv_path else config.DATA_PATH
        self.data = None
        self.column_map = {}
        self.processed_data = None
        self.cache_file = config.CACHE_DIR / "processor_cache.pkl"
        
        # 确保缓存目录存在
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        self._load_data()
        self._map_columns()
        self._preprocess_data()
        
    def _load_data(self):
        """加载DrugAge数据"""
        try:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"数据文件未找到: {self.csv_path}")
            
            self.data = pd.read_csv(self.csv_path)
            logger.info(f"数据加载成功: {len(self.data)} 条记录")
            
            # 清理列名
            self.data.columns = self.data.columns.str.strip()
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
    
    def _map_columns(self):
        """映射数据列名到标准格式"""
        self.column_map = config.get_data_columns(self.data)
        logger.info(f"列映射完成: {self.column_map}")
    
    def _preprocess_data(self):
        """预处理数据"""
        self.processed_data = self.data.copy()
        
        # 标准化化合物名称
        if 'compound' in self.column_map:
            compound_col = self.column_map['compound']
            self.processed_data[compound_col] = (
                self.processed_data[compound_col]
                .astype(str)
                .str.strip()
                .str.lower()
            )
        
        # 标准化生物模型名称
        if 'organism' in self.column_map:
            organism_col = self.column_map['organism']
            self.processed_data[organism_col] = (
                self.processed_data[organism_col]
                .astype(str)
                .str.strip()
            )
        
        # 处理数值列
        if 'lifespan_effect' in self.column_map:
            effect_col = self.column_map['lifespan_effect']
            self.processed_data[effect_col] = pd.to_numeric(
                self.processed_data[effect_col], errors='coerce'
            )
        
        logger.info("数据预处理完成")
    
    def search_drugs(self, 
                    query: str, 
                    exact_match: bool = False,
                    min_effect: Optional[float] = None) -> pd.DataFrame:
        """
        智能药物搜索
        
        Args:
            query: 搜索查询
            exact_match: 是否精确匹配
            min_effect: 最小效果阈值
            
        Returns:
            匹配的药物数据
        """
        if 'compound' not in self.column_map:
            logger.warning("未找到化合物列")
            return pd.DataFrame()
        
        compound_col = self.column_map['compound']
        query_lower = str(query).lower().strip()
        
        if exact_match:
            mask = self.processed_data[compound_col] == query_lower
        else:
            # 模糊匹配
            mask = self.processed_data[compound_col].str.contains(
                query_lower, case=False, na=False, regex=False
            )
        
        results = self.processed_data[mask].copy()
        
        # 应用效果过滤
        if min_effect is not None and 'lifespan_effect' in self.column_map:
            effect_col = self.column_map['lifespan_effect']
            results = results[
                pd.to_numeric(results[effect_col], errors='coerce') >= min_effect
            ]
        
        # 按效果排序
        if 'lifespan_effect' in self.column_map and len(results) > 0:
            effect_col = self.column_map['lifespan_effect']
            # 转换为数值并排序
            results = results.copy()
            results['_effect_numeric'] = pd.to_numeric(results[effect_col], errors='coerce')
            results = results.sort_values('_effect_numeric', ascending=False, na_last=True)
            results = results.drop('_effect_numeric', axis=1)
        
        logger.info(f"药物搜索 '{query}': 找到 {len(results)} 条记录")
        return results
    
    def get_top_drugs(self, 
                     n: int = 10, 
                     organism: Optional[str] = None,
                     min_effect: float = 0) -> pd.DataFrame:
        """
        获取效果最好的药物
        
        Args:
            n: 返回数量
            organism: 特定生物模型
            min_effect: 最小效果值
            
        Returns:
            排序后的药物数据
        """
        if 'lifespan_effect' not in self.column_map:
            logger.warning("未找到寿命效果列")
            return pd.DataFrame()
        
        effect_col = self.column_map['lifespan_effect']
        data = self.processed_data.copy()
        
        # 过滤有效数据
        data['_effect_numeric'] = pd.to_numeric(data[effect_col], errors='coerce')
        data = data.dropna(subset=['_effect_numeric'])
        data = data[data['_effect_numeric'] >= min_effect]
        
        # 按生物模型过滤
        if organism and 'organism' in self.column_map:
            organism_col = self.column_map['organism']
            data = data[data[organism_col].str.contains(
                organism, case=False, na=False
            )]
        
        # 排序并返回前N个
        top_drugs = data.nlargest(n, '_effect_numeric')
        top_drugs = top_drugs.drop('_effect_numeric', axis=1)
        
        logger.info(f"获取前{n}个药物，条件: organism={organism}, min_effect={min_effect}")
        return top_drugs
    
    def compare_drugs(self, drug_names: List[str]) -> Dict[str, pd.DataFrame]:
        """
        比较多个药物
        
        Args:
            drug_names: 药物名称列表
            
        Returns:
            每个药物的数据字典
        """
        comparison_results = {}
        
        for drug_name in drug_names:
            drug_data = self.search_drugs(drug_name)
            if len(drug_data) > 0:
                comparison_results[drug_name] = drug_data
        
        logger.info(f"药物比较: {list(comparison_results.keys())}")
        return comparison_results
    
    def analyze_by_organism(self, organism: str) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        按生物模型分析药物
        
        Args:
            organism: 生物模型名称
            
        Returns:
            分析结果字典
        """
        if 'organism' not in self.column_map:
            return {'data': pd.DataFrame(), 'stats': {}}
        
        organism_col = self.column_map['organism']
        
        # 筛选数据
        organism_data = self.processed_data[
            self.processed_data[organism_col].str.contains(
                organism, case=False, na=False
            )
        ]
        
        # 计算统计信息
        stats = {'total_studies': len(organism_data)}
        
        if 'compound' in self.column_map:
            compound_col = self.column_map['compound']
            stats['unique_compounds'] = organism_data[compound_col].nunique()
        
        if 'lifespan_effect' in self.column_map:
            effect_col = self.column_map['lifespan_effect']
            valid_effects = pd.to_numeric(organism_data[effect_col], errors='coerce').dropna()
            
            if len(valid_effects) > 0:
                stats.update({
                    'mean_effect': float(valid_effects.mean()),
                    'median_effect': float(valid_effects.median()),
                    'std_effect': float(valid_effects.std()),
                    'max_effect': float(valid_effects.max()),
                    'min_effect': float(valid_effects.min()),
                    'positive_effects': int(len(valid_effects[valid_effects > 0])),
                    'negative_effects': int(len(valid_effects[valid_effects < 0]))
                })
        
        return {
            'data': organism_data,
            'stats': stats
        }
    
    def get_organisms_list(self) -> List[str]:
        """获取所有生物模型的列表"""
        if 'organism' not in self.column_map:
            return []
        
        organism_col = self.column_map['organism']
        return self.processed_data[organism_col].dropna().unique().tolist()
    
    def generate_summary_stats(self) -> Dict[str, any]:
        """生成数据集总结统计"""
        stats = {
            'total_records': len(self.data),
            'data_columns': list(self.data.columns),
            'mapped_columns': self.column_map
        }
        
        # 化合物统计
        if 'compound' in self.column_map:
            compound_col = self.column_map['compound']
            stats['unique_compounds'] = self.data[compound_col].nunique()
            stats['top_studied_compounds'] = (
                self.data[compound_col].value_counts().head(10).to_dict()
            )
        
        # 生物模型统计
        if 'organism' in self.column_map:
            organism_col = self.column_map['organism']
            stats['unique_organisms'] = self.data[organism_col].nunique()
            stats['organism_distribution'] = (
                self.data[organism_col].value_counts().to_dict()
            )
        
        # 效果统计
        if 'lifespan_effect' in self.column_map:
            effect_col = self.column_map['lifespan_effect']
            valid_effects = pd.to_numeric(self.data[effect_col], errors='coerce').dropna()
            
            if len(valid_effects) > 0:
                stats['effect_statistics'] = {
                    'mean': float(valid_effects.mean()),
                    'median': float(valid_effects.median()),
                    'std': float(valid_effects.std()),
                    'min': float(valid_effects.min()),
                    'max': float(valid_effects.max()),
                    'positive_effects': int(len(valid_effects[valid_effects > 0])),
                    'negative_effects': int(len(valid_effects[valid_effects < 0])),
                    'zero_effects': int(len(valid_effects[valid_effects == 0]))
                }
        
        return stats
    
    def save_cache(self):
        """保存处理结果到缓存"""
        cache_data = {
            'processed_data': self.processed_data,
            'column_map': self.column_map
        }
        
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info("缓存保存成功")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def load_cache(self) -> bool:
        """从缓存加载处理结果"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.processed_data = cache_data['processed_data']
                self.column_map = cache_data['column_map']
                logger.info("缓存加载成功")
                return True
        except Exception as e:
            logger.warning(f"缓存加载失败: {e}")
        
        return False

# 测试代码
if __name__ == "__main__":
    try:
        processor = DrugAgeDataProcessor()
        
        # 基本统计
        print("=== 数据概览 ===")
        stats = processor.generate_summary_stats()
        print(f"总记录数: {stats['total_records']}")
        print(f"唯一化合物: {stats.get('unique_compounds', 'N/A')}")
        print(f"唯一生物模型: {stats.get('unique_organisms', 'N/A')}")
        
        # 测试搜索
        print("\n=== 搜索测试 ===")
        rapamycin_data = processor.search_drugs("rapamycin")
        print(f"Rapamycin搜索结果: {len(rapamycin_data)} 条记录")
        
        if len(rapamycin_data) > 0:
            print("前3条记录:")
            print(rapamycin_data.head(3))
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()