import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sys

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config.config import config

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """查询类型枚举"""
    DRUG_SEARCH = "drug_search"
    EFFECT_ANALYSIS = "effect_analysis"
    COMPARISON = "comparison"
    RANKING = "ranking"
    ORGANISM_SPECIFIC = "organism_specific"
    MECHANISM = "mechanism"
    GENERAL = "general"

@dataclass
class QueryContext:
    """查询上下文数据类"""
    query_type: QueryType
    primary_intent: str
    entities: Dict[str, List[str]]
    parameters: Dict[str, Any]
    confidence: float
    suggestions: List[str]

class QueryAnalyzer:
    """自然语言查询分析器"""
    
    def __init__(self):
        """初始化查询分析器"""
        self.drug_patterns = self._build_drug_patterns()
        self.organism_patterns = self._build_organism_patterns()
        self.effect_patterns = self._build_effect_patterns()
        self.comparison_patterns = self._build_comparison_patterns()
        self.ranking_patterns = self._build_ranking_patterns()
        self.mechanism_patterns = self._build_mechanism_patterns()
        
    def _build_drug_patterns(self) -> List[Tuple[str, str]]:
        """构建药物识别模式"""
        return [
            (r'\b(rapamycin|sirolimus)\b', 'rapamycin'),
            (r'\b(metformin)\b', 'metformin'),
            (r'\b(resveratrol)\b', 'resveratrol'),
            (r'\b(aspirin|acetylsalicylic acid)\b', 'aspirin'),
            (r'\b(curcumin|turmeric)\b', 'curcumin'),
            (r'\b(lithium|lithium chloride)\b', 'lithium'),
            (r'\b(caffeine)\b', 'caffeine'),
            (r'\b(vitamin [a-z])\b', r'\1'),
            (r'\b(spermidine)\b', 'spermidine'),
            (r'\b(nicotinamide|nam|niacinamide)\b', 'nicotinamide'),
            (r'\b(quercetin)\b', 'quercetin'),
            (r'\b(green tea|egcg|epigallocatechin)\b', 'green tea'),
            (r'\b(caloric restriction|cr)\b', 'caloric restriction'),
            (r'\b(hydroxytyrosol)\b', 'hydroxytyrosol'),
            (r'\b(polyphenol)\b', 'polyphenol'),
            (r'\b(n-acetylcysteine|nac)\b', 'n-acetylcysteine'),
        ]
    
    def _build_organism_patterns(self) -> List[Tuple[str, str]]:
        """构建生物模型识别模式"""
        return [
            (r'\b(mouse|mice|mus musculus)\b', 'mouse'),
            (r'\b(rat|rats|rattus norvegicus)\b', 'rat'),
            (r'\b(worm|worms|c\.?\s*elegans|caenorhabditis elegans|nematode)\b', 'C. elegans'),
            (r'\b(fly|flies|drosophila|d\.?\s*melanogaster|fruit fly)\b', 'Drosophila'),
            (r'\b(yeast|s\.?\s*cerevisiae|saccharomyces)\b', 'yeast'),
            (r'\b(human|humans|homo sapiens)\b', 'human'),
            (r'\b(primate|primates|monkey|monkeys)\b', 'primate'),
            (r'\b(zebrafish|danio rerio)\b', 'zebrafish'),
            (r'\b(killifish)\b', 'killifish'),
        ]
    
    def _build_effect_patterns(self) -> List[str]:
        """构建效果分析模式"""
        return [
            r'\b(lifespan|life span|longevity|aging|ageing)\b',
            r'\b(extend|extension|increase|prolong|benefit|improve)\b',
            r'\b(effect|impact|influence|result|outcome)\b',
            r'\b(survival|mortality|death|live longer)\b',
            r'\b(percent|percentage|%|fold|times)\b',
            r'\b(healthspan|health span)\b',
        ]
    
    def _build_comparison_patterns(self) -> List[str]:
        """构建比较模式"""
        return [
            r'\b(compare|comparison|versus|vs\.?|against)\b',
            r'\b(better|worse|more effective|less effective)\b',
            r'\b(difference|differ|similar|same|alike)\b',
            r'\b(which is|what is the difference|which one)\b',
            r'\b(superior|inferior|outperform)\b',
        ]
    
    def _build_ranking_patterns(self) -> List[str]:
        """构建排名模式"""
        return [
            r'\b(best|top|most|highest|greatest|maximum)\b',
            r'\b(worst|bottom|least|lowest|smallest|minimum)\b',
            r'\b(rank|ranking|order|list|sort)\b',
            r'\b(first|second|third|\d+st|\d+nd|\d+rd|\d+th)\b',
            r'\b(leading|premier|superior)\b',
        ]
    
    def _build_mechanism_patterns(self) -> List[str]:
        """构建机制查询模式"""
        return [
            r'\b(how|why|mechanism|pathway|target)\b',
            r'\b(work|function|operate|act)\b',
            r'\b(molecular|cellular|biochemical)\b',
            r'\b(gene|protein|enzyme|receptor)\b',
            r'\b(signaling|signal|cascade)\b',
        ]
    
    def analyze_query(self, query: str) -> QueryContext:
        """
        分析自然语言查询
        
        Args:
            query: 用户查询文本
            
        Returns:
            QueryContext: 查询上下文对象
        """
        query_lower = query.lower().strip()
        
        # 提取实体
        entities = self._extract_entities(query_lower)
        
        # 确定查询类型
        query_type, confidence = self._determine_query_type(query_lower, entities)
        
        # 提取参数
        parameters = self._extract_parameters(query_lower, query_type, entities)
        
        # 确定主要意图
        primary_intent = self._determine_primary_intent(query_lower, query_type)
        
        # 生成建议
        suggestions = self._generate_suggestions(query_type, entities, parameters)
        
        return QueryContext(
            query_type=query_type,
            primary_intent=primary_intent,
            entities=entities,
            parameters=parameters,
            confidence=confidence,
            suggestions=suggestions
        )
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """提取查询中的实体"""
        entities = {
            'drugs': [],
            'organisms': [],
            'numbers': [],
            'effects': [],
            'time_periods': []
        }
        
        # 提取药物
        for pattern, drug_name in self.drug_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    # 处理带有捕获组的模式
                    entities['drugs'].extend([match for match in matches[0] if match])
                else:
                    entities['drugs'].extend([drug_name] * len(matches))
        
        # 提取生物模型
        for pattern, organism_name in self.organism_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                entities['organisms'].extend([organism_name] * len(matches))
        
        # 提取数字和数值
        number_patterns = [
            (r'\b(\d+(?:\.\d+)?)\s*(?:percent|%)\b', 'percentage'),
            (r'\b(\d+(?:\.\d+)?)\s*(?:fold|times)\b', 'fold_change'),
            (r'\b(\d+(?:\.\d+)?)\s*(?:year|years|yr)\b', 'years'),
            (r'\b(\d+(?:\.\d+)?)\s*(?:month|months|mo)\b', 'months'),
            (r'\btop\s*(\d+)\b', 'top_n'),
            (r'\bfirst\s*(\d+)\b', 'first_n'),
            (r'\b(\d+(?:\.\d+)?)\s*(?:mg|g|kg|ml|l)\b', 'dosage'),
        ]
        
        for pattern, number_type in number_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                entities['numbers'].append({
                    'value': float(match),
                    'type': number_type
                })
        
        # 提取效果相关词汇
        for pattern in self.effect_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                matches = re.findall(pattern, query, re.IGNORECASE)
                entities['effects'].extend(matches)
        
        # 去重
        entities['drugs'] = list(set(entities['drugs']))
        entities['organisms'] = list(set(entities['organisms']))
        entities['effects'] = list(set(entities['effects']))
        
        return entities
    
    def _determine_query_type(self, query: str, entities: Dict) -> Tuple[QueryType, float]:
        """确定查询类型和置信度"""
        scores = {QueryType.GENERAL: 0.1}  # 默认分数
        
        # 药物搜索
        if entities['drugs']:
            scores[QueryType.DRUG_SEARCH] = 0.6
            if any(word in query for word in ['information', 'about', 'tell me', 'what is', 'describe']):
                scores[QueryType.DRUG_SEARCH] += 0.3
        
        # 效果分析
        effect_score = 0
        for pattern in self.effect_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                effect_score += 0.2
        if effect_score > 0:
            scores[QueryType.EFFECT_ANALYSIS] = min(effect_score, 0.8)
        
        # 比较查询
        comparison_score = 0
        for pattern in self.comparison_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                comparison_score += 0.3
        
        if len(entities['drugs']) >= 2:
            comparison_score += 0.4
        
        if comparison_score > 0:
            scores[QueryType.COMPARISON] = min(comparison_score, 0.9)
        
        # 排名查询
        ranking_score = 0
        for pattern in self.ranking_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                ranking_score += 0.3
        
        # 检查是否有数量限制词
        if any(num_info['type'] in ['top_n', 'first_n'] for num_info in entities['numbers']):
            ranking_score += 0.2
        
        if ranking_score > 0:
            scores[QueryType.RANKING] = min(ranking_score, 0.8)
        
        # 生物模型特定查询
        if entities['organisms']:
            scores[QueryType.ORGANISM_SPECIFIC] = 0.5
            if any(word in query for word in ['in', 'on', 'for', 'using', 'with', 'tested']):
                scores[QueryType.ORGANISM_SPECIFIC] += 0.3
        
        # 机制查询
        mechanism_score = 0
        for pattern in self.mechanism_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                mechanism_score += 0.2
        
        if mechanism_score > 0:
            scores[QueryType.MECHANISM] = min(mechanism_score, 0.7)
        
        # 选择最高分的类型
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return best_type, min(confidence, 1.0)
    
    def _extract_parameters(self, query: str, query_type: QueryType, entities: Dict) -> Dict[str, Any]:
        """根据查询类型提取参数"""
        parameters = {}
        
        # 提取数量限制
        for num_info in [n for n in entities.get('numbers', []) if isinstance(n, dict)]:
            if num_info['type'] in ['top_n', 'first_n']:
                parameters['limit'] = int(num_info['value'])
            elif num_info['type'] == 'percentage':
                parameters['min_effect'] = num_info['value']
            elif num_info['type'] in ['years', 'months']:
                parameters['time_period'] = {
                    'value': num_info['value'],
                    'unit': num_info['type']
                }
            elif num_info['type'] == 'dosage':
                parameters['dosage'] = num_info['value']
        
        # 如果没有明确的数量限制，根据查询类型设置默认值
        if 'limit' not in parameters:
            if query_type == QueryType.RANKING:
                if re.search(r'\b(best|top|most)\b', query):
                    parameters['limit'] = 10  # 默认前10个
        
        # 提取最小效果阈值
        if 'min_effect' not in parameters:
            percent_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:percent|%)', query)
            if percent_match:
                parameters['min_effect'] = float(percent_match.group(1))
        
        # 提取比较参数
        if query_type == QueryType.COMPARISON:
            parameters['comparison_type'] = 'comprehensive'
            if 'statistical' in query or 'significant' in query or 'stats' in query:
                parameters['include_statistics'] = True
            if 'summary' in query or 'brief' in query:
                parameters['comparison_type'] = 'summary'
        
        # 提取排序参数
        if any(word in query for word in ['ascending', 'lowest', 'worst', 'smallest']):
            parameters['sort_order'] = 'ascending'
        else:
            parameters['sort_order'] = 'descending'
        
        # 提取搜索参数
        if query_type == QueryType.DRUG_SEARCH:
            if 'exact' in query or 'exactly' in query:
                parameters['exact_match'] = True
            if 'similar' in query or 'related' in query:
                parameters['include_similar'] = True
        
        return parameters
    
    def _determine_primary_intent(self, query: str, query_type: QueryType) -> str:
        """确定主要意图"""
        intent_map = {
            QueryType.DRUG_SEARCH: "查找特定药物的详细信息",
            QueryType.EFFECT_ANALYSIS: "分析化合物的寿命延长效果",
            QueryType.COMPARISON: "比较多个药物或治疗方法",
            QueryType.RANKING: "按效果对药物进行排名",
            QueryType.ORGANISM_SPECIFIC: "查找在特定生物模型中测试的药物",
            QueryType.MECHANISM: "了解药物的作用机制",
            QueryType.GENERAL: "获取关于长寿研究的一般信息"
        }
        
        return intent_map.get(query_type, "一般查询")
    
    def _generate_suggestions(self, 
                            query_type: QueryType, 
                            entities: Dict, 
                            parameters: Dict) -> List[str]:
        """生成查询改进建议"""
        suggestions = []
        
        # 基于查询类型的建议
        if query_type == QueryType.DRUG_SEARCH and not entities['drugs']:
            suggestions.append("尝试指定具体的药物名称，如'rapamycin'或'metformin'")
        
        if query_type == QueryType.COMPARISON and len(entities['drugs']) < 2:
            suggestions.append("比较查询需要至少两个药物名称")
        
        if query_type == QueryType.ORGANISM_SPECIFIC and not entities['organisms']:
            suggestions.append("请指定生物模型，如'小鼠'、'大鼠'或'C. elegans'")
        
        if query_type == QueryType.RANKING and 'limit' not in parameters:
            suggestions.append("考虑指定需要的结果数量（例如：'前10个'）")
        
        # 基于实体的建议
        if not entities['drugs'] and not entities['organisms']:
            suggestions.extend([
                "尝试更具体地描述感兴趣的药物或生物模型",
                "示例：'rapamycin在小鼠中的效果'或'比较metformin和resveratrol'"
            ])
        
        # 查询优化建议
        if query_type == QueryType.GENERAL:
            suggestions.extend([
                "尝试使用更具体的查询",
                "指定药物名称、生物模型或研究类型"
            ])
        
        return suggestions[:3]  # 限制建议数量
    
    def get_query_examples(self) -> Dict[str, List[str]]:
        """获取各类查询的示例"""
        return {
            "药物搜索": [
                "告诉我关于rapamycin的信息",
                "metformin对寿命有什么影响？",
                "resveratrol的研究结果如何？"
            ],
            "效果分析": [
                "哪些药物延长寿命超过20%？",
                "显示最有效的延寿化合物",
                "分析寿命延长效果显著的药物"
            ],
            "药物比较": [
                "比较rapamycin和metformin",
                "哪个更好：resveratrol还是curcumin？",
                "rapamycin vs metformin vs resveratrol的效果"
            ],
            "效果排名": [
                "前10个最有效的长寿药物",
                "延寿效果最好的化合物排名",
                "在小鼠中效果最佳的药物列表"
            ],
            "生物模型查询": [
                "在小鼠中测试的药物",
                "哪些化合物在C. elegans中有效？",
                "大鼠研究中的显著效果"
            ],
            "机制查询": [
                "rapamycin如何延长寿命？",
                "metformin的作用机制是什么？",
                "为什么resveratrol影响衰老？"
            ]
        }
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """验证查询的有效性和完整性"""
        validation = {
            'is_valid': True,
            'issues': [],
            'suggestions': [],
            'confidence': 1.0
        }
        
        if len(query.strip()) < 3:
            validation['is_valid'] = False
            validation['issues'].append("查询太短")
            validation['suggestions'].append("请提供更详细的查询")
        
        # 检查是否包含有意义的内容
        meaningful_words = ['drug', 'compound', 'effect', 'lifespan', 'longevity', 
                          'mouse', 'rat', 'compare', 'best', 'top']
        
        if not any(word in query.lower() for word in meaningful_words):
            validation['confidence'] -= 0.3
            validation['suggestions'].append("尝试包含药物名称或研究相关的关键词")
        
        # 检查查询的清晰度
        if '?' not in query and not any(word in query.lower() for word in 
                                       ['show', 'tell', 'find', 'compare', 'what', 'which', 'how']):
            validation['suggestions'].append("考虑将查询表述为问题形式")
        
        return validation

# 测试代码
if __name__ == "__main__":
    analyzer = QueryAnalyzer()
    
    # 测试查询
    test_queries = [
        "告诉我关于rapamycin在小鼠中的效果",
        "比较rapamycin和metformin",
        "效果最好的前10种延寿药物是什么？",
        "哪些药物在C. elegans中有效？",
        "resveratrol如何延长寿命？",
        "显示延寿效果超过20%的药物"
    ]
    
    print("=== 查询分析测试 ===")
    for query in test_queries:
        print(f"\n查询: {query}")
        context = analyzer.analyze_query(query)
        print(f"类型: {context.query_type.value}")
        print(f"意图: {context.primary_intent}")
        print(f"置信度: {context.confidence:.2f}")
        print(f"实体: {context.entities}")
        print(f"参数: {context.parameters}")
        if context.suggestions:
            print(f"建议: {context.suggestions}")
    
    print("\n=== 查询示例 ===")
    examples = analyzer.get_query_examples()
    for category, example_list in examples.items():
        print(f"\n{category}:")
        for example in example_list:
            print(f"  - {example}")