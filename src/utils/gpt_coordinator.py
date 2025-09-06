import os
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import openai
from pathlib import Path
import sys
import pandas as pd

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config.config import config

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """查询结果数据类"""
    success: bool
    data: Any = None
    error_message: str = ""
    suggestions: List[str] = None

class GPTCoordinator:
    """GPT协调器 - 负责理解查询并协调工具调用"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化GPT协调器
        
        Args:
            api_key: OpenAI API密钥
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API密钥未配置")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = config.OPENAI_MODEL
        
        # 注册的工具
        self.tools = {}
        
        # 对话历史
        self.conversation_history = []
        
    def register_tool(self, name: str, tool_function, description: str):
        """
        注册工具函数
        
        Args:
            name: 工具名称
            tool_function: 工具函数
            description: 工具描述
        """
        self.tools[name] = {
            'function': tool_function,
            'description': description
        }
        logger.info(f"工具已注册: {name}")
    
    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        """
        分析用户查询，确定需要调用的工具和参数
        
        Args:
            user_query: 用户查询
            
        Returns:
            查询分析结果
        """
        # 简化的查询分析逻辑
        query_lower = user_query.lower()
        
        analysis = {
            'query': user_query,
            'intent': 'general',
            'entities': {
                'drugs': [],
                'organisms': [],
                'numbers': []
            },
            'tools_needed': [],
            'confidence': 0.5
        }
        
        # 药物实体识别
        drug_keywords = ['rapamycin', 'metformin', 'resveratrol', 'aspirin', 'curcumin', 
                        'lithium', 'caffeine', 'spermidine', 'nicotinamide', 'quercetin']
        
        for drug in drug_keywords:
            if drug in query_lower:
                analysis['entities']['drugs'].append(drug)
        
        # 生物模型识别
        organism_keywords = ['mouse', 'mice', 'rat', 'worm', 'fly', 'yeast', 'human', 
                           'c. elegans', 'drosophila']
        
        for organism in organism_keywords:
            if organism in query_lower:
                analysis['entities']['organisms'].append(organism)
        
        # 数字提取
        import re
        numbers = re.findall(r'\d+', query_lower)
        analysis['entities']['numbers'] = [int(n) for n in numbers]
        
        # 意图识别
        if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference']):
            analysis['intent'] = 'comparison'
            analysis['tools_needed'].append('compare_drugs')
        elif any(word in query_lower for word in ['search', 'find', 'information', 'about', 'tell me']):
            analysis['intent'] = 'search'
            analysis['tools_needed'].append('search_drug')
        elif any(word in query_lower for word in ['best', 'top', 'most', 'rank']):
            analysis['intent'] = 'ranking'
            analysis['tools_needed'].append('get_top_drugs')
        elif any(word in query_lower for word in ['effect', 'impact', 'influence']):
            analysis['intent'] = 'analysis'
            analysis['tools_needed'].append('analyze_effects')
        
        # 调整置信度
        if analysis['entities']['drugs'] or analysis['entities']['organisms']:
            analysis['confidence'] += 0.3
        
        if analysis['tools_needed']:
            analysis['confidence'] += 0.2
        
        analysis['confidence'] = min(analysis['confidence'], 1.0)
        
        return analysis
    
    def execute_tools(self, analysis: Dict[str, Any]) -> List[QueryResult]:
        """
        执行所需的工具
        
        Args:
            analysis: 查询分析结果
            
        Returns:
            工具执行结果列表
        """
        results = []
        
        for tool_name in analysis['tools_needed']:
            if tool_name in self.tools:
                try:
                    # 根据工具类型准备参数
                    if tool_name == 'search_drug' and analysis['entities']['drugs']:
                        for drug in analysis['entities']['drugs']:
                            result = self.tools[tool_name]['function'](drug)
                            results.append(QueryResult(
                                success=True,
                                data=result,
                                suggestions=[f"找到关于 {drug} 的信息"]
                            ))
                    
                    elif tool_name == 'compare_drugs' and len(analysis['entities']['drugs']) >= 2:
                        result = self.tools[tool_name]['function'](analysis['entities']['drugs'])
                        results.append(QueryResult(
                            success=True,
                            data=result,
                            suggestions=[f"比较了 {', '.join(analysis['entities']['drugs'])} 的效果"]
                        ))
                    
                    elif tool_name == 'get_top_drugs':
                        n = 10
                        if analysis['entities']['numbers']:
                            n = analysis['entities']['numbers'][0]
                        
                        organism = analysis['entities']['organisms'][0] if analysis['entities']['organisms'] else None
                        result = self.tools[tool_name]['function'](n=n, organism=organism)
                        results.append(QueryResult(
                            success=True,
                            data=result,
                            suggestions=["获取了效果最好的药物列表"]
                        ))
                    
                    elif tool_name == 'analyze_effects':
                        result = self.tools[tool_name]['function'](min_effect=0)
                        results.append(QueryResult(
                            success=True,
                            data=result,
                            suggestions=["分析了药物效果数据"]
                        ))
                
                except Exception as e:
                    logger.error(f"工具执行失败 {tool_name}: {e}")
                    results.append(QueryResult(
                        success=False,
                        error_message=str(e)
                    ))
            else:
                results.append(QueryResult(
                    success=False,
                    error_message=f"工具 {tool_name} 未注册"
                ))
        
        return results
    
    def generate_response(self, 
                         user_query: str, 
                         tool_results: List[QueryResult],
                         analysis: Dict[str, Any]) -> str:
        """
        基于工具结果生成最终响应
        
        Args:
            user_query: 用户查询
            tool_results: 工具执行结果
            analysis: 查询分析结果
            
        Returns:
            GPT生成的响应
        """
        # 构建上下文
        context_parts = [
            f"用户查询: {user_query}",
            f"查询意图: {analysis['intent']}",
            f"识别实体: {analysis['entities']}"
        ]
        
        # 添加工具结果
        if tool_results:
            context_parts.append("工具执行结果:")
            for i, result in enumerate(tool_results):
                if result.success:
                    context_parts.append(f"{i+1}. 成功获取数据")
                    if hasattr(result.data, '__len__') and not isinstance(result.data, str):
                        if isinstance(result.data, pd.DataFrame):
                            context_parts.append(f"   数据量: {len(result.data)} 条记录")
                        elif isinstance(result.data, dict):
                            context_parts.append(f"   数据类型: 字典，包含 {len(result.data)} 个项目")
                else:
                    context_parts.append(f"{i+1}. 执行失败: {result.error_message}")
        
        context = "\n".join(context_parts)
        
        # 选择合适的系统提示
        if analysis['intent'] == 'comparison':
            system_prompt = config.SYSTEM_PROMPTS['comparison']
        elif analysis['intent'] in ['search', 'analysis']:
            system_prompt = config.SYSTEM_PROMPTS['drug_analysis']
        else:
            system_prompt = config.SYSTEM_PROMPTS['general']
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"GPT响应生成失败: {e}")
            return f"抱歉，生成响应时出现错误。请稍后重试。错误信息: {str(e)}"
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        处理完整的用户查询流程
        
        Args:
            user_query: 用户查询
            
        Returns:
            完整的处理结果
        """
        # 1. 分析查询
        analysis = self.analyze_query(user_query)
        
        # 2. 执行工具
        tool_results = self.execute_tools(analysis)
        
        # 3. 生成响应
        response = self.generate_response(user_query, tool_results, analysis)
        
        # 4. 保存到历史
        result = {
            'query': user_query,
            'analysis': analysis,
            'tool_results': tool_results,
            'response': response,
            'timestamp': pd.Timestamp.now()
        }
        
        self.conversation_history.append(result)
        
        return result
    
    def get_suggestions(self, partial_query: str) -> List[str]:
        """
        基于部分查询生成建议
        
        Args:
            partial_query: 部分查询
            
        Returns:
            建议列表
        """
        suggestions = []
        partial_lower = partial_query.lower()
        
        if 'compare' in partial_lower:
            suggestions.extend([
                "比较rapamycin和metformin的效果",
                "比较在小鼠中测试的前三种药物"
            ])
        elif any(word in partial_lower for word in ['best', 'top']):
            suggestions.extend([
                "效果最好的10种延寿药物",
                "在C. elegans中表现最佳的化合物"
            ])
        elif any(drug in partial_lower for drug in ['rapamycin', 'metformin']):
            suggestions.extend([
                f"关于{partial_query}的详细信息",
                f"{partial_query}在不同生物模型中的效果"
            ])
        else:
            suggestions.extend([
                "告诉我关于rapamycin的信息",
                "比较metformin和resveratrol",
                "显示延寿效果最好的药物"
            ])
        
        return suggestions[:5]
    
    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []
        logger.info("对话历史已清除")

# 测试代码
if __name__ == "__main__":
    try:
        coordinator = GPTCoordinator()
        
        # 测试查询分析
        test_query = "比较rapamycin和metformin的效果"
        analysis = coordinator.analyze_query(test_query)
        
        print("=== 查询分析测试 ===")
        print(f"查询: {analysis['query']}")
        print(f"意图: {analysis['intent']}")
        print(f"实体: {analysis['entities']}")
        print(f"需要工具: {analysis['tools_needed']}")
        print(f"置信度: {analysis['confidence']:.2f}")
        
        print("\nGPTCoordinator初始化成功")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()