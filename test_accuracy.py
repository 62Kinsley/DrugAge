#!/usr/bin/env python3
"""
同义词匹配系统准确率测试
测试同义词匹配系统能提高多少查询准确率
"""

import pandas as pd
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple
import random

class AccuracyTester:
    def __init__(self, csv_path: str):
        """初始化测试器"""
        self.data = pd.read_csv(csv_path)
        self.drug_names = set(self.data['compound_name'].str.lower().unique())
        self.organism_names = set(self.data['species'].str.lower().unique())
        
        # 构建同义词词典
        self.drug_synonyms = self._build_drug_synonyms()
        self.organism_synonyms = self._build_organism_synonyms()
        
        # 测试用例
        self.test_cases = self._create_test_cases()
        
    def _build_drug_synonyms(self) -> Dict[str, str]:
        """构建药物同义词词典"""
        return {
            'sirolimus': 'rapamycin',
            'glucophage': 'metformin',
            'dimethylbiguanide': 'metformin',
            'tocopherol': 'vitamin e',
            'alpha-tocopherol': 'vitamin e',
            'ascorbic acid': 'vitamin c',
            'l-ascorbic acid': 'vitamin c',
            'n-acetyl-l-cysteine': 'n-acetyl-l-cysteine',  # 数据库中就是这个名称
            'nac': 'n-acetyl-l-cysteine',
            'acetylcysteine': 'n-acetyl-l-cysteine',
            'egcg': 'epigallocatechin-3-gallate',
            'epigallocatechin gallate': 'epigallocatechin-3-gallate',
            'green tea extract': 'green tea extract',  # 数据库中就是这个名称
            'gtee': 'green tea extract',
            'catechin': 'epigallocatechin-3-gallate',
            'curcuma longa': 'curcumin',
            'turmeric': 'curcumin',
            'diferuloylmethane': 'curcumin',
            'quercetol': 'quercetin',
            'sophoretin': 'quercetin',
            'meletin': 'quercetin',
            'caf': 'caffeine',
            'theine': 'caffeine',
            'coffee': 'caffeine',
            'asp': 'aspirin',
            'asa': 'aspirin',
            'acetylsalicylic acid': 'aspirin',
            'spd': 'spermidine',
            'nam': 'nicotinamide',
            'niacinamide': 'nicotinamide',
            'vitamin b3': 'nicotinamide',
            'pyridine-3-carboxamide': 'nicotinamide',
            'ht': 'hydroxytyrosol',
            '3,4-dihydroxyphenylethanol': 'hydroxytyrosol',
            'dopet': 'hydroxytyrosol',
            'licl': 'lithium',
            'lithium chloride': 'lithium',
            'lithium carbonate': 'lithium',
            'coq10': 'coenzyme q10',
            'ubiquinone': 'coenzyme q10',
            'ubiquinol': 'coenzyme q10',
            'fish oil': 'omega-3',
            'epa': 'omega-3',
            'dha': 'omega-3',
            'eicosapentaenoic acid': 'omega-3',
            'docosahexaenoic acid': 'omega-3',
            'mt': 'melatonin',
            'n-acetyl-5-methoxytryptamine': 'melatonin',
            'mel': 'melatonin'
        }
    
    def _build_organism_synonyms(self) -> Dict[str, str]:
        """构建生物模型同义词词典"""
        return {
            'mouse': 'mus musculus',
            'mice': 'mus musculus',
            'house mouse': 'mus musculus',
            'laboratory mouse': 'mus musculus',
            'c57bl/6': 'mus musculus',
            'balb/c': 'mus musculus',
            'dba/2': 'mus musculus',
            'c3h': 'mus musculus',
            'fvb': 'mus musculus',
            'nude mouse': 'mus musculus',
            'rat': 'rattus norvegicus',
            'rats': 'rattus norvegicus',
            'norway rat': 'rattus norvegicus',
            'brown rat': 'rattus norvegicus',
            'wistar': 'rattus norvegicus',
            'sprague-dawley': 'rattus norvegicus',
            'fischer 344': 'rattus norvegicus',
            'lewis rat': 'rattus norvegicus',
            'c. elegans': 'caenorhabditis elegans',
            'c elegans': 'caenorhabditis elegans',
            'c.elegans': 'caenorhabditis elegans',
            'worm': 'caenorhabditis elegans',
            'worms': 'caenorhabditis elegans',
            'nematode': 'caenorhabditis elegans',
            'roundworm': 'caenorhabditis elegans',
            'elegans': 'caenorhabditis elegans',
            'drosophila': 'drosophila melanogaster',
            'd. melanogaster': 'drosophila melanogaster',
            'd.melanogaster': 'drosophila melanogaster',
            'fruit fly': 'drosophila melanogaster',
            'fly': 'drosophila melanogaster',
            'flies': 'drosophila melanogaster',
            'vinegar fly': 'drosophila melanogaster',
            'yeast': 'saccharomyces cerevisiae',
            's. cerevisiae': 'saccharomyces cerevisiae',
            's.cerevisiae': 'saccharomyces cerevisiae',
            'baker\'s yeast': 'saccharomyces cerevisiae',
            'budding yeast': 'saccharomyces cerevisiae',
            'saccharomyces': 'saccharomyces cerevisiae',
            'human': 'homo sapiens',
            'humans': 'homo sapiens',
            'h. sapiens': 'homo sapiens',
            'h.sapiens': 'homo sapiens',
            'human cells': 'homo sapiens',
            'human tissue': 'homo sapiens',
            'clinical': 'homo sapiens',
            'zebrafish': 'danio rerio',
            'zebra fish': 'danio rerio',
            'zebra-fish': 'danio rerio',
            'd. rerio': 'danio rerio',
            'd.rerio': 'danio rerio',
            'killifish': 'nothobranchius guentheri',
            'n. guentheri': 'nothobranchius guentheri',
            'n.guentheri': 'nothobranchius guentheri',
            'annual fish': 'nothobranchius guentheri',
            'turquoise killifish': 'nothobranchius guentheri'
        }
    
    def _create_test_cases(self) -> List[Dict]:
        """创建测试用例"""
        return [
            # 药物查询测试用例
            {'type': 'drug', 'query': 'sirolimus', 'expected': 'rapamycin', 'category': 'synonym'},
            {'type': 'drug', 'query': 'glucophage', 'expected': 'metformin', 'category': 'synonym'},
            {'type': 'drug', 'query': 'tocopherol', 'expected': 'vitamin e', 'category': 'synonym'},
            {'type': 'drug', 'query': 'ascorbic acid', 'expected': 'vitamin c', 'category': 'synonym'},
            {'type': 'drug', 'query': 'nac', 'expected': 'n-acetyl-l-cysteine', 'category': 'synonym'},
            {'type': 'drug', 'query': 'egcg', 'expected': 'epigallocatechin-3-gallate', 'category': 'synonym'},
            {'type': 'drug', 'query': 'turmeric', 'expected': 'curcumin', 'category': 'synonym'},
            {'type': 'drug', 'query': 'quercetol', 'expected': 'quercetin', 'category': 'synonym'},
            {'type': 'drug', 'query': 'caf', 'expected': 'caffeine', 'category': 'synonym'},
            {'type': 'drug', 'query': 'asp', 'expected': 'aspirin', 'category': 'synonym'},
            {'type': 'drug', 'query': 'spd', 'expected': 'spermidine', 'category': 'synonym'},
            {'type': 'drug', 'query': 'nam', 'expected': 'nicotinamide', 'category': 'synonym'},
            {'type': 'drug', 'query': 'ht', 'expected': 'hydroxytyrosol', 'category': 'synonym'},
            {'type': 'drug', 'query': 'licl', 'expected': 'lithium', 'category': 'synonym'},
            {'type': 'drug', 'query': 'coq10', 'expected': 'coenzyme q10', 'category': 'synonym'},
            {'type': 'drug', 'query': 'fish oil', 'expected': 'omega-3', 'category': 'synonym'},
            {'type': 'drug', 'query': 'mt', 'expected': 'melatonin', 'category': 'synonym'},
            
            # 生物模型查询测试用例
            {'type': 'organism', 'query': 'mouse', 'expected': 'mus musculus', 'category': 'synonym'},
            {'type': 'organism', 'query': 'mice', 'expected': 'mus musculus', 'category': 'synonym'},
            {'type': 'organism', 'query': 'rat', 'expected': 'rattus norvegicus', 'category': 'synonym'},
            {'type': 'organism', 'query': 'rats', 'expected': 'rattus norvegicus', 'category': 'synonym'},
            {'type': 'organism', 'query': 'c. elegans', 'expected': 'caenorhabditis elegans', 'category': 'synonym'},
            {'type': 'organism', 'query': 'worm', 'expected': 'caenorhabditis elegans', 'category': 'synonym'},
            {'type': 'organism', 'query': 'drosophila', 'expected': 'drosophila melanogaster', 'category': 'synonym'},
            {'type': 'organism', 'query': 'fruit fly', 'expected': 'drosophila melanogaster', 'category': 'synonym'},
            {'type': 'organism', 'query': 'yeast', 'expected': 'saccharomyces cerevisiae', 'category': 'synonym'},
            {'type': 'organism', 'query': 'human', 'expected': 'homo sapiens', 'category': 'synonym'},
            {'type': 'organism', 'query': 'zebrafish', 'expected': 'danio rerio', 'category': 'synonym'},
            {'type': 'organism', 'query': 'killifish', 'expected': 'nothobranchius guentheri', 'category': 'synonym'},
            
            # 精确匹配测试用例（应该都能找到）
            {'type': 'drug', 'query': 'rapamycin', 'expected': 'rapamycin', 'category': 'exact'},
            {'type': 'drug', 'query': 'metformin', 'expected': 'metformin', 'category': 'exact'},
            {'type': 'drug', 'query': 'resveratrol', 'expected': 'resveratrol', 'category': 'exact'},
            {'type': 'drug', 'query': 'curcumin', 'expected': 'curcumin', 'category': 'exact'},
            {'type': 'drug', 'query': 'caffeine', 'expected': 'caffeine', 'category': 'exact'},
            {'type': 'drug', 'query': 'aspirin', 'expected': 'aspirin', 'category': 'exact'},
            {'type': 'drug', 'query': 'quercetin', 'expected': 'quercetin', 'category': 'exact'},
            {'type': 'drug', 'query': 'spermidine', 'expected': 'spermidine', 'category': 'exact'},
            {'type': 'drug', 'query': 'nicotinamide', 'expected': 'nicotinamide', 'category': 'exact'},
            {'type': 'drug', 'query': 'hydroxytyrosol', 'expected': 'hydroxytyrosol', 'category': 'exact'},
            {'type': 'drug', 'query': 'lithium', 'expected': 'lithium', 'category': 'exact'},
            {'type': 'drug', 'query': 'coenzyme q10', 'expected': 'coenzyme q10', 'category': 'exact'},
            {'type': 'drug', 'query': 'omega-3', 'expected': 'omega-3', 'category': 'exact'},
            {'type': 'drug', 'query': 'melatonin', 'expected': 'melatonin', 'category': 'exact'},
            
            # 生物模型精确匹配
            {'type': 'organism', 'query': 'mus musculus', 'expected': 'mus musculus', 'category': 'exact'},
            {'type': 'organism', 'query': 'rattus norvegicus', 'expected': 'rattus norvegicus', 'category': 'exact'},
            {'type': 'organism', 'query': 'caenorhabditis elegans', 'expected': 'caenorhabditis elegans', 'category': 'exact'},
            {'type': 'organism', 'query': 'drosophila melanogaster', 'expected': 'drosophila melanogaster', 'category': 'exact'},
            {'type': 'organism', 'query': 'saccharomyces cerevisiae', 'expected': 'saccharomyces cerevisiae', 'category': 'exact'},
            {'type': 'organism', 'query': 'homo sapiens', 'expected': 'homo sapiens', 'category': 'exact'},
            {'type': 'organism', 'query': 'danio rerio', 'expected': 'danio rerio', 'category': 'exact'},
            {'type': 'organism', 'query': 'nothobranchius guentheri', 'expected': 'nothobranchius guentheri', 'category': 'exact'},
            
            # 模糊匹配测试用例
            {'type': 'drug', 'query': 'rapamyc', 'expected': 'rapamycin', 'category': 'fuzzy'},
            {'type': 'drug', 'query': 'metform', 'expected': 'metformin', 'category': 'fuzzy'},
            {'type': 'drug', 'query': 'resveratr', 'expected': 'resveratrol', 'category': 'fuzzy'},
            {'type': 'drug', 'query': 'curcum', 'expected': 'curcumin', 'category': 'fuzzy'},
            {'type': 'drug', 'query': 'caffein', 'expected': 'caffeine', 'category': 'fuzzy'},
            {'type': 'drug', 'query': 'aspir', 'expected': 'aspirin', 'category': 'fuzzy'},
            {'type': 'drug', 'query': 'quercet', 'expected': 'quercetin', 'category': 'fuzzy'},
            {'type': 'drug', 'query': 'spermid', 'expected': 'spermidine', 'category': 'fuzzy'},
            {'type': 'drug', 'query': 'nicotinam', 'expected': 'nicotinamide', 'category': 'fuzzy'},
            {'type': 'drug', 'query': 'hydroxytyr', 'expected': 'hydroxytyrosol', 'category': 'fuzzy'},
            {'type': 'drug', 'query': 'lith', 'expected': 'lithium', 'category': 'fuzzy'},
            {'type': 'drug', 'query': 'coenzyme', 'expected': 'coenzyme q10', 'category': 'fuzzy'},
            {'type': 'drug', 'query': 'omega', 'expected': 'omega-3', 'category': 'fuzzy'},
            {'type': 'drug', 'query': 'melaton', 'expected': 'melatonin', 'category': 'fuzzy'},
            
            # 生物模型模糊匹配
            {'type': 'organism', 'query': 'mus', 'expected': 'mus musculus', 'category': 'fuzzy'},
            {'type': 'organism', 'query': 'rattus', 'expected': 'rattus norvegicus', 'category': 'fuzzy'},
            {'type': 'organism', 'query': 'caenorhabditis', 'expected': 'caenorhabditis elegans', 'category': 'fuzzy'},
            {'type': 'organism', 'query': 'drosophila', 'expected': 'drosophila melanogaster', 'category': 'fuzzy'},
            {'type': 'organism', 'query': 'saccharomyces', 'expected': 'saccharomyces cerevisiae', 'category': 'fuzzy'},
            {'type': 'organism', 'query': 'homo', 'expected': 'homo sapiens', 'category': 'fuzzy'},
            {'type': 'organism', 'query': 'danio', 'expected': 'danio rerio', 'category': 'fuzzy'},
            {'type': 'organism', 'query': 'nothobranchius', 'expected': 'nothobranchius guentheri', 'category': 'fuzzy'},
            
            # 错误查询测试用例（应该找不到）
            {'type': 'drug', 'query': 'nonexistent_drug', 'expected': None, 'category': 'error'},
            {'type': 'drug', 'query': 'xyz123', 'expected': None, 'category': 'error'},
            {'type': 'drug', 'query': 'random_compound', 'expected': None, 'category': 'error'},
            {'type': 'organism', 'query': 'nonexistent_species', 'expected': None, 'category': 'error'},
            {'type': 'organism', 'query': 'xyz456', 'expected': None, 'category': 'error'},
            {'type': 'organism', 'query': 'random_organism', 'expected': None, 'category': 'error'}
        ]
    
    def test_without_synonyms(self, query: str, query_type: str) -> bool:
        """不使用同义词匹配的测试"""
        query_lower = query.lower()
        
        if query_type == 'drug':
            return query_lower in self.drug_names
        elif query_type == 'organism':
            return query_lower in self.organism_names
        return False
    
    def test_with_synonyms(self, query: str, query_type: str) -> Tuple[bool, str]:
        """使用同义词匹配的测试"""
        query_lower = query.lower()
        
        if query_type == 'drug':
            # 直接匹配
            if query_lower in self.drug_names:
                return True, query_lower
            
            # 同义词匹配
            if query_lower in self.drug_synonyms:
                synonym = self.drug_synonyms[query_lower]
                if synonym.lower() in self.drug_names:
                    return True, synonym.lower()
            
            # 模糊匹配
            best_match = self._fuzzy_match(query_lower, self.drug_names)
            if best_match and self._calculate_similarity(query_lower, best_match) >= 0.8:
                return True, best_match
                
        elif query_type == 'organism':
            # 直接匹配
            if query_lower in self.organism_names:
                return True, query_lower
            
            # 同义词匹配
            if query_lower in self.organism_synonyms:
                synonym = self.organism_synonyms[query_lower]
                if synonym.lower() in self.organism_names:
                    return True, synonym.lower()
            
            # 模糊匹配
            best_match = self._fuzzy_match(query_lower, self.organism_names)
            if best_match and self._calculate_similarity(query_lower, best_match) >= 0.8:
                return True, best_match
        
        return False, None
    
    def _fuzzy_match(self, query: str, names: set) -> str:
        """模糊匹配"""
        best_match = None
        best_similarity = 0
        
        for name in names:
            similarity = self._calculate_similarity(query, name)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        return best_match if best_similarity >= 0.8 else None
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算相似度"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def run_accuracy_test(self) -> Dict:
        """运行准确率测试"""
        results = {
            'without_synonyms': {'correct': 0, 'total': 0, 'accuracy': 0.0},
            'with_synonyms': {'correct': 0, 'total': 0, 'accuracy': 0.0},
            'improvement': 0.0,
            'detailed_results': []
        }
        
        for test_case in self.test_cases:
            query = test_case['query']
            query_type = test_case['type']
            expected = test_case['expected']
            category = test_case['category']
            
            # 测试不使用同义词
            without_synonyms_result = self.test_without_synonyms(query, query_type)
            if expected is None:
                without_synonyms_correct = not without_synonyms_result
            else:
                without_synonyms_correct = without_synonyms_result
            
            # 测试使用同义词
            with_synonyms_result, matched_name = self.test_with_synonyms(query, query_type)
            if expected is None:
                with_synonyms_correct = not with_synonyms_result
            else:
                with_synonyms_correct = with_synonyms_result
            
            # 记录结果
            results['without_synonyms']['total'] += 1
            results['with_synonyms']['total'] += 1
            
            if without_synonyms_correct:
                results['without_synonyms']['correct'] += 1
            if with_synonyms_correct:
                results['with_synonyms']['correct'] += 1
            
            # 详细结果
            results['detailed_results'].append({
                'query': query,
                'type': query_type,
                'expected': expected,
                'category': category,
                'without_synonyms': without_synonyms_correct,
                'with_synonyms': with_synonyms_correct,
                'matched_name': matched_name
            })
        
        # 计算准确率
        results['without_synonyms']['accuracy'] = results['without_synonyms']['correct'] / results['without_synonyms']['total']
        results['with_synonyms']['accuracy'] = results['with_synonyms']['correct'] / results['with_synonyms']['total']
        results['improvement'] = results['with_synonyms']['accuracy'] - results['without_synonyms']['accuracy']
        
        return results
    
    def print_results(self, results: Dict):
        """打印测试结果"""
        print("=" * 60)
        print("同义词匹配系统准确率测试结果")
        print("=" * 60)
        
        print(f"\n📊 总体准确率对比:")
        print(f"不使用同义词匹配: {results['without_synonyms']['accuracy']:.1%} ({results['without_synonyms']['correct']}/{results['without_synonyms']['total']})")
        print(f"使用同义词匹配:   {results['with_synonyms']['accuracy']:.1%} ({results['with_synonyms']['correct']}/{results['with_synonyms']['total']})")
        print(f"准确率提升:       {results['improvement']:.1%} ({results['improvement']*100:.1f}个百分点)")
        
        # 按类别分析
        categories = {}
        for result in results['detailed_results']:
            category = result['category']
            if category not in categories:
                categories[category] = {'without': 0, 'with': 0, 'total': 0}
            categories[category]['total'] += 1
            if result['without_synonyms']:
                categories[category]['without'] += 1
            if result['with_synonyms']:
                categories[category]['with'] += 1
        
        print(f"\n📈 按类别分析:")
        for category, stats in categories.items():
            without_acc = stats['without'] / stats['total']
            with_acc = stats['with'] / stats['total']
            improvement = with_acc - without_acc
            print(f"{category:12}: {without_acc:.1%} → {with_acc:.1%} (提升 {improvement:.1%})")
        
        # 显示失败的案例
        print(f"\n❌ 失败的案例:")
        failed_cases = [r for r in results['detailed_results'] if not r['with_synonyms'] and r['expected'] is not None]
        for case in failed_cases[:10]:  # 只显示前10个
            print(f"  {case['query']} ({case['type']}) - 期望: {case['expected']}")
        
        # 显示成功的同义词匹配案例
        print(f"\n✅ 成功的同义词匹配案例:")
        success_cases = [r for r in results['detailed_results'] if r['with_synonyms'] and not r['without_synonyms']]
        for case in success_cases[:10]:  # 只显示前10个
            print(f"  {case['query']} → {case['matched_name']} ({case['type']})")

if __name__ == "__main__":
    # 运行测试
    tester = AccuracyTester('data/drugage.csv')
    results = tester.run_accuracy_test()
    tester.print_results(results)
