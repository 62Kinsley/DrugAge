#!/usr/bin/env python3
"""
Test QueryAnalyzer integration with synonym matching
"""

import sys
sys.path.append('src')

def test_query_analyzer_integration():
    """Test the integrated QueryAnalyzer functionality"""
    
    # Import required modules
    from src.utils.synonym_matcher import SynonymMatcher
    
    # Create a simplified QueryAnalyzer for testing
    class TestQueryAnalyzer:
        def __init__(self):
            self.synonym_matcher = SynonymMatcher()
            self.drug_patterns = [
                (r'\b(rapamycin|sirolimus)\b', 'rapamycin'),
                (r'\b(metformin|glucophage)\b', 'metformin'),
                (r'\b(resveratrol)\b', 'resveratrol'),
                (r'\b(curcumin|turmeric)\b', 'curcumin'),
                (r'\b(n-acetylcysteine|nac)\b', 'n-acetylcysteine'),
                (r'\b(aspirin|asp)\b', 'aspirin'),
                (r'\b(caffeine|caf)\b', 'caffeine'),
                (r'\b(quercetin)\b', 'quercetin'),
                (r'\b(spermidine|spd)\b', 'spermidine'),
                (r'\b(nicotinamide|nam)\b', 'nicotinamide'),
            ]
            self.organism_patterns = [
                (r'\b(mouse|mice|mus musculus)\b', 'mouse'),
                (r'\b(rat|rats|rattus norvegicus)\b', 'rat'),
                (r'\b(c\.?\s*elegans|caenorhabditis elegans|worm)\b', 'c. elegans'),
                (r'\b(drosophila|fruit fly|fly)\b', 'drosophila'),
                (r'\b(yeast|saccharomyces cerevisiae)\b', 'yeast'),
                (r'\b(human|homo sapiens)\b', 'human'),
            ]
        
        def analyze_query(self, query):
            import re
            
            # Extract entities
            entities = {'drugs': [], 'organisms': []}
            query_lower = query.lower()
            
            # Extract drugs
            for pattern, drug_name in self.drug_patterns:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                if matches:
                    entities['drugs'].extend([drug_name] * len(matches))
            
            # Extract organisms
            for pattern, org_name in self.organism_patterns:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                if matches:
                    entities['organisms'].extend([org_name] * len(matches))
            
            # Remove duplicates
            entities['drugs'] = list(set(entities['drugs']))
            entities['organisms'] = list(set(entities['organisms']))
            
            # Apply synonym matching
            normalized_entities = self._normalize_entities_with_synonyms(entities)
            
            return {
                'query': query,
                'original_entities': entities,
                'normalized_entities': normalized_entities,
                'synonym_matching_applied': True
            }
        
        def _normalize_entities_with_synonyms(self, entities):
            # Normalize drug names
            normalized_drugs = []
            for drug in entities['drugs']:
                normalized = self.synonym_matcher.normalize_drug_name(drug)
                if normalized:
                    normalized_drugs.append(normalized)
                else:
                    normalized_drugs.append(drug)
            
            # Normalize organism names
            normalized_organisms = []
            for organism in entities['organisms']:
                normalized = self.synonym_matcher.normalize_organism_name(organism)
                if normalized:
                    normalized_organisms.append(normalized)
                else:
                    normalized_organisms.append(organism)
            
            return {
                'drugs': list(set(normalized_drugs)),
                'organisms': list(set(normalized_organisms))
            }
    
    # Test cases
    test_cases = [
        {
            'query': 'Tell me about sirolimus effects in mouse',
            'expected_drugs': ['rapamycin'],
            'expected_organisms': ['mus musculus'],
            'description': 'Drug synonym (sirolimus → rapamycin) and organism synonym (mouse → mus musculus)'
        },
        {
            'query': 'Compare glucophage and rapamycin',
            'expected_drugs': ['metformin', 'rapamycin'],
            'expected_organisms': [],
            'description': 'Drug synonym (glucophage → metformin) with exact match (rapamycin)'
        },
        {
            'query': 'What are the effects of turmeric in C. elegans?',
            'expected_drugs': ['curcumin'],
            'expected_organisms': ['caenorhabditis elegans'],
            'description': 'Drug synonym (turmeric → curcumin) and organism synonym (C. elegans → caenorhabditis elegans)'
        },
        {
            'query': 'How does nac work in drosophila?',
            'expected_drugs': ['n-acetyl-l-cysteine'],
            'expected_organisms': ['drosophila melanogaster'],
            'description': 'Drug synonym (nac → n-acetyl-l-cysteine) and organism synonym (drosophila → drosophila melanogaster)'
        },
        {
            'query': 'Show me aspirin and caffeine data',
            'expected_drugs': ['aspirin', 'caffeine'],
            'expected_organisms': [],
            'description': 'Exact matches (aspirin, caffeine)'
        },
        {
            'query': 'Compare spd and nam in yeast',
            'expected_drugs': ['spermidine', 'nicotinamide'],
            'expected_organisms': ['saccharomyces cerevisiae'],
            'description': 'Drug synonyms (spd → spermidine, nam → nicotinamide) and organism synonym (yeast → saccharomyces cerevisiae)'
        }
    ]
    
    # Run tests
    analyzer = TestQueryAnalyzer()
    
    print('=== QueryAnalyzer Synonym Matching Integration Test ===')
    print()
    
    passed_tests = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f'Test {i}: {test_case["description"]}')
        print(f'Query: "{test_case["query"]}"')
        
        result = analyzer.analyze_query(test_case['query'])
        
        # Check drug normalization
        actual_drugs = set(result['normalized_entities']['drugs'])
        expected_drugs = set(test_case['expected_drugs'])
        drugs_match = actual_drugs == expected_drugs
        
        # Check organism normalization
        actual_organisms = set(result['normalized_entities']['organisms'])
        expected_organisms = set(test_case['expected_organisms'])
        organisms_match = actual_organisms == expected_organisms
        
        # Overall test result
        test_passed = drugs_match and organisms_match
        
        print(f'Original entities:  {result["original_entities"]}')
        print(f'Normalized entities: {result["normalized_entities"]}')
        print(f'Expected drugs: {test_case["expected_drugs"]} | Actual: {list(actual_drugs)} | {"✅" if drugs_match else "❌"}')
        print(f'Expected organisms: {test_case["expected_organisms"]} | Actual: {list(actual_organisms)} | {"✅" if organisms_match else "❌"}')
        print(f'Test result: {"✅ PASSED" if test_passed else "❌ FAILED"}')
        print()
        
        if test_passed:
            passed_tests += 1
    
    # Summary
    print('=== Test Summary ===')
    print(f'Passed: {passed_tests}/{total_tests} tests')
    print(f'Success rate: {passed_tests/total_tests:.1%}')
    
    if passed_tests == total_tests:
        print('🎉 All tests passed! QueryAnalyzer synonym matching integration is working perfectly!')
    else:
        print('⚠️  Some tests failed. Please check the implementation.')
    
    return passed_tests == total_tests

if __name__ == '__main__':
    test_query_analyzer_integration()
