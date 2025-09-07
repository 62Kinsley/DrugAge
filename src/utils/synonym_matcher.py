import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
import json
from difflib import SequenceMatcher
# import pandas as pd

logger = logging.getLogger(__name__)

class SynonymMatcher:
    """Synonym matching system - improves drug and organism name recognition accuracy"""
    
    def __init__(self):
        """Initialize synonym matcher"""
        self.drug_synonyms = self._build_drug_synonyms()
        self.organism_synonyms = self._build_organism_synonyms()
        self.effect_synonyms = self._build_effect_synonyms()
        
        # Fuzzy matching thresholds
        self.fuzzy_threshold = 0.8
        self.exact_threshold = 0.95
        
        # Statistics
        self.match_stats = {
            'total_queries': 0,
            'exact_matches': 0,
            'fuzzy_matches': 0,
            'no_matches': 0
        }
    
    def _build_drug_synonyms(self) -> Dict[str, str]:
        """Build drug synonym dictionary"""
        return {
            # mTOR inhibitors
            'sirolimus': 'rapamycin',
            'rapa': 'rapamycin',
            'rapamune': 'rapamycin',
            'ay-22989': 'rapamycin',
            'wy-090217': 'rapamycin',
            'cci-779': 'rapamycin',
            'temsirolimus': 'rapamycin',
            'everolimus': 'rapamycin',
            'ridaforolimus': 'rapamycin',
            
            # Biguanides
            'glucophage': 'metformin',
            'met': 'metformin',
            'dimethylbiguanide': 'metformin',
            'n,n-dimethylbiguanide': 'metformin',
            '1,1-dimethylbiguanide': 'metformin',
            'metformin hydrochloride': 'metformin',
            
            # Resveratrol
            'res': 'resveratrol',
            'trans-resveratrol': 'resveratrol',
            '3,4': 'resveratrol',
            '5-trihydroxystilbene': 'resveratrol',
            'cis-resveratrol': 'resveratrol',
            'piceatannol': 'resveratrol',
            
            # Aspirin
            'acetylsalicylic acid': 'aspirin',
            'asa': 'aspirin',
            'asp': 'aspirin',
            'acetylsalicylate': 'aspirin',
            '2-acetoxybenzoic acid': 'aspirin',
            'salicylic acid acetate': 'aspirin',
            
            # Curcumin
            'curcuma longa': 'curcumin',
            'turmeric': 'curcumin',
            'diferuloylmethane': 'curcumin',
            '1,7-bis(4-hydroxy-3-methoxyphenyl)-1,6-heptadiene-3,5-dione': 'curcumin',
            
            # Lithium
            'lithium chloride': 'lithium',
            'licl': 'lithium',
            'lithium carbonate': 'lithium',
            'lithium citrate': 'lithium',
            'lithium acetate': 'lithium',
            'lithium sulfate': 'lithium',
            
            # Caffeine
            'caf': 'caffeine',
            '1,3,7-trimethylxanthine': 'caffeine',
            'coffee': 'caffeine',
            'theine': 'caffeine',
            'guaranine': 'caffeine',
            'methyltheobromine': 'caffeine',
            
            # Spermidine
            'spd': 'spermidine',
            'n-(3-aminopropyl)-1,4-butanediamine': 'spermidine',
            '1,5,10-triazadecane': 'spermidine',
            'n-(3-aminopropyl)butane-1,4-diamine': 'spermidine',
            
            # Nicotinamide
            'nam': 'nicotinamide',
            'niacinamide': 'nicotinamide',
            'vitamin b3': 'nicotinamide',
            'pyridine-3-carboxamide': 'nicotinamide',
            'nicotinic acid amide': 'nicotinamide',
            '3-pyridinecarboxamide': 'nicotinamide',
            
            # Quercetin
            'quercetol': 'quercetin',
            'sophoretin': 'quercetin',
            'meletin': 'quercetin',
            'xanthaurine': 'quercetin',
            '3,3': 'quercetin',
            '4': 'quercetin',
            '5,7-pentahydroxyflavone': 'quercetin',
            
            # Green tea
            'egcg': 'epigallocatechin-3-gallate',
            'epigallocatechin gallate': 'epigallocatechin-3-gallate',
            'catechin': 'epigallocatechin-3-gallate',
            'camellia sinensis': 'epigallocatechin-3-gallate',
            'green tea extract': 'epigallocatechin-3-gallate',
            'gtee': 'epigallocatechin-3-gallate',
            'polyphenol': 'epigallocatechin-3-gallate',
            
            # Caloric restriction
            'cr': 'caloric restriction',
            'dietary restriction': 'caloric restriction',
            'calorie restriction': 'caloric restriction',
            'food restriction': 'caloric restriction',
            'undernutrition': 'caloric restriction',
            'caloric restriction mimetic': 'caloric restriction',
            
            # N-acetylcysteine
            'nac': 'n-acetyl-l-cysteine',
            'n-acetyl-l-cysteine': 'n-acetyl-l-cysteine',
            'acetylcysteine': 'n-acetyl-l-cysteine',
            'n-acetyl cysteine': 'n-acetyl-l-cysteine',
            'mucomyst': 'n-acetyl-l-cysteine',
            'acetadote': 'n-acetyl-l-cysteine',
            
            # Hydroxytyrosol
            'ht': 'hydroxytyrosol',
            '3,4-dihydroxyphenylethanol': 'hydroxytyrosol',
            'dopet': 'hydroxytyrosol',
            '3,4-dhpea': 'hydroxytyrosol',
            
            # Vitamin D
            'cholecalciferol': 'vitamin d',
            'vitamin d3': 'vitamin d',
            'calcitriol': 'vitamin d',
            '1,25-dihydroxyvitamin d3': 'vitamin d',
            'ergocalciferol': 'vitamin d',
            'vitamin d2': 'vitamin d',
            
            # Vitamin E
            'tocopherol': 'vitamin e',
            'alpha-tocopherol': 'vitamin e',
            'vitamin e acetate': 'vitamin e',
            'tocotrienol': 'vitamin e',
            
            # Vitamin C
            'ascorbic acid': 'vitamin c',
            'ascorbate': 'vitamin c',
            'l-ascorbic acid': 'vitamin c',
            'vitamin c sodium': 'vitamin c',
            
            # Coenzyme Q10
            'coq10': 'coenzyme q10',
            'ubiquinone': 'coenzyme q10',
            'ubiquinol': 'coenzyme q10',
            'coenzyme q': 'coenzyme q10',
            'q10': 'coenzyme q10',
            
            # Omega-3
            'fish oil': 'omega-3',
            'epa': 'omega-3',
            'dha': 'omega-3',
            'eicosapentaenoic acid': 'omega-3',
            'docosahexaenoic acid': 'omega-3',
            'omega 3': 'omega-3',
            'n-3 fatty acids': 'omega-3',
            
            # Melatonin
            'mt': 'melatonin',
            'n-acetyl-5-methoxytryptamine': 'melatonin',
            'mel': 'melatonin',
            '5-methoxy-n-acetyltryptamine': 'melatonin'
        }
    
    def _build_organism_synonyms(self) -> Dict[str, str]:
        """Build organism synonym dictionary"""
        return {
            # Mouse
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
            
            # Rat
            'rat': 'rattus norvegicus',
            'rats': 'rattus norvegicus',
            'norway rat': 'rattus norvegicus',
            'brown rat': 'rattus norvegicus',
            'wistar': 'rattus norvegicus',
            'sprague-dawley': 'rattus norvegicus',
            'fischer 344': 'rattus norvegicus',
            'lewis rat': 'rattus norvegicus',
            
            # C. elegans
            'c. elegans': 'caenorhabditis elegans',
            'c elegans': 'caenorhabditis elegans',
            'c.elegans': 'caenorhabditis elegans',
            'worm': 'caenorhabditis elegans',
            'worms': 'caenorhabditis elegans',
            'nematode': 'caenorhabditis elegans',
            'roundworm': 'caenorhabditis elegans',
            'elegans': 'caenorhabditis elegans',
            
            # Drosophila
            'drosophila': 'drosophila melanogaster',
            'd. melanogaster': 'drosophila melanogaster',
            'd.melanogaster': 'drosophila melanogaster',
            'fruit fly': 'drosophila melanogaster',
            'fly': 'drosophila melanogaster',
            'flies': 'drosophila melanogaster',
            'vinegar fly': 'drosophila melanogaster',
            
            # Yeast
            'yeast': 'saccharomyces cerevisiae',
            's. cerevisiae': 'saccharomyces cerevisiae',
            's.cerevisiae': 'saccharomyces cerevisiae',
            'baker\'s yeast': 'saccharomyces cerevisiae',
            'budding yeast': 'saccharomyces cerevisiae',
            'saccharomyces': 'saccharomyces cerevisiae',
            
            # Human
            'human': 'homo sapiens',
            'humans': 'homo sapiens',
            'h. sapiens': 'homo sapiens',
            'h.sapiens': 'homo sapiens',
            'human cells': 'homo sapiens',
            'human tissue': 'homo sapiens',
            'clinical': 'homo sapiens',
            
            # Zebrafish
            'zebrafish': 'danio rerio',
            'zebra fish': 'danio rerio',
            'zebra-fish': 'danio rerio',
            'd. rerio': 'danio rerio',
            'd.rerio': 'danio rerio',
            
            # Killifish
            'killifish': 'nothobranchius guentheri',
            'n. guentheri': 'nothobranchius guentheri',
            'n.guentheri': 'nothobranchius guentheri',
            'annual fish': 'nothobranchius guentheri',
            'turquoise killifish': 'nothobranchius guentheri'
        }
    
    def _build_effect_synonyms(self) -> Dict[str, str]:
        """Build effect-related synonym dictionary"""
        return {
            'life span': 'lifespan',
            'longevity': 'lifespan',
            'survival': 'lifespan',
            'lifespan extension': 'lifespan',
            'life extension': 'lifespan',
            'longevity extension': 'lifespan',
            'survival time': 'lifespan',
            
            'ageing': 'aging',
            'senescence': 'aging',
            'cellular aging': 'aging',
            'biological aging': 'aging',
            'chronological aging': 'aging',
            'replicative aging': 'aging',
            
            'impact': 'effect',
            'influence': 'effect',
            'result': 'effect',
            'outcome': 'effect',
            'response': 'effect',
            'change': 'effect',
            'modification': 'effect',
            'alteration': 'effect',
            
            'prolong': 'extend',
            'increase': 'extend',
            'enhance': 'extend',
            'improve': 'extend',
            'boost': 'extend',
            'augment': 'extend',
            'amplify': 'extend',
            'strengthen': 'extend',
            
            'percentage': 'percent',
            '%': 'percent',
            'percent change': 'percent',
            'relative change': 'percent',
            'fold change': 'percent',
            'times': 'percent',
            'ratio': 'percent'
        }
    
    def normalize_drug_name(self, drug_name: str) -> Optional[str]:
        """
        Normalize drug name
        
        Args:
            drug_name: Original drug name
            
        Returns:
            Normalized drug name, None if not found
        """
        if not drug_name:
            return None
            
        drug_lower = drug_name.lower().strip()
        
        # 1. Exact match
        if drug_lower in self.drug_synonyms:
            return self.drug_synonyms[drug_lower]
        
        # 2. Fuzzy match
        best_match = self._fuzzy_match_drug(drug_lower)
        if best_match:
            return best_match
        
        return None
    
    def normalize_organism_name(self, organism_name: str) -> Optional[str]:
        """
        Normalize organism name
        
        Args:
            organism_name: Original organism name
            
        Returns:
            Normalized organism name, None if not found
        """
        if not organism_name:
            return None
            
        organism_lower = organism_name.lower().strip()
        
        # 1. Exact match
        if organism_lower in self.organism_synonyms:
            return self.organism_synonyms[organism_lower]
        
        # 2. Fuzzy match
        best_match = self._fuzzy_match_organism(organism_lower)
        if best_match:
            return best_match
        
        return None
    
    def _fuzzy_match_drug(self, query: str) -> Optional[str]:
        """Fuzzy match for drugs"""
        best_match = None
        best_similarity = 0
        
        for synonym, standard_name in self.drug_synonyms.items():
            similarity = self._calculate_similarity(query, synonym)
            if similarity > best_similarity and similarity >= self.fuzzy_threshold:
                best_similarity = similarity
                best_match = standard_name
        
        return best_match
    
    def _fuzzy_match_organism(self, query: str) -> Optional[str]:
        """Fuzzy match for organisms"""
        best_match = None
        best_similarity = 0
        
        for synonym, standard_name in self.organism_synonyms.items():
            similarity = self._calculate_similarity(query, synonym)
            if similarity > best_similarity and similarity >= self.fuzzy_threshold:
                best_similarity = similarity
                best_match = standard_name
        
        return best_match
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: Text 1
            text2: Text 2
            
        Returns:
            Similarity score (0-1)
        """
        # Exact match
        if text1 == text2:
            return 1.0
        
        # Substring match
        if text1 in text2 or text2 in text1:
            return 0.95
        
        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, text1, text2).ratio()
        
        # Consider length difference
        length_penalty = min(len(text1), len(text2)) / max(len(text1), len(text2))
        adjusted_similarity = similarity * length_penalty
        
        return adjusted_similarity
    
    def extract_entities_with_synonyms(self, query: str) -> Dict[str, List[Dict]]:
        """
        Extract entities from query and match synonyms
        
        Args:
            query: Query text
            
        Returns:
            Extracted entities dictionary
        """
        entities = {
            'drugs': [],
            'organisms': [],
            'effects': []
        }
        
        query_lower = query.lower()
        
        # Extract drugs
        for synonym, standard_name in self.drug_synonyms.items():
            if synonym in query_lower:
                entities['drugs'].append({
                    'original': synonym,
                    'standardized': standard_name,
                    'similarity': 1.0,
                    'confidence': 'high'
                })
        
        # Extract organisms
        for synonym, standard_name in self.organism_synonyms.items():
            if synonym in query_lower:
                entities['organisms'].append({
                    'original': synonym,
                    'standardized': standard_name,
                    'similarity': 1.0,
                    'confidence': 'high'
                })
        
        # Extract effect-related terms
        for synonym, standard_name in self.effect_synonyms.items():
            if synonym in query_lower:
                entities['effects'].append({
                    'original': synonym,
                    'standardized': standard_name,
                    'similarity': 1.0,
                    'confidence': 'high'
                })
        
        # Update statistics
        self.match_stats['total_queries'] += 1
        if entities['drugs'] or entities['organisms']:
            self.match_stats['exact_matches'] += 1
        else:
            self.match_stats['no_matches'] += 1
        
        return entities
    
    def get_match_accuracy(self) -> Dict[str, float]:
        """
        Get matching accuracy statistics
        
        Returns:
            Accuracy statistics dictionary
        """
        total = self.match_stats['total_queries']
        if total == 0:
            return {'exact_accuracy': 0.0, 'fuzzy_accuracy': 0.0, 'overall_accuracy': 0.0}
        
        exact_accuracy = self.match_stats['exact_matches'] / total
        fuzzy_accuracy = (self.match_stats['exact_matches'] + self.match_stats['fuzzy_matches']) / total
        overall_accuracy = fuzzy_accuracy
        
        return {
            'exact_accuracy': exact_accuracy,
            'fuzzy_accuracy': fuzzy_accuracy,
            'overall_accuracy': overall_accuracy
        }
    
    def add_custom_synonyms(self, category: str, standard_name: str, synonyms: List[str]):
        """
        Add custom synonyms
        
        Args:
            category: Category ('drug', 'organism', 'effect')
            standard_name: Standard name
            synonyms: Synonym list
        """
        if category == 'drug':
            for synonym in synonyms:
                self.drug_synonyms[synonym.lower()] = standard_name.lower()
        elif category == 'organism':
            for synonym in synonyms:
                self.organism_synonyms[synonym.lower()] = standard_name.lower()
        elif category == 'effect':
            for synonym in synonyms:
                self.effect_synonyms[synonym.lower()] = standard_name.lower()
        else:
            raise ValueError(f"Unknown category: {category}")
        
        logger.info(f"Added {category} synonyms: {standard_name} -> {synonyms}")
    
    def save_synonyms(self, filepath: str):
        """
        Save synonym dictionary to file
        
        Args:
            filepath: File path
        """
        data = {
            'drug_synonyms': self.drug_synonyms,
            'organism_synonyms': self.organism_synonyms,
            'effect_synonyms': self.effect_synonyms,
            'match_stats': self.match_stats
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Synonym dictionary saved to: {filepath}")
    
    def load_synonyms(self, filepath: str):
        """
        Load synonym dictionary from file
        
        Args:
            filepath: File path
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.drug_synonyms = data.get('drug_synonyms', {})
        self.organism_synonyms = data.get('organism_synonyms', {})
        self.effect_synonyms = data.get('effect_synonyms', {})
        self.match_stats = data.get('match_stats', {
            'total_queries': 0,
            'exact_matches': 0,
            'fuzzy_matches': 0,
            'no_matches': 0
        })
        
        logger.info(f"Synonym dictionary loaded from: {filepath}")

# Test code
if __name__ == "__main__":
    matcher = SynonymMatcher()
    
    # Test drug synonym matching
    test_queries = [
        "sirolimus",
        "glucophage", 
        "mouse",
        "c. elegans",
        "rapamycin",
        "metformin"
    ]
    
    print("=== Synonym Matching Test ===")
    for query in test_queries:
        if query in ["sirolimus", "glucophage", "mouse", "c. elegans"]:
            if query in ["sirolimus", "glucophage"]:
                result = matcher.normalize_drug_name(query)
                print(f"Drug query: {query} -> {result}")
            else:
                result = matcher.normalize_organism_name(query)
                print(f"Organism query: {query} -> {result}")
        else:
            result = matcher.normalize_drug_name(query)
            print(f"Exact match: {query} -> {result}")
    
    print("\n=== Entity Extraction Test ===")
    test_query = "Compare sirolimus and glucophage effects in mouse"
    entities = matcher.extract_entities_with_synonyms(test_query)
    print(f"Query: {test_query}")
    print(f"Extracted entities: {entities}")
