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
from src.utils.synonym_matcher import SynonymMatcher

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Query type enumeration"""
    DRUG_SEARCH = "drug_search"
    EFFECT_ANALYSIS = "effect_analysis"
    COMPARISON = "comparison"
    RANKING = "ranking"
    ORGANISM_SPECIFIC = "organism_specific"
    MECHANISM = "mechanism"
    GENERAL = "general"

@dataclass
class QueryContext:
    """Query context data class"""
    query_type: QueryType
    primary_intent: str
    entities: Dict[str, List[str]]
    parameters: Dict[str, Any]
    confidence: float
    suggestions: List[str]

class QueryAnalyzer:
    """Natural language query analyzer with synonym matching"""
    
    def __init__(self):
        """Initialize query analyzer"""
        self.drug_patterns = self._build_drug_patterns()
        self.organism_patterns = self._build_organism_patterns()
        self.effect_patterns = self._build_effect_patterns()
        self.comparison_patterns = self._build_comparison_patterns()
        self.ranking_patterns = self._build_ranking_patterns()
        self.mechanism_patterns = self._build_mechanism_patterns()
        
        # Initialize synonym matcher
        self.synonym_matcher = SynonymMatcher()
        
    def _build_drug_patterns(self) -> List[Tuple[str, str]]:
        """Build drug recognition patterns"""
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
        """Build organism recognition patterns"""
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
        """Build effect analysis patterns"""
        return [
            r'\b(lifespan|life span|longevity|aging|ageing)\b',
            r'\b(extend|extension|increase|prolong|benefit|improve)\b',
            r'\b(effect|impact|influence|result|outcome)\b',
            r'\b(survival|mortality|death|live longer)\b',
            r'\b(percent|percentage|%|fold|times)\b',
            r'\b(healthspan|health span)\b',
        ]
    
    def _build_comparison_patterns(self) -> List[str]:
        """Build comparison patterns"""
        return [
            r'\b(compare|comparison|versus|vs\.?|against)\b',
            r'\b(better|worse|more effective|less effective)\b',
            r'\b(difference|differ|similar|same|alike)\b',
            r'\b(which is|what is the difference|which one)\b',
            r'\b(superior|inferior|outperform)\b',
        ]
    
    def _build_ranking_patterns(self) -> List[str]:
        """Build ranking patterns"""
        return [
            r'\b(best|top|most|highest|greatest|maximum)\b',
            r'\b(worst|bottom|least|lowest|smallest|minimum)\b',
            r'\b(rank|ranking|order|list|sort)\b',
            r'\b(first|second|third|\d+st|\d+nd|\d+rd|\d+th)\b',
            r'\b(leading|premier|superior)\b',
        ]
    
    def _build_mechanism_patterns(self) -> List[str]:
        """Build mechanism query patterns"""
        return [
            r'\b(how|why|mechanism|pathway|target)\b',
            r'\b(work|function|operate|act)\b',
            r'\b(molecular|cellular|biochemical)\b',
            r'\b(gene|protein|enzyme|receptor)\b',
            r'\b(signaling|signal|cascade)\b',
        ]
    
    def analyze_query(self, query: str) -> QueryContext:
        """
        Analyze natural language query with synonym matching
        
        Args:
            query: User query text
            
        Returns:
            QueryContext: Query context object
        """
        query_lower = query.lower().strip()
        
        # Apply synonym matching to normalize query BEFORE entity extraction
        normalized_query = self._normalize_query_with_synonyms(query_lower)
        
        # Extract entities from normalized query
        entities = self._extract_entities(normalized_query)
        
        # Determine query type
        query_type, confidence = self._determine_query_type(query_lower, entities)
        
        # Extract parameters
        parameters = self._extract_parameters(query_lower, query_type, entities)
        
        # Determine primary intent
        primary_intent = self._determine_primary_intent(query_lower, query_type)
        
        # Generate suggestions
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
        """Extract entities from query"""
        entities = {
            'drugs': [],
            'organisms': [],
            'numbers': [],
            'effects': [],
            'time_periods': []
        }
        
        # Extract drugs
        for pattern, drug_name in self.drug_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    # Handle patterns with capture groups
                    entities['drugs'].extend([match for match in matches[0] if match])
                else:
                    entities['drugs'].extend([drug_name] * len(matches))
        
        # Extract organisms
        for pattern, organism_name in self.organism_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                entities['organisms'].extend([organism_name] * len(matches))
        
        # Extract numbers and values
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
        
        # Extract effect-related terms
        for pattern in self.effect_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                matches = re.findall(pattern, query, re.IGNORECASE)
                entities['effects'].extend(matches)
        
        # Remove duplicates
        entities['drugs'] = list(set(entities['drugs']))
        entities['organisms'] = list(set(entities['organisms']))
        entities['effects'] = list(set(entities['effects']))
        
        return entities
    

    def _normalize_query_with_synonyms(self, query: str) -> str:
        """
        Normalize query text by replacing synonyms with standard names
        
        Args:
            query: Original query text
            
        Returns:
            str: Normalized query text
        """
        normalized_query = query
        
        # Replace drug synonyms
        for synonym, standard in self.synonym_matcher.drug_synonyms.items():
            pattern = r'\b' + re.escape(synonym) + r'\b'
            normalized_query = re.sub(pattern, standard, normalized_query, flags=re.IGNORECASE)
        
        # Replace organism synonyms
        for synonym, standard in self.synonym_matcher.organism_synonyms.items():
            pattern = r'\b' + re.escape(synonym) + r'\b'
            normalized_query = re.sub(pattern, standard, normalized_query, flags=re.IGNORECASE)
        
        return normalized_query

    def _normalize_query_with_synonyms(self, query: str) -> str:
        """
        Normalize query text by replacing synonyms with standard names
        
        Args:
            query: Original query text
            
        Returns:
            str: Normalized query text
        """
        normalized_query = query
        
        # Replace drug synonyms
        for synonym, standard in self.synonym_matcher.drug_synonyms.items():
            pattern = r'\b' + re.escape(synonym) + r'\b'
            normalized_query = re.sub(pattern, standard, normalized_query, flags=re.IGNORECASE)
        
        # Replace organism synonyms
        for synonym, standard in self.synonym_matcher.organism_synonyms.items():
            pattern = r'\b' + re.escape(synonym) + r'\b'
            normalized_query = re.sub(pattern, standard, normalized_query, flags=re.IGNORECASE)
        
        return normalized_query
    def _normalize_entities_with_synonyms(self, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Normalize entities using synonym matching
        
        Args:
            entities: Raw extracted entities
            
        Returns:
            Normalized entities
        """
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
        
        # Update entities with normalized names
        entities['drugs'] = list(set(normalized_drugs))
        entities['organisms'] = list(set(normalized_organisms))
        
        return entities
    
    def _determine_query_type(self, query: str, entities: Dict) -> Tuple[QueryType, float]:
        """Determine query type and confidence"""
        scores = {QueryType.GENERAL: 0.1}  # Default score
        
        # Drug search
        if entities['drugs']:
            scores[QueryType.DRUG_SEARCH] = 0.6
            if any(word in query for word in ['information', 'about', 'tell me', 'what is', 'describe']):
                scores[QueryType.DRUG_SEARCH] += 0.3
        
        # Effect analysis
        effect_score = 0
        for pattern in self.effect_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                effect_score += 0.2
        if effect_score > 0:
            scores[QueryType.EFFECT_ANALYSIS] = min(effect_score, 0.8)
        
        # Comparison query
        comparison_score = 0
        for pattern in self.comparison_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                comparison_score += 0.3
        
        if len(entities['drugs']) >= 2:
            comparison_score += 0.4
        
        if comparison_score > 0:
            scores[QueryType.COMPARISON] = min(comparison_score, 0.9)
        
        # Ranking query
        ranking_score = 0
        for pattern in self.ranking_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                ranking_score += 0.3
        
        # Check for quantity limiting words
        if any(num_info['type'] in ['top_n', 'first_n'] for num_info in entities['numbers']):
            ranking_score += 0.2
        
        if ranking_score > 0:
            scores[QueryType.RANKING] = min(ranking_score, 0.8)
        
        # Organism-specific query
        if entities['organisms']:
            scores[QueryType.ORGANISM_SPECIFIC] = 0.5
            if any(word in query for word in ['in', 'on', 'for', 'using', 'with', 'tested']):
                scores[QueryType.ORGANISM_SPECIFIC] += 0.3
        
        # Mechanism query
        mechanism_score = 0
        for pattern in self.mechanism_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                mechanism_score += 0.2
        
        if mechanism_score > 0:
            scores[QueryType.MECHANISM] = min(mechanism_score, 0.7)
        
        # Select highest scoring type
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return best_type, min(confidence, 1.0)
    
    def _extract_parameters(self, query: str, query_type: QueryType, entities: Dict) -> Dict[str, Any]:
        """Extract parameters based on query type"""
        parameters = {}
        
        # Extract quantity limits
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
        
        # Set default quantity limit if not specified
        if 'limit' not in parameters:
            if query_type == QueryType.RANKING:
                if re.search(r'\b(best|top|most)\b', query):
                    parameters['limit'] = 10  # Default top 10
        
        # Extract minimum effect threshold
        if 'min_effect' not in parameters:
            percent_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:percent|%)', query)
            if percent_match:
                parameters['min_effect'] = float(percent_match.group(1))
        
        # Extract comparison parameters
        if query_type == QueryType.COMPARISON:
            parameters['comparison_type'] = 'comprehensive'
            if 'statistical' in query or 'significant' in query or 'stats' in query:
                parameters['include_statistics'] = True
            if 'summary' in query or 'brief' in query:
                parameters['comparison_type'] = 'summary'
        
        # Extract sorting parameters
        if any(word in query for word in ['ascending', 'lowest', 'worst', 'smallest']):
            parameters['sort_order'] = 'ascending'
        else:
            parameters['sort_order'] = 'descending'
        
        # Extract search parameters
        if query_type == QueryType.DRUG_SEARCH:
            if 'exact' in query or 'exactly' in query:
                parameters['exact_match'] = True
            if 'similar' in query or 'related' in query:
                parameters['include_similar'] = True
        
        return parameters
    
    def _determine_primary_intent(self, query: str, query_type: QueryType) -> str:
        """Determine primary intent"""
        intent_map = {
            QueryType.DRUG_SEARCH: "Find detailed information about specific drugs",
            QueryType.EFFECT_ANALYSIS: "Analyze lifespan extension effects of compounds",
            QueryType.COMPARISON: "Compare multiple drugs or treatments",
            QueryType.RANKING: "Rank drugs by effectiveness",
            QueryType.ORGANISM_SPECIFIC: "Find drugs tested in specific model organisms",
            QueryType.MECHANISM: "Understand drug mechanisms of action",
            QueryType.GENERAL: "Get general information about longevity research"
        }
        
        return intent_map.get(query_type, "General query")
    
    def _generate_suggestions(self, 
                            query_type: QueryType, 
                            entities: Dict, 
                            parameters: Dict) -> List[str]:
        """Generate query improvement suggestions"""
        suggestions = []
        
        # Suggestions based on query type
        if query_type == QueryType.DRUG_SEARCH and not entities['drugs']:
            suggestions.append("Try specifying specific drug names like 'rapamycin' or 'metformin'")
        
        if query_type == QueryType.COMPARISON and len(entities['drugs']) < 2:
            suggestions.append("Comparison queries need at least two drug names")
        
        if query_type == QueryType.ORGANISM_SPECIFIC and not entities['organisms']:
            suggestions.append("Please specify model organisms like 'mouse', 'rat', or 'C. elegans'")
        
        if query_type == QueryType.RANKING and 'limit' not in parameters:
            suggestions.append("Consider specifying the number of results needed (e.g., 'top 10')")
        
        # Suggestions based on entities
        if not entities['drugs'] and not entities['organisms']:
            suggestions.extend([
                "Try being more specific about drugs or model organisms of interest",
                "Examples: 'rapamycin effects in mouse' or 'compare metformin and resveratrol'"
            ])
        
        # Query optimization suggestions
        if query_type == QueryType.GENERAL:
            suggestions.extend([
                "Try using more specific queries",
                "Specify drug names, model organisms, or research types"
            ])
        
        return suggestions[:3]  # Limit number of suggestions
    
    def get_query_examples(self) -> Dict[str, List[str]]:
        """Get query examples for different categories"""
        return {
            "Drug Search": [
                "Tell me about rapamycin",
                "What are the effects of metformin on lifespan?",
                "How does resveratrol work?"
            ],
            "Effect Analysis": [
                "Which drugs extend lifespan by more than 20%?",
                "Show me the most effective longevity compounds",
                "Analyze drugs with significant lifespan extension effects"
            ],
            "Drug Comparison": [
                "Compare rapamycin and metformin",
                "Which is better: resveratrol or curcumin?",
                "Rapamycin vs metformin vs resveratrol effects"
            ],
            "Effect Ranking": [
                "Top 10 most effective longevity drugs",
                "Rank compounds by lifespan extension effectiveness",
                "Best performing drugs in mouse studies"
            ],
            "Organism-Specific Queries": [
                "Drugs tested in mouse",
                "Which compounds are effective in C. elegans?",
                "Significant effects in rat studies"
            ],
            "Mechanism Queries": [
                "How does rapamycin extend lifespan?",
                "What is the mechanism of metformin?",
                "Why does resveratrol affect aging?"
            ]
        }
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """Validate query validity and completeness"""
        validation = {
            'is_valid': True,
            'issues': [],
            'suggestions': [],
            'confidence': 1.0
        }
        
        if len(query.strip()) < 3:
            validation['is_valid'] = False
            validation['issues'].append("Query too short")
            validation['suggestions'].append("Please provide more detailed query")
        
        # Check for meaningful content
        meaningful_words = ['drug', 'compound', 'effect', 'lifespan', 'longevity', 
                          'mouse', 'rat', 'compare', 'best', 'top']
        
        if not any(word in query.lower() for word in meaningful_words):
            validation['confidence'] -= 0.3
            validation['suggestions'].append("Try including drug names or research-related keywords")
        
        # Check query clarity
        if '?' not in query and not any(word in query.lower() for word in 
                                       ['show', 'tell', 'find', 'compare', 'what', 'which', 'how']):
            validation['suggestions'].append("Consider phrasing the query as a question")
        
        return validation

# Test code
if __name__ == "__main__":
    analyzer = QueryAnalyzer()
    
    # Test queries
    test_queries = [
        "Tell me about sirolimus effects in mouse",
        "Compare glucophage and rapamycin",
        "What are the top 10 longevity drugs?",
        "Which drugs are effective in C. elegans?",
        "How does resveratrol extend lifespan?",
        "Show me drugs with more than 20% lifespan extension"
    ]
    
    print("=== Query Analysis Test with Synonym Matching ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        context = analyzer.analyze_query(query)
        print(f"Type: {context.query_type.value}")
        print(f"Intent: {context.primary_intent}")
        print(f"Confidence: {context.confidence:.2f}")
        print(f"Entities: {context.entities}")
        print(f"Parameters: {context.parameters}")
        if context.suggestions:
            print(f"Suggestions: {context.suggestions}")
    
    print("\n=== Query Examples ===")
    examples = analyzer.get_query_examples()
    for category, example_list in examples.items():
        print(f"\n{category}:")
        for example in example_list:
            print(f"  - {example}")
