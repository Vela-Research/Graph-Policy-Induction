"""
Heterogeneous graph construction for founder networks.

This module builds a heterogeneous graph with node types:
- founder: Individual founders
- university: University tiers based on QS ranking
- company_size: Company size categories
- industry: Industry categories
- role_type: Job role types

Edge types connect founders to their educational background,
work experience, and other attributes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional

from ..data.preprocessing import parse_json_column, parse_qs_rank


# Fixed university tiers (always 5, always same order)
UNIVERSITY_TIERS = [
    'uni_top10',      # QS 1-10
    'uni_top50',      # QS 11-50
    'uni_top100',     # QS 51-100
    'uni_other',      # QS 100+
    'uni_unknown'     # No QS ranking
]

# Fixed role types (always 8, always same order)
ROLE_TYPES = [
    'role_cxo',       # C-Level
    'role_founder',   # Founder
    'role_vp',        # VP
    'role_director',  # Director
    'role_engineer',  # Engineering
    'role_product',   # Product
    'role_business',  # Business
    'role_other'      # Other
]


def build_graph(
    df: pd.DataFrame, 
    fixed_company_sizes: Optional[List[str]] = None,
    fixed_industries: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[Dict, Dict]:
    """
    Build heterogeneous graph structure from founder data.
    
    The graph has:
    - Fixed university tiers (5 tiers based on QS ranking)
    - Fixed role types (8 categories)
    - Dynamic or fixed company sizes and industries
    
    Args:
        df: DataFrame with founder data
        fixed_company_sizes: Optional fixed list of company sizes
        fixed_industries: Optional fixed list of industries
        verbose: Whether to print graph statistics
        
    Returns:
        Tuple of (entities dict, edges dict)
    """
    if verbose:
        print("\n" + "="*50)
        print("BUILDING GRAPH STRUCTURE")
        print("="*50)
    
    # Get founder information
    founders = df['founder_uuid'].tolist()
    founder_success = df['success'].tolist()
    
    # Use fixed lists if provided, otherwise collect from data
    if fixed_company_sizes is not None and fixed_industries is not None:
        company_sizes = fixed_company_sizes
        industries = fixed_industries
        if verbose:
            print("  Using FIXED company_sizes and industries")
    else:
        company_sizes, industries = _collect_categories_from_data(df)
        if verbose:
            print("  Collected company_sizes and industries from data")
    
    # Build entities dictionary
    entities = {
        'founder': {
            'ids': founders, 
            'labels': founder_success, 
            'count': len(founders)
        },
        'university': {
            'ids': UNIVERSITY_TIERS, 
            'count': len(UNIVERSITY_TIERS)
        },
        'company_size': {
            'ids': company_sizes, 
            'count': len(company_sizes)
        },
        'industry': {
            'ids': industries, 
            'count': len(industries)
        },
        'role_type': {
            'ids': ROLE_TYPES, 
            'count': len(ROLE_TYPES)
        }
    }
    
    # Build edges
    edges = _build_edges(df, founders, UNIVERSITY_TIERS, company_sizes, 
                         industries, ROLE_TYPES)
    
    if verbose:
        _print_graph_summary(entities, edges)
    
    return entities, edges


def _collect_categories_from_data(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Collect company sizes and industries from data."""
    company_sizes = set()
    industries = set()
    
    for idx, row in df.iterrows():
        jobs_data = parse_json_column(row.get('jobs_json', '[]'))
        for j in jobs_data:
            cs = j.get('company_size', '')
            if cs:
                company_sizes.add(f'company_{cs.replace(" ", "_").replace("-", "_")[:20]}')
            ind = j.get('industry', '')
            if ind:
                industries.add(f'industry_{ind.replace(" ", "_")[:20]}')
    
    # Also add founder industries
    for ind in df['industry'].dropna().unique():
        industries.add(f'industry_{str(ind).replace(" ", "_")[:20]}')
    
    return list(company_sizes), list(industries)


def _build_edges(
    df: pd.DataFrame,
    founders: List[str],
    universities: List[str],
    company_sizes: List[str],
    industries: List[str],
    role_types: List[str]
) -> Dict[str, Tuple[List[int], List[int]]]:
    """Build edge lists for all edge types."""
    
    # Create index mappings
    founder_to_idx = {f: i for i, f in enumerate(founders)}
    uni_to_idx = {u: i for i, u in enumerate(universities)}
    cs_to_idx = {c: i for i, c in enumerate(company_sizes)}
    ind_to_idx = {ind: i for i, ind in enumerate(industries)}
    role_to_idx = {r: i for i, r in enumerate(role_types)}
    
    edges = {
        'founder_studied_at_university': ([], []),
        'founder_worked_at_company': ([], []),
        'founder_in_industry': ([], []),
        'founder_had_role': ([], [])
    }
    
    for idx, row in df.iterrows():
        founder_idx = founder_to_idx[row['founder_uuid']]
        
        # Education edges
        _add_education_edges(row, founder_idx, uni_to_idx, edges)
        
        # Job edges
        _add_job_edges(row, founder_idx, cs_to_idx, role_to_idx, edges)
        
        # Industry edges
        _add_industry_edges(row, founder_idx, ind_to_idx, edges)
    
    return edges


def _add_education_edges(
    row: pd.Series, 
    founder_idx: int, 
    uni_to_idx: Dict[str, int],
    edges: Dict[str, Tuple[List[int], List[int]]]
):
    """Add education-related edges for a founder."""
    edu_data = parse_json_column(row.get('educations_json', '[]'))
    
    for e in edu_data:
        qs = parse_qs_rank(e.get('qs_ranking'))
        
        if qs <= 10:
            uni_key = 'uni_top10'
        elif qs <= 50:
            uni_key = 'uni_top50'
        elif qs <= 100:
            uni_key = 'uni_top100'
        elif qs < 999:
            uni_key = 'uni_other'
        else:
            uni_key = 'uni_unknown'
        
        edges['founder_studied_at_university'][0].append(founder_idx)
        edges['founder_studied_at_university'][1].append(uni_to_idx[uni_key])


def _add_job_edges(
    row: pd.Series,
    founder_idx: int,
    cs_to_idx: Dict[str, int],
    role_to_idx: Dict[str, int],
    edges: Dict[str, Tuple[List[int], List[int]]]
):
    """Add job-related edges for a founder."""
    jobs_data = parse_json_column(row.get('jobs_json', '[]'))
    
    for j in jobs_data:
        # Company size edge
        cs = j.get('company_size', '')
        if cs:
            cs_key = f'company_{cs.replace(" ", "_").replace("-", "_")[:20]}'
            if cs_key in cs_to_idx:
                edges['founder_worked_at_company'][0].append(founder_idx)
                edges['founder_worked_at_company'][1].append(cs_to_idx[cs_key])
        
        # Role edge
        role = j.get('role', '').lower()
        role_key = _classify_role(role)
        edges['founder_had_role'][0].append(founder_idx)
        edges['founder_had_role'][1].append(role_to_idx[role_key])


def _classify_role(role: str) -> str:
    """Classify a job role into one of the fixed role types."""
    role_lower = role.lower()
    
    if any(kw in role_lower for kw in ['ceo', 'cto', 'cfo', 'chief']):
        return 'role_cxo'
    elif any(kw in role_lower for kw in ['founder', 'co-founder']):
        return 'role_founder'
    elif any(kw in role_lower for kw in ['vp', 'vice president']):
        return 'role_vp'
    elif any(kw in role_lower for kw in ['director', 'head of']):
        return 'role_director'
    elif any(kw in role_lower for kw in ['engineer', 'developer', 'scientist']):
        return 'role_engineer'
    elif any(kw in role_lower for kw in ['product', 'pm']):
        return 'role_product'
    elif any(kw in role_lower for kw in ['sales', 'marketing', 'business']):
        return 'role_business'
    else:
        return 'role_other'


def _add_industry_edges(
    row: pd.Series,
    founder_idx: int,
    ind_to_idx: Dict[str, int],
    edges: Dict[str, Tuple[List[int], List[int]]]
):
    """Add industry edges for a founder."""
    industry = row.get('industry')
    if pd.notna(industry):
        ind_key = f'industry_{str(industry).replace(" ", "_")[:20]}'
        if ind_key in ind_to_idx:
            edges['founder_in_industry'][0].append(founder_idx)
            edges['founder_in_industry'][1].append(ind_to_idx[ind_key])


def _print_graph_summary(entities: Dict, edges: Dict):
    """Print graph statistics."""
    print("\nEntities:")
    for name, data in entities.items():
        print(f"  {name}: {data['count']}")
    
    print("\nEdges:")
    total_edges = 0
    for name, (src, tgt) in edges.items():
        print(f"  {name}: {len(src)}")
        total_edges += len(src)
    print(f"  Total: {total_edges}")


def get_graph_structure(entities: Dict, edges: Dict) -> Dict:
    """
    Get summary of graph structure for saving/loading.
    
    Args:
        entities: Entities dictionary
        edges: Edges dictionary
        
    Returns:
        Dictionary with graph structure metadata
    """
    return {
        'node_counts': {k: v['count'] for k, v in entities.items()},
        'edge_counts': {k: len(v[0]) for k, v in edges.items()},
        'company_sizes': entities['company_size']['ids'],
        'industries': entities['industry']['ids'],
    }
