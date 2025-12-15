"""
Data preprocessing and feature extraction utilities.

This module handles parsing of JSON columns and extraction of
education and job-related features from founder profiles.
"""

import json
import numpy as np
import pandas as pd
from typing import Any, List, Dict, Tuple, Optional


def parse_json_column(json_str: Any) -> List[Dict]:
    """
    Safely parse JSON columns from DataFrame.
    
    Args:
        json_str: JSON string or NaN value
        
    Returns:
        Parsed list of dictionaries or empty list if parsing fails
    """
    if pd.isna(json_str):
        return []
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return []


def parse_qs_rank(qs_value: Any) -> int:
    """
    Parse QS university ranking handling various formats.
    
    Handles formats like '200+', '101-150', plain integers, etc.
    
    Args:
        qs_value: QS ranking value in various formats
        
    Returns:
        Integer ranking (999 if unknown/invalid)
    """
    if pd.isna(qs_value) or qs_value == '':
        return 999
    
    qs_str = str(qs_value).strip()
    
    # Handle '200+' format
    if '+' in qs_str:
        try:
            return int(qs_str.replace('+', ''))
        except ValueError:
            return 999
    
    # Handle '101-150' range format
    if '-' in qs_str:
        try:
            return int(qs_str.split('-')[0])
        except ValueError:
            return 999
    
    # Handle plain integer
    try:
        return int(float(qs_str))
    except ValueError:
        return 999


def parse_duration(duration_str: Any) -> float:
    """
    Parse job duration strings to years.
    
    Args:
        duration_str: Duration string like '2-3 years', '<2 years', etc.
        
    Returns:
        Estimated duration in years as float
    """
    if pd.isna(duration_str) or duration_str == '':
        return 0.0
    
    d = str(duration_str).lower()
    
    # Handle various duration formats
    if '10+' in d or '>10' in d:
        return 12.0
    elif '6-9' in d or '6-10' in d:
        return 7.5
    elif '4-5' in d or '4-6' in d:
        return 4.5
    elif '2-3' in d or '2-4' in d:
        return 2.5
    elif '<2' in d or '0-2' in d or '1-2' in d:
        return 1.0
    elif '<1' in d or '0-1' in d:
        return 0.5
    else:
        try:
            return float(''.join(c for c in d if c.isdigit() or c == '.'))
        except ValueError:
            return 0.0


def compute_jaccard_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    """
    Compute Jaccard similarity between two binary feature vectors.
    
    Args:
        f1: First binary feature vector
        f2: Second binary feature vector
        
    Returns:
        Jaccard similarity coefficient [0, 1]
    """
    f1_bool = f1.astype(bool)
    f2_bool = f2.astype(bool)
    intersection = np.sum(f1_bool & f2_bool)
    union = np.sum(f1_bool | f2_bool)
    return intersection / union if union > 0 else 0.0


def compute_feature_stats(y: np.ndarray, feature_values: np.ndarray, 
                          threshold: Optional[float] = None) -> Dict:
    """
    Compute precision, coverage, and lift for a feature.
    
    Args:
        y: Binary target labels
        feature_values: Feature values to evaluate
        threshold: Optional threshold for continuous features
        
    Returns:
        Dictionary with precision, coverage, lift, and counts
    """
    base_rate = y.mean()
    
    if threshold is not None:
        applies = feature_values > threshold
    else:
        applies = feature_values.astype(bool)
    
    coverage = np.mean(applies)
    n_applies = np.sum(applies)
    
    if n_applies > 0:
        precision = y[applies].mean()
        lift = precision / base_rate if base_rate > 0 else 0
    else:
        precision = 0.0
        lift = 0.0
    
    return {
        "precision": precision,
        "coverage": coverage,
        "lift": lift,
        "n_applies": int(n_applies),
        "n_success_applies": int(y[applies].sum()) if n_applies > 0 else 0,
        "base_rate": base_rate
    }


def remove_redundant_features(X: pd.DataFrame, y: pd.Series, 
                              threshold: float = 0.8,
                              verbose: bool = True) -> Tuple[pd.DataFrame, List]:
    """
    Remove features with high Jaccard similarity (redundant features).
    
    Features are sorted by lift and kept if they don't have Jaccard
    similarity > threshold with any already-kept feature.
    
    Args:
        X: Feature DataFrame
        y: Target labels
        threshold: Maximum Jaccard similarity allowed
        verbose: Whether to print summary
        
    Returns:
        Tuple of (filtered DataFrame, list of removed features with reasons)
    """
    # Compute lift for each feature
    lifts = {}
    for col in X.columns:
        stats = compute_feature_stats(y.values, X[col].fillna(0).values)
        lifts[col] = stats['lift']
    
    # Sort columns by lift (highest first)
    sorted_cols = sorted(X.columns, key=lambda c: lifts[c], reverse=True)
    
    kept_features = []
    removed_features = []
    
    for col in sorted_cols:
        is_redundant = False
        for kept in kept_features:
            sim = compute_jaccard_similarity(
                X[col].fillna(0).values,
                X[kept].fillna(0).values
            )
            if sim > threshold:
                is_redundant = True
                removed_features.append((col, kept, sim))
                break
        
        if not is_redundant:
            kept_features.append(col)
    
    if verbose:
        print(f"Redundancy removal: {len(X.columns)} → {len(kept_features)} features")
    
    return X[kept_features], removed_features


def extract_education_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Extract education-related features from founder profiles.
    
    Features include:
    - Number of degrees
    - Highest degree score
    - QS ranking of best school
    - Various binary flags (PhD, MBA, STEM, etc.)
    
    Args:
        df: DataFrame with 'educations_json' column
        verbose: Whether to print progress
        
    Returns:
        DataFrame with education features
    """
    features = []
    
    if verbose:
        print("Extracting education features...")
    
    for idx, row in df.iterrows():
        edu_data = parse_json_column(row.get('educations_json', '[]'))
        
        degrees = [e.get('degree', '') for e in edu_data if e.get('degree')]
        fields = [e.get('field', '') for e in edu_data if e.get('field')]
        qs_ranks = [parse_qs_rank(e.get('qs_ranking')) for e in edu_data]
        qs_ranks = [r for r in qs_ranks if r < 999]
        
        # Degree analysis
        degree_score, has_phd, has_mba, has_masters = 0, 0, 0, 0
        for d in degrees:
            d_lower = d.lower()
            if 'phd' in d_lower or 'doctor' in d_lower:
                degree_score = max(degree_score, 4)
                has_phd = 1
            elif 'mba' in d_lower:
                degree_score = max(degree_score, 3)
                has_mba = 1
            elif 'master' in d_lower or 'msc' in d_lower:
                degree_score = max(degree_score, 2)
                has_masters = 1
        
        # Field analysis
        stem_kw = ['computer', 'engineering', 'math', 'physics', 'science', 'data']
        business_kw = ['business', 'mba', 'economics', 'finance', 'management']
        is_stem = any(any(kw in f.lower() for kw in stem_kw) for f in fields)
        is_business = any(any(kw in f.lower() for kw in business_kw) for f in fields)
        
        # QS ranking
        best_qs = min(qs_ranks) if qs_ranks else 999
        
        features.append({
            'edu_num_degrees': len(degrees),
            'edu_highest_degree_score': degree_score,
            'edu_best_qs_rank': best_qs if best_qs < 999 else np.nan,
            'edu_is_top10_school': int(best_qs <= 10),
            'edu_is_top50_school': int(best_qs <= 50),
            'edu_is_top100_school': int(best_qs <= 100),
            'edu_has_phd': has_phd,
            'edu_has_mba': has_mba,
            'edu_has_masters': has_masters,
            'edu_has_advanced_degree': int(has_phd or has_mba or has_masters),
            'edu_is_stem': int(is_stem),
            'edu_is_business': int(is_business),
            'edu_is_stem_and_business': int(is_stem and is_business),
        })
    
    result = pd.DataFrame(features)
    if verbose:
        print(f"  ✓ Extracted {len(result.columns)} education features")
    return result


def extract_job_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Extract job-related features from founder profiles.
    
    Features include:
    - Number of prior jobs
    - Senior role counts
    - Role type distributions
    - Company size experience
    - Total years of experience
    
    Args:
        df: DataFrame with 'jobs_json' column
        verbose: Whether to print progress
        
    Returns:
        DataFrame with job features
    """
    features = []
    
    if verbose:
        print("Extracting job features...")
    
    for idx, row in df.iterrows():
        jobs_data = parse_json_column(row.get('jobs_json', '[]'))
        
        num_jobs = len(jobs_data)
        roles = [j.get('role', '') for j in jobs_data]
        industries = [j.get('industry', '') for j in jobs_data if j.get('industry')]
        durations = [j.get('duration', '') for j in jobs_data]
        company_sizes = [j.get('company_size', '') for j in jobs_data]
        
        # Seniority analysis
        num_cxo = sum(1 for r in roles if any(kw in r.lower() for kw in ['ceo', 'cto', 'cfo', 'chief']))
        num_founder = sum(1 for r in roles if any(kw in r.lower() for kw in ['founder', 'co-founder']))
        num_vp = sum(1 for r in roles if any(kw in r.lower() for kw in ['vp', 'vice president']))
        num_director = sum(1 for r in roles if any(kw in r.lower() for kw in ['director', 'head of']))
        total_senior = num_cxo + num_founder + num_vp + num_director
        
        # Role types
        num_tech = sum(1 for r in roles if any(kw in r.lower() for kw in ['engineer', 'developer', 'scientist']))
        num_product = sum(1 for r in roles if any(kw in r.lower() for kw in ['product', 'pm', 'ux']))
        num_business = sum(1 for r in roles if any(kw in r.lower() for kw in ['sales', 'marketing', 'business']))
        
        # Company size analysis
        big_co_kw = ['5001', '10001', '10000+', '1001-5000']
        startup_kw = ['1-10', '11-50', '51-200']
        has_big_co = any(any(kw in str(cs) for kw in big_co_kw) for cs in company_sizes if cs)
        has_startup = any(any(kw in str(cs) for kw in startup_kw) for cs in company_sizes if cs)
        
        total_years = sum(parse_duration(d) for d in durations)
        unique_industries = len(set(industries))
        
        features.append({
            'job_num_prior_jobs': num_jobs,
            'job_num_senior_roles': total_senior,
            'job_num_cxo_roles': num_cxo,
            'job_num_founder_roles': num_founder,
            'job_num_tech_roles': num_tech,
            'job_num_product_roles': num_product,
            'job_num_business_roles': num_business,
            'job_total_experience_years': total_years,
            'job_num_industries': unique_industries,
            'job_has_cxo_experience': int(num_cxo > 0),
            'job_has_prior_founder_exp': int(num_founder > 0),
            'job_has_big_company_exp': int(has_big_co),
            'job_has_startup_exp': int(has_startup),
            'job_is_technical': int(num_tech > 0),
            'job_is_technical_senior': int(num_tech > 0 and total_senior > 0),
            'job_is_repeat_founder': int(num_founder >= 2),
            'job_big_company_then_startup': int(has_big_co and has_startup),
        })
    
    result = pd.DataFrame(features)
    if verbose:
        print(f"  ✓ Extracted {len(result.columns)} job features")
    return result


def extract_all_baseline_features(
    df: pd.DataFrame,
    remove_redundant: bool = True,
    redundancy_threshold: float = 0.6,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract all baseline features from founder DataFrame.
    
    Args:
        df: Raw founder DataFrame
        remove_redundant: Whether to remove redundant features
        redundancy_threshold: Jaccard similarity threshold
        verbose: Whether to print progress
        
    Returns:
        Tuple of (feature DataFrame, target Series)
    """
    # Extract features
    edu_features = extract_education_features(df, verbose=verbose)
    job_features = extract_job_features(df, verbose=verbose)
    
    # Combine features
    X_baseline = pd.concat([edu_features, job_features], axis=1)
    y = df['success']
    
    if verbose:
        print(f"\nTotal baseline features: {len(X_baseline.columns)}")
    
    # Remove redundant features if requested
    if remove_redundant:
        X_baseline, _ = remove_redundant_features(
            X_baseline, y, 
            threshold=redundancy_threshold,
            verbose=verbose
        )
    
    return X_baseline, y
