"""
utils.py - Shared utility functions

Common utilities used across all modules.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def safe_float(x: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        if x is None:
            return default
        return float(x)
    except (ValueError, TypeError):
        return default


def safe_int(x: Any, default: int = 0) -> int:
    """Safely convert value to int"""
    try:
        if x is None:
            return default
        # Handle strings with commas
        if isinstance(x, str):
            x = x.replace(',', '')
        return int(float(x))
    except (ValueError, TypeError):
        return default


def safe_str(x: Any, default: str = "") -> str:
    """Safely convert value to string"""
    if x is None:
        return default
    return str(x)


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value between bounds"""
    return max(lo, min(hi, x))


def normalize(x: float, min_val: float, max_val: float) -> float:
    """Normalize value to 0-1 range"""
    if max_val == min_val:
        return 0.5
    return clamp((x - min_val) / (max_val - min_val), 0.0, 1.0)


def load_json(path: Path) -> Any:
    """Load JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj: Any, path: Path, indent: int = 2) -> None:
    """Save object as JSON file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent)


def sanitize_filename(s: str, max_length: int = 200) -> str:
    """Sanitize string for use as filename"""
    bad_chars = '<>:"/\\|?*'
    sanitized = ''.join('_' if c in bad_chars else c for c in str(s).strip())
    sanitized = sanitized.replace(' ', '_')
    return sanitized[:max_length] if sanitized else 'unknown'


def format_currency(amount: float, decimals: int = 2) -> str:
    """Format amount as currency string"""
    return f"${amount:,.{decimals}f}M"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage string"""
    return f"{value * 100:.{decimals}f}%"


def calculate_percentile(value: float, values: List[float]) -> float:
    """Calculate percentile rank of value in list"""
    if not values:
        return 0.5
    
    sorted_values = sorted(values)
    count_below = sum(1 for v in sorted_values if v < value)
    
    return count_below / len(sorted_values)


def weighted_average(values: List[float], weights: List[float]) -> float:
    """Calculate weighted average"""
    if not values or not weights or len(values) != len(weights):
        return 0.0
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def moving_average(values: List[float], window: int = 3) -> List[float]:
    """Calculate moving average"""
    if len(values) < window:
        return values
    
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        end = i + 1
        window_values = values[start:end]
        result.append(sum(window_values) / len(window_values))
    
    return result


def find_files(directory: Path, pattern: str = "*.json", recursive: bool = False) -> List[Path]:
    """Find files matching pattern in directory"""
    if not directory.exists():
        return []
    
    if recursive:
        return sorted(directory.rglob(pattern))
    else:
        return sorted(directory.glob(pattern))


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, create if needed"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text_file(path: Path) -> str:
    """Read text file"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def write_text_file(content: str, path: Path) -> None:
    """Write text file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def merge_dicts(dict1: Dict, dict2: Dict, deep: bool = True) -> Dict:
    """Merge two dictionaries"""
    if not deep:
        return {**dict1, **dict2}
    
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value, deep=True)
        else:
            result[key] = value
    
    return result


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def group_by(items: List[Dict], key: str) -> Dict[Any, List[Dict]]:
    """Group list of dicts by key"""
    groups = {}
    for item in items:
        group_key = item.get(key)
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(item)
    return groups


def sort_by(items: List[Dict], key: str, reverse: bool = False) -> List[Dict]:
    """Sort list of dicts by key"""
    return sorted(items, key=lambda x: x.get(key, 0), reverse=reverse)


def filter_by(items: List[Dict], key: str, value: Any) -> List[Dict]:
    """Filter list of dicts by key-value pair"""
    return [item for item in items if item.get(key) == value]


def summarize_stats(values: List[float]) -> Dict[str, float]:
    """Calculate summary statistics"""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0
        }
    
    import statistics
    
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0
    }
