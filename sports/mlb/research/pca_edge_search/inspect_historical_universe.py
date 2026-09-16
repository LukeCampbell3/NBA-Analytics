from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()

    # Read in chunks so the inspection remains cheap even for the ~53 MB corpus.
    first = pd.read_csv(a.input, nrows=25, low_memory=False)
    columns = list(first.columns)
    counts = {c: 0 for c in columns}
    nonnull = {c: 0 for c in columns}
    uniques: dict[str, set[str]] = {c: set() for c in columns if c.lower() in {
        'date','target','direction','market_source','price_confirmed','result','season'
    }}
    rows = 0
    min_date = None
    max_date = None
    real_rows = 0
    price_confirmed_rows = 0
    real_price_confirmed_rows = 0

    for chunk in pd.read_csv(a.input, chunksize=25000, low_memory=False):
        rows += len(chunk)
        for c in columns:
            if c in chunk:
                counts[c] += len(chunk)
                nonnull[c] += int(chunk[c].notna().sum())
        date_col = next((c for c in ['date','prediction_date','event_date','game_date'] if c in chunk), None)
        if date_col:
            d = pd.to_datetime(chunk[date_col], errors='coerce', utc=True)
            if d.notna().any():
                lo, hi = d.min(), d.max()
                min_date = lo if min_date is None or lo < min_date else min_date
                max_date = hi if max_date is None or hi > max_date else max_date
        real_mask = pd.Series([True] * len(chunk), index=chunk.index)
        if 'market_source' in chunk:
            real_mask = chunk['market_source'].astype(str).str.lower().eq('real')
            real_rows += int(real_mask.sum())
        conf_mask = pd.Series([True] * len(chunk), index=chunk.index)
        if 'price_confirmed' in chunk:
            conf_mask = chunk['price_confirmed'].astype(str).str.lower().isin({'1','true','t','yes','y'})
            price_confirmed_rows += int(conf_mask.sum())
        real_price_confirmed_rows += int((real_mask & conf_mask).sum())
        for c in uniques:
            if c in chunk and len(uniques[c]) < 500:
                uniques[c].update(str(x) for x in chunk[c].dropna().unique()[:500])

    candidate_features = [
        'history_rows','days_since_history','historical_bet_profile_win_rate',
        'historical_bet_profile_support','historical_market_availability_rate',
        'historical_market_availability_support','books','line','side_price',
        'actual','result','player_id','game_id','target','direction','market_source',
        'price_confirmed','units','american_odds','odds','quoted_odds',
        'model_hit_probability','hit_probability','probability','selection_score','abs_edge'
    ]
    feature_presence = {
        c: {
            'present': c in columns,
            'nonnull': nonnull.get(c, 0),
            'nonnull_rate': (nonnull.get(c, 0) / rows if rows else 0.0),
        }
        for c in candidate_features
    }
    sample_records = first.head(5).where(pd.notna(first), None).to_dict(orient='records')
    out = {
        'rows': rows,
        'columns': columns,
        'column_count': len(columns),
        'min_date': str(min_date) if min_date is not None else None,
        'max_date': str(max_date) if max_date is not None else None,
        'real_rows': real_rows,
        'price_confirmed_rows': price_confirmed_rows,
        'real_price_confirmed_rows': real_price_confirmed_rows,
        'feature_presence': feature_presence,
        'selected_unique_values': {k: sorted(v)[:100] for k,v in uniques.items()},
        'sample_records': sample_records,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, default=str) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2, default=str))


if __name__ == '__main__':
    main()
