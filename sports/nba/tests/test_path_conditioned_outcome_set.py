from __future__ import annotations
import numpy as np
import pandas as pd

from sports.nba.conditional_chain.outcome_worlds import build_world_distribution
from sports.nba.conditional_chain.path_world_evidence import (
    CHECKPOINT_LABELS,
    build_candidate_evidence_path,
    direct_final_distribution,
    fit_path_evidence_bundle,
    merge_candidates_with_paths,
    path_posteriors,
    transform_path_arrays,
)
from sports.nba.conditional_chain.path_conditioned_backtest import chronological_path_conditioned_replay


def _paths(days=28):
    rows=[]
    players=['A','B','C','D']
    for day in range(days):
        date=pd.Timestamp('2026-01-01')+pd.Timedelta(days=day)
        for i,p in enumerate(players):
            open_share=.30-.04*i
            drift=(.012 if i<2 else -.009)*(1+day%3/10)
            shares=np.array([open_share,open_share+drift*.2,open_share+drift*.45,open_share+drift*.75,open_share+drift])
            lines=20+10*shares
            row={'event_date':date,'event_id':f'e{day}','player':p}
            for m,v in zip((240,120,60,30,5),shares): row[f'share_m{m}']=v
            for m,v in zip((240,120,60,30,5),lines): row[f'line_m{m}']=v
            rows.append(row)
    return pd.DataFrame(rows)


def _reservoir(days=28):
    rows=[]; players=['A','B','C','D']
    for day in range(days):
        date=pd.Timestamp('2026-01-01')+pd.Timedelta(days=day)
        for i,p in enumerate(players):
            # A/B benefit from positive path; C/D from negative, with some noise.
            hit=int((i<2 and day%5!=0) or (i>=2 and day%4==0))
            rows.append({'event_date':date,'event_id':f'e{day}','player':p,'market':'player_points','side':'OVER','line':20+i,'robust_score':.64-.02*i,'leg_result':float(hit)})
    return pd.DataFrame(rows)


def test_incremental_evidence_equals_direct_final_update_and_order_is_stable():
    prior=build_world_distribution(['a','b','c'],[.55,.60,.52])
    post=np.array([[.56,.61,.53],[.57,.62,.54],[.58,.63,.55],[.59,.64,.56],[.60,.65,.57]])
    path=build_candidate_evidence_path(prior,post)
    direct=direct_final_distribution(prior,post[-1])
    assert path.world_path.distributions[-1].candidate_ids==prior.candidate_ids
    np.testing.assert_allclose(path.world_path.distributions[-1].probabilities,direct.probabilities,atol=1e-12)
    np.testing.assert_allclose(path.incremental_world_log_evidence.sum(axis=0),path.cumulative_world_log_evidence[-1],atol=1e-12)


def test_destroyed_controls_preserve_endpoints_but_change_interior_shape():
    shares=np.array([[.20,.22,.18,.25,.24]])
    lines=np.array([[20,21,19,22,21.5]])
    real_s,real_l=transform_path_arrays(shares,lines,mode='real')
    shuffled_s,shuffled_l=transform_path_arrays(shares,lines,mode='shuffled')
    inverted_s,inverted_l=transform_path_arrays(shares,lines,mode='inverted')
    for altered_s,altered_l in ((shuffled_s,shuffled_l),(inverted_s,inverted_l)):
        np.testing.assert_allclose(altered_s[:,[0,-1]],real_s[:,[0,-1]])
        np.testing.assert_allclose(altered_l[:,[0,-1]],real_l[:,[0,-1]])
        assert not np.allclose(altered_s[:,1:-1],real_s[:,1:-1])


def test_evidence_fit_is_strictly_prior_and_future_path_rows_do_not_change_current_model():
    reservoir=_reservoir(); paths=_paths(); cutoff=pd.Timestamp('2026-01-22')
    first=fit_path_evidence_bundle(reservoir,paths,as_of_date=cutoff,mode='real')
    altered=paths.copy(); altered.loc[altered['event_date']>=cutoff,'share_m60']=.99
    second=fit_path_evidence_bundle(reservoir,altered,as_of_date=cutoff,mode='real')
    assert first.history_end_exclusive==cutoff
    assert first.training_rows==second.training_rows
    current=merge_candidates_with_paths(reservoir[reservoir['event_date']==cutoff],paths[paths['event_date']==cutoff],require_complete=False)
    fallback=np.full(len(current),.60)
    np.testing.assert_allclose(path_posteriors(first,current,fallback),path_posteriors(second,current,fallback))


def test_replay_waits_for_20_prior_calibration_slates_and_freezes_t240_target():
    reservoir=_reservoir(28); paths=_paths(28)
    cert={'status':'PATH_INCREMENTAL_VALUE_SUPPORTED','path_authorized':True}
    replay=chronological_path_conditioned_replay(reservoir,paths,path_certificate=cert,block_label='test')
    evaluated=replay.decisions[replay.decisions['evaluated'].fillna(False)]
    assert len(evaluated)==8
    assert (evaluated['evidence_history_end_exclusive']<=evaluated['event_date']).all()
    assert replay.report['target_freeze_checkpoint']==CHECKPOINT_LABELS[0]
    assert replay.report['production_authorized'] is False
    assert replay.selective_risk_report['status']=='NO_PREDECLARED_RISK_TARGET'
    assert not replay.proof_trajectories.empty


def test_same_day_outcomes_cannot_change_same_day_world_evidence():
    reservoir=_reservoir(25); paths=_paths(25); cert={'status':'PATH_INCREMENTAL_VALUE_SUPPORTED','path_authorized':True}
    first=chronological_path_conditioned_replay(reservoir,paths,path_certificate=cert,block_label='first')
    altered=reservoir.copy(); target=pd.Timestamp('2026-01-25'); mask=altered['event_date'].eq(target); altered.loc[mask,'leg_result']=1-altered.loc[mask,'leg_result']
    second=chronological_path_conditioned_replay(altered,paths,path_certificate=cert,block_label='second')
    cols=['candidate_order','evidence_training_rows']
    a=first.decisions[first.decisions['event_date'].eq(target)].iloc[0]
    b=second.decisions[second.decisions['event_date'].eq(target)].iloc[0]
    for col in cols: assert a[col]==b[col]
    # Evidence probabilities are computed before appending the day's realized world.
    ea=first.candidate_evidence[first.candidate_evidence['event_date'].eq(target)].sort_values(['variant','checkpoint','candidate_id'])['posterior_probability'].to_numpy()
    eb=second.candidate_evidence[second.candidate_evidence['event_date'].eq(target)].sort_values(['variant','checkpoint','candidate_id'])['posterior_probability'].to_numpy()
    np.testing.assert_allclose(ea,eb)


def test_endpoint_control_ignores_interior_path_shape():
    reservoir=_reservoir(); paths=_paths(); cutoff=pd.Timestamp('2026-01-22')
    bundle=fit_path_evidence_bundle(reservoir,paths,as_of_date=cutoff,mode='real')
    current_res=reservoir[reservoir['event_date']==cutoff]
    current_paths=paths[paths['event_date']==cutoff].copy()
    merged=merge_candidates_with_paths(current_res,current_paths,require_complete=False)
    fallback=np.full(len(merged),.60)
    from sports.nba.conditional_chain.path_world_evidence import endpoint_posteriors
    first=endpoint_posteriors(bundle,merged,fallback)
    altered=current_paths.copy()
    altered['share_m120']=.01
    altered['share_m60']=.98
    altered['share_m30']=.02
    altered['line_m120']=1.0
    altered['line_m60']=50.0
    altered['line_m30']=2.0
    merged_altered=merge_candidates_with_paths(current_res,altered,require_complete=False)
    second=endpoint_posteriors(bundle,merged_altered,fallback)
    np.testing.assert_allclose(first,second)


def test_initial_history_must_precede_replay_block():
    reservoir=_reservoir(5); paths=_paths(5)
    cert={'status':'PATH_INCREMENTAL_VALUE_SUPPORTED','path_authorized':True}
    overlap=reservoir.iloc[:4].copy()
    overlap_paths=paths.iloc[:4].copy()
    import pytest
    with pytest.raises(ValueError,match='initial_history must end before replay block'):
        chronological_path_conditioned_replay(
            reservoir,
            paths,
            path_certificate=cert,
            block_label='test',
            initial_history=overlap,
        )
    with pytest.raises(ValueError,match='initial_path_history must end before replay block'):
        chronological_path_conditioned_replay(
            reservoir,
            paths,
            path_certificate=cert,
            block_label='test',
            initial_path_history=overlap_paths,
        )


def test_path_certificate_never_authorizes_production():
    reservoir=_reservoir(25); paths=_paths(25)
    replay=chronological_path_conditioned_replay(
        reservoir,
        paths,
        path_certificate={'status':'INSUFFICIENT_REAL_PATH_EVENTS','path_authorized':False},
        block_label='test',
    )
    assert replay.report['production_authorized'] is False
    assert replay.report['status']=='REAL_NBA_PATH_MECHANISM_IMPLEMENTED_INCREMENTAL_VALUE_UNPROVEN'
    evaluated=replay.decisions[replay.decisions['evaluated'].fillna(False)]
    assert not evaluated['path_certificate_authorized'].any()
