from sports.nfl.predictions.pipeline import TargetSpec, build_features
from sports.nfl.tests.test_pipeline import make_stats


def test_touchdown_target_uses_role_context():
    spec = TargetSpec("pass_tds", "Passing touchdowns", "passing_tds", "attempts", 10.0, 0.5)
    frame, features = build_features(make_stats(seasons=[2024], players=2, weeks=6), spec)
    assert "passing_tds_roll3" in features
    assert "completions_roll3" in features
    assert frame["baseline_prediction"].equals(frame["passing_tds_roll5"])
