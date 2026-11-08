"""Unit tests for Glasgow 7-Point checklist scoring and Bayesian prior fusion."""

from ml.anamnesis.questionnaire import (
    AnamnesisEngine,
    GlasgowChecklistInput,
)


def test_glasgow_scoring_high_risk():
    # 2 Major (2+2=4) + 1 Minor (1) = 5 points -> High risk
    inp = GlasgowChecklistInput(
        change_in_size=True,
        irregular_color=True,
        oozing_or_bleeding=True,
        patient_age=65,
    )
    res = AnamnesisEngine.evaluate_glasgow(inp)

    assert res.score == 5
    assert res.major_score == 4
    assert res.minor_score == 1
    assert res.is_high_risk is True
    assert res.risk_level == "high"
    assert len(res.positive_features) == 3


def test_glasgow_scoring_low_risk():
    # All false / negative -> 0 points -> Low risk
    inp = GlasgowChecklistInput(
        change_in_size=False,
        irregular_shape=False,
        irregular_color=False,
        diameter_ge_7mm=False,
        patient_age=25,
    )
    res = AnamnesisEngine.evaluate_glasgow(inp)

    assert res.score == 0
    assert res.is_high_risk is False
    assert res.risk_level == "low"


def test_bayesian_fusion_preserves_pure_probabilities_and_sets_referral():
    # Model had ambiguous probabilities
    prior = {"bcc": 0.35, "ack": 0.25, "mel": 0.15, "nev": 0.25}

    # Patient reports lesion grew rapidly and is bleeding (Glasgow score >= 3)
    anamnesis = GlasgowChecklistInput(
        change_in_size=True,
        oozing_or_bleeding=True,
        irregular_shape=True,
        patient_age=68,
    )

    fused = AnamnesisEngine.apply_bayesian_fusion(prior, anamnesis)
    # Probabilities remain pure and uncorrupted by heuristic multipliers
    assert fused["fused_probabilities"] == prior
    assert fused["referral_recommended"] is True
    assert fused["checklist_evaluation"]["is_high_risk"] is True
    assert fused["triage"]["tier"] == "urgent_referral"


def test_triage_decision_tiers():
    # Test urgent referral via vision threshold
    urgent_vis = AnamnesisEngine.determine_triage(
        image_probabilities={"mel": 0.25, "nev": 0.75},
        anamnesis=None,
    )
    assert urgent_vis.tier == "urgent_referral"
    assert urgent_vis.referral_recommended is True
    assert urgent_vis.urgency_days == 14

    # Test routine consultation via BCC
    routine = AnamnesisEngine.determine_triage(
        image_probabilities={"bcc": 0.40, "nev": 0.60},
        anamnesis=None,
    )
    assert routine.tier == "routine_consultation"
    assert routine.urgency_days == 30

    # Test self monitoring (all negative, benign > 0.8, low entropy)
    benign_inp = GlasgowChecklistInput(
        change_in_size=False,
        irregular_shape=False,
        irregular_color=False,
        diameter_ge_7mm=False,
        inflammation=False,
        oozing_or_bleeding=False,
        change_in_sensation=False,
    )
    self_mon = AnamnesisEngine.determine_triage(
        image_probabilities={"nev": 0.95, "mel": 0.05},
        anamnesis=benign_inp,
        entropy=0.5,
    )
    assert self_mon.tier == "self_monitoring"
    assert self_mon.is_benign_observation is True

    # Test inconclusive quality
    inconclusive = AnamnesisEngine.determine_triage(
        image_probabilities={"nev": 0.90, "mel": 0.10},
        is_blurry=True,
    )
    assert inconclusive.tier == "inconclusive_quality"

