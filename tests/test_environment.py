"""Environment and baseline configuration sanity tests."""

from pathlib import Path
import yaml


def test_configs_exist():
    configs_dir = Path("configs")
    assert configs_dir.is_dir(), "configs/ directory must exist"

    derm_cfg = configs_dir / "dermoscopy.yaml"
    assert derm_cfg.exists(), "configs/dermoscopy.yaml must exist"

    smart_cfg = configs_dir / "smartphone_clinical.yaml"
    assert smart_cfg.exists(), "configs/smartphone_clinical.yaml must exist"


def test_dermoscopy_config_structure():
    with open("configs/dermoscopy.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["profile"] == "dermoscopy"
    assert "classes" in data and len(data["classes"]) >= 6
    assert "thresholds" in data
    assert "tau_abstain" in data["thresholds"]
    assert "tau_referral_melanoma" in data["thresholds"]


def test_smartphone_config_structure():
    with open("configs/smartphone_clinical.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["profile"] == "smartphone_clinical"
    assert "classes" in data and len(data["classes"]) == 6
    assert "thresholds" in data
    assert "tau_abstain" in data["thresholds"]
    assert "tau_referral_melanoma" in data["thresholds"]


def test_agents_and_docs_exist():
    assert Path("docs/ARCHITECTURE.md").exists()
    assert Path("docs/DECISIONS.md").exists()
    assert Path("docs/STATUS.md").exists()
    assert Path("docs/DATA_GOVERNANCE.md").exists()
