"""Decision and data-isolation contracts for the experimental tractor classifier."""

import numpy as np
import pytest

from scripts.benchmark_tractor_conditions import choose_cutoff, validate_training
from scripts.predict_tractor_conditions import decide, load_head


def head_for(label):
    labels = ["mtz_80", "mtz_82", "mtz_820", "mtz_1221", "nh_t5000", "claas_axion_800", "other"]
    return {
        "classes": labels,
        "exact_policy": {
            name: {"enabled": True, "min_probability": 0.8, "min_similarity": 0.7}
            for name in labels
        },
        "coef": [[12.0 if name == label else 0.0, 0.0] for name in labels],
        "intercept": [0.0] * len(labels),
        "train_reference": [[1.0, 0.0] for _ in labels],
        "train_reference_classes": labels,
        "reject": {"min_probability": 0.8, "min_similarity": 0.7},
    }


@pytest.mark.parametrize("label", ["mtz_80", "mtz_82", "mtz_820"])
def test_even_high_scores_do_not_resolve_ambiguous_small_mtz(label):
    result = decide([1.0, 0.0], head_for(label))
    assert result["score"] > 0.99
    assert result["status"] == "family_only"
    assert result["model"] is None


def test_explicit_other_prediction_never_returns_a_supported_model():
    result = decide([1.0, 0.0], head_for("other"))
    assert result["status"] == "manual_review"
    assert result["model"] is None


def test_supported_match_is_withdrawn_when_reference_similarity_is_low():
    head = head_for("mtz_1221")
    head["train_reference"] = [[0.0, 1.0] for _ in head["classes"]]
    assert decide([1.0, 0.0], head)["model"] is None
    assert decide([1.0, 0.0], head_for("mtz_1221"))["model"] == "mtz_1221"


def test_changed_encoder_is_rejected_before_inference(tmp_path):
    import json

    weights = tmp_path / "weights"
    weights.write_bytes(b"changed weights")
    spec = tmp_path / "head.json"
    spec.write_text(json.dumps({"encoder_sha256": "wrong"}))
    with pytest.raises(ValueError, match="different encoder"):
        load_head(spec, weights)


def test_another_view_of_the_same_machine_cannot_enter_control():
    train = [{"group": "machine-1", "sha256": "photo-1"}]
    control = [{"group": "machine-1", "sha256": "photo-2"}]
    with pytest.raises(ValueError, match="series"):
        validate_training(train, control)


def test_renamed_copy_cannot_enter_control():
    train = [{"group": "machine-1", "sha256": "same-photo"}]
    control = [{"group": "machine-2", "sha256": "same-photo"}]
    with pytest.raises(ValueError, match="image"):
        validate_training(train, control)


def test_explicit_other_rejections_do_not_raise_thresholds_for_known_models():
    known = np.asarray([[0.7, 0.7], [0.8, 0.8], [0.9, 0.9]])
    unknown = np.asarray([[0.999, 0.999]])
    rejected, kept, _, _ = choose_cutoff(known, unknown, np.asarray([True]))
    assert rejected == 1
    assert kept == 1


def test_class_without_independent_support_requires_manual_review():
    head = head_for("claas_axion_800")
    head["exact_policy"]["claas_axion_800"] = {"enabled": False}
    assert decide([1.0, 0.0], head)["model"] is None


def test_fusion_requires_matching_secondary_encoder(tmp_path, monkeypatch):
    import hashlib
    import json

    from scripts import predict_tractor_conditions as predictor

    weights = tmp_path / "primary"
    weights.write_bytes(b"primary")
    secondary = tmp_path / "secondary"
    secondary.write_bytes(b"changed secondary")
    monkeypatch.setattr(predictor, "CONVNEXT_WEIGHTS", secondary)
    spec = tmp_path / "head.json"
    spec.write_text(
        json.dumps(
            {
                "feature_encoder": "fused",
                "encoder_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "secondary_encoder_sha256": "wrong",
            }
        )
    )
    with pytest.raises(ValueError, match="secondary encoder"):
        load_head(spec, weights)


def test_prior_control_cannot_be_moved_into_training():
    from scripts.benchmark_tractor_conditions import reference_digests

    prior = [{"sha256": "same", "model": "mtz_80", "group": "machine", "split": "external_test"}]
    changed = [{**prior[0], "split": "external_train"}]
    with pytest.raises(ValueError, match="prior image"):
        reference_digests(prior, changed)


def test_changed_pixels_invalidate_cached_augmentation(tmp_path):
    from scripts.benchmark_tractor_conditions import reusable_vectors

    path = tmp_path / "features.npz"
    encoder_hash = "a" * 64
    np.savez_compressed(
        path,
        signature=encoder_hash + "|augment-v1-pad224|one:old|two:same",
        keys=np.asarray(["one", "one@0", "two"]),
        values=np.asarray([[1, 0], [2, 0], [3, 0]], dtype=float),
    )
    reused = reusable_vectors(
        path, encoder_hash, {"one": "new", "two": "same"}, ["one", "one@0", "two"], 2
    )
    assert set(reused) == {"two"}
    assert not reusable_vectors(path, "b" * 64, {"two": "same"}, ["two"], 2)
