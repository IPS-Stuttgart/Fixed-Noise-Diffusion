import json

import pytest

from fixed_noise_diffusion.persistence_protocol import source_hash
from fixed_noise_diffusion.persistence_study import conditions_for, exclusive_lock, initialize, study_manifest


def test_registration_requires_current_gpu_smoke_receipt(tmp_path):
    with pytest.raises(ValueError, match="smoke receipt"):
        initialize(tmp_path / "missing")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"passed": True, "gpu": False, "source_sha256": source_hash()}))
    with pytest.raises(ValueError, match="CPU-only"):
        initialize(tmp_path / "cpu", smoke_receipt=receipt)
    receipt.write_text(json.dumps({"passed": True, "gpu": True, "source_sha256": source_hash()}))
    initialize(tmp_path / "study", smoke_receipt=receipt)
    assert study_manifest(tmp_path / "study")["protocol"]["max_training_runs"] == 18


def test_smoke_cannot_authorize_confirmation_or_extra_replicates(tmp_path):
    initialize(tmp_path, smoke=True)
    assert len(conditions_for(tmp_path, 0)) == 6
    with pytest.raises(ValueError, match="cannot authorize"):
        conditions_for(tmp_path, 1)
    with pytest.raises(ValueError, match="budget"):
        conditions_for(tmp_path, 4)


def test_mutated_manifest_is_rejected(tmp_path):
    initialize(tmp_path, smoke=True)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["protocol"]["ordering_margin"] = 0
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="modified"):
        study_manifest(tmp_path)


def test_run_lock_prevents_duplicate_worker(tmp_path):
    with exclusive_lock(tmp_path / "worker.lock"):
        with pytest.raises(RuntimeError, match="Another worker"):
            with exclusive_lock(tmp_path / "worker.lock"):
                pass
