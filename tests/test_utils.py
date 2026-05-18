import pytest

from fixed_noise_diffusion.utils import make_run_dir


def test_make_run_dir_refuses_existing_nonempty_run(tmp_path):
    run_dir = make_run_dir(tmp_path, "run")
    stale_metrics = run_dir / "metrics.csv"
    stale_metrics.write_text("stale\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to append"):
        make_run_dir(tmp_path, "run")

    assert stale_metrics.read_text(encoding="utf-8") == "stale\n"


def test_make_run_dir_overwrite_replaces_existing_run(tmp_path):
    run_dir = make_run_dir(tmp_path, "run")
    (run_dir / "metrics.csv").write_text("stale\n", encoding="utf-8")

    replacement = make_run_dir(tmp_path, "run", overwrite=True)

    assert replacement == run_dir
    assert not (replacement / "metrics.csv").exists()
    assert (replacement / "checkpoints").is_dir()
    assert (replacement / "samples").is_dir()
