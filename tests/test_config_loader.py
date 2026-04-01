from pathlib import Path

from utils.config import load_config


def test_load_config_reads_yaml(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("learning_rate: 0.001\nname: demo\n", encoding="utf-8")

    cfg = load_config(str(cfg_file))

    assert cfg["learning_rate"] == 0.001
    assert cfg["name"] == "demo"
