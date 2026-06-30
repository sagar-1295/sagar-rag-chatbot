import importlib
from pathlib import Path

import pytest

from ai_student_impact.rag import RagChatbot


def test_get_dataset_path_default():
    bot = RagChatbot(dataset_path=None)
    assert Path(bot.dataset_path).exists()


def test_trim_history():
    bot = RagChatbot(dataset_path=None)
    history = [(f"q{i}", f"a{i}") for i in range(10)]
    trimmed = bot._trim_history(history, max_turns=5)
    assert len(trimmed) == 5
    assert trimmed[0] == ("q5", "a5")


def test_load_openai_key_from_dotenv(monkeypatch):
    env_path = Path(__file__).resolve().parents[1] / ".env"
    backup = env_path.read_text(encoding="utf-8") if env_path.exists() else None
    env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        import ai_student_impact.config as config

        importlib.reload(config)
        assert config.OPENAI_API_KEY == "test-key"
    finally:
        if backup is None:
            env_path.unlink(missing_ok=True)
        else:
            env_path.write_text(backup, encoding="utf-8")


def test_empty_dataset_directory_fails_clearly(tmp_path):
    empty_dir = tmp_path / "empty_dataset"
    empty_dir.mkdir()
    (empty_dir / "sample.csv").write_text("student_id,impact\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No readable content"):
        RagChatbot(dataset_path=str(empty_dir), max_rows=10, model="gpt-5-mini")
