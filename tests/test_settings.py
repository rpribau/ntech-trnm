from config.settings import get_settings


def test_paths_are_absolute():
    s = get_settings()
    assert s.repos_dir.is_absolute()
    assert s.index_dir.is_absolute()
    assert s.guidelines_dir.is_absolute()


def test_active_model_switches_with_backend():
    s = get_settings()
    expected = {
        "cloudrun": s.cloudrun_model,
        "anthropic": s.anthropic_model,
        "ollama": s.ollama_model,
    }[s.llm_backend]
    assert s.active_model == expected


def test_derived_paths_under_index_dir():
    s = get_settings()
    assert s.bm25_docs_path.parent == s.index_dir
    assert s.checkpointer_path.parent == s.index_dir
