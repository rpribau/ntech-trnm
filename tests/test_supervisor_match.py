from ntech_agent.graph.supervisor import _match_repo


def test_exact_match():
    assert _match_repo("ws-arg", ["ws-arg", "otro"]) == "ws-arg"


def test_case_insensitive_match():
    assert _match_repo("WS-ARG", ["ws-arg", "otro"]) == "ws-arg"


def test_fuzzy_match():
    # "wsarg" es cercano a "ws-arg".
    assert _match_repo("wsarg", ["ws-arg", "otro"]) == "ws-arg"


def test_none_returns_none():
    assert _match_repo(None, ["ws-arg"]) is None
