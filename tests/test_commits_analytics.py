from __future__ import annotations

from datetime import date

from ntech_agent.commits.analytics import (
    change_type_breakdown,
    combined_table,
    commits_only,
    daily_activity,
    filter_rows,
    metadata_table,
    summary_table,
    to_dataframe,
    top_contributors,
    top_files,
    weekly_activity,
)

_ROWS = [
    {
        "repo": "ws-arg",
        "commit_sha": "aaaaaaaa1111",
        "file_path": "a.py",
        "author": "dev-a",
        "committed_at": "2026-06-01T10:00:00+00:00",
        "commit_message": "fix bug",
        "change_type": "modified",
        "summary": "Corrige un bug de parseo.",
    },
    {
        "repo": "ws-arg",
        "commit_sha": "aaaaaaaa1111",
        "file_path": "b.py",
        "author": "dev-a",
        "committed_at": "2026-06-01T10:00:00+00:00",
        "commit_message": "fix bug",
        "change_type": "added",
        "summary": None,
    },
    {
        "repo": "ws-arg",
        "commit_sha": "bbbbbbbb2222",
        "file_path": "a.py",
        "author": "dev-b",
        "committed_at": "2026-06-03T09:00:00+00:00",
        "commit_message": "refactor mlflow tracking",
        "change_type": "modified",
        "summary": "Refactor de MLflow.",
    },
]


def test_to_dataframe_empty_keeps_expected_columns():
    df = to_dataframe([])
    assert df.empty
    assert list(df.columns) == [
        "repo", "commit_sha", "file_path", "author", "committed_at",
        "commit_message", "change_type", "summary",
    ]


def test_to_dataframe_parses_dates():
    df = to_dataframe(_ROWS)
    assert len(df) == 3
    expected = [date(2026, 6, 1), date(2026, 6, 1), date(2026, 6, 3)]
    assert df["committed_at"].dt.date.tolist() == expected


def test_filter_rows_by_date_range():
    df = to_dataframe(_ROWS)
    filtered = filter_rows(df, since=date(2026, 6, 2), until=date(2026, 6, 3))
    assert set(filtered["commit_sha"]) == {"bbbbbbbb2222"}


def test_filter_rows_by_authors():
    df = to_dataframe(_ROWS)
    filtered = filter_rows(df, authors=["dev-b"])
    assert set(filtered["author"]) == {"dev-b"}


def test_filter_rows_by_search_matches_message_file_or_summary():
    df = to_dataframe(_ROWS)
    assert set(filter_rows(df, search="mlflow")["commit_sha"]) == {"bbbbbbbb2222"}
    assert set(filter_rows(df, search="b.py")["commit_sha"]) == {"aaaaaaaa1111"}
    assert set(filter_rows(df, search="parseo")["commit_sha"]) == {"aaaaaaaa1111"}


def test_filter_rows_on_empty_df_returns_empty():
    assert filter_rows(to_dataframe([]), search="x").empty


def test_commits_only_dedupes_by_sha():
    df = to_dataframe(_ROWS)
    commits = commits_only(df)
    assert len(commits) == 2  # aaaa (2 archivos) + bbbb (1 archivo) = 2 commits


def test_daily_activity_fills_gap_days_with_zero():
    df = to_dataframe(_ROWS)
    daily = daily_activity(df)
    assert daily["date"].dt.date.tolist() == [date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)]
    assert daily["commits"].tolist() == [1, 0, 1]


def test_daily_activity_empty_df():
    daily = daily_activity(to_dataframe([]))
    assert daily.empty
    assert list(daily.columns) == ["date", "commits"]


def test_weekly_activity_total_matches_commit_count():
    df = to_dataframe(_ROWS)
    weekly = weekly_activity(df)
    assert weekly["commits"].sum() == 2  # mismo total que commits_only


def test_top_contributors_sorted_desc_and_limited():
    df = to_dataframe(_ROWS)
    top = top_contributors(df, limit=1)
    assert len(top) == 1
    assert top.iloc[0]["author"] == "dev-a"
    assert top.iloc[0]["commits"] == 1


def test_top_files_counts_distinct_commits_not_rows():
    df = to_dataframe(_ROWS)
    top = top_files(df)
    a_py = top[top["file_path"] == "a.py"].iloc[0]
    assert a_py["commits"] == 2  # a.py aparece en los 2 commits distintos
    b_py = top[top["file_path"] == "b.py"].iloc[0]
    assert b_py["commits"] == 1


def test_change_type_breakdown_labels_and_counts():
    df = to_dataframe(_ROWS)
    breakdown = change_type_breakdown(df)
    counts = dict(zip(breakdown["change_type"], breakdown["files"], strict=True))
    assert counts == {"Modificado": 2, "Agregado": 1}


def test_change_type_breakdown_unknown_type_falls_back_to_raw_value():
    df = to_dataframe(
        [{**_ROWS[0], "change_type": "copied"}]
    )
    breakdown = change_type_breakdown(df)
    assert breakdown.iloc[0]["change_type"] == "Copiado"


def test_metadata_table_has_short_sha_and_no_summary_column():
    df = to_dataframe(_ROWS)
    table = metadata_table(df)
    assert "summary" not in table.columns
    assert table.iloc[0]["commit"] == "aaaaaaaa"  # primeros 8 caracteres


def test_summary_table_drops_rows_without_summary():
    df = to_dataframe(_ROWS)
    table = summary_table(df)
    assert len(table) == 2  # la fila de b.py tiene summary=None
    assert set(table["file_path"]) == {"a.py"}


def test_combined_table_has_all_columns():
    df = to_dataframe(_ROWS)
    table = combined_table(df)
    assert set(table.columns) == {
        "commit", "file_path", "author", "committed_at", "commit_message", "repo", "summary",
    }
    assert len(table) == 3
