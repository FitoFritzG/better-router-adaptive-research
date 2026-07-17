from scripts.verify_professor_submission import verify_tracked_files


def test_submission_verifier_allows_empty_directory_markers() -> None:
    tracked = (
        "data/raw/.gitkeep",
        "data/interim/.gitkeep",
        "data/processed/.gitkeep",
    )

    assert verify_tracked_files(tracked) == []


def test_submission_verifier_rejects_real_data_and_secrets() -> None:
    tracked = (
        "data/raw/routerbench.pkl",
        "data/processed/results.csv",
        ".env",
        "src/better_router_adaptive/config.py",
    )

    assert verify_tracked_files(tracked) == [
        ".env",
        "data/processed/results.csv",
        "data/raw/routerbench.pkl",
    ]
