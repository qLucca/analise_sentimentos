from src.database.connection import build_connection_string


def test_build_connection_string_for_mysql(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "mysql")
    monkeypatch.setenv("DB_HOST", "rds.example.amazonaws.com")
    monkeypatch.setenv("DB_PORT", "3306")
    monkeypatch.setenv("DB_NAME", "analytics")
    monkeypatch.setenv("DB_USER", "app_user")
    monkeypatch.setenv("DB_PASSWORD", "safe-password-123")

    connection_string = build_connection_string()

    assert connection_string == (
        "mysql+pymysql://app_user:safe-password-123@"
        "rds.example.amazonaws.com:3306/analytics"
    )
