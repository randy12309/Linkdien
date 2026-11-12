import json
import os

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def login(u, p):
    return client.post("/api/v1/auth/login", json={"username": u, "password": p})


def test_login_success_and_me():
    for user in ("john", "randy"):
        r = login(user, "password")
        assert r.status_code == 200
        token = r.json()["access_token"]
        me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.json() == {"username": user}


def test_login_rejects_bad_credentials():
    assert login("john", "wrong").status_code == 401
    assert login("nobody", "password").status_code == 401


def test_me_requires_valid_token():
    assert client.get("/api/v1/auth/me").status_code == 401
    bad = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer junk"})
    assert bad.status_code == 401


NEW = {"username": "Newbie_1", "email": "New@Example.com", "display_name": "Nina", "password": "s3cret-pass"}


def register(**over):
    return client.post("/api/v1/auth/register", json={**NEW, **over})


def test_register_then_login_and_home():
    r = register(username="nina_one", email="nina1@example.com")
    assert r.status_code == 201 and r.json()["username"] == "nina_one"
    token = r.json()["access_token"]
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}).json() == {"username": "nina_one"}
    assert login("nina_one", "s3cret-pass").status_code == 200
    home = client.get("/api/v1/home", headers={"Authorization": f"Bearer {token}"}).json()
    assert home["welcome"]["display_name"] == "Nina" and home["premium_alert"] is None
    assert home["inbox"]["unread_count"] == 0


def test_register_stores_only_a_hash():
    register(username="hash_user", email="hash@example.com")
    stored = json.loads(open(os.environ["REGISTERED_USERS_FILE"], encoding="utf-8").read())["hash_user"]
    assert "s3cret-pass" not in json.dumps(stored) and stored["password_hash"].startswith("$2")


def test_register_username_is_case_insensitive_and_unique():
    assert register(username="Dup_User", email="dup1@example.com").status_code == 201
    assert register(username="dup_user", email="dup2@example.com").status_code == 409
    assert register(username="other_user", email="DUP1@example.com").status_code == 409


def test_register_cannot_take_a_demo_username():
    assert register(username="Randy", email="r@example.com").status_code == 409


@pytest.mark.parametrize(
    "bad",
    [
        {"username": "ab"},
        {"username": "has space"},
        {"email": "not-an-email"},
        {"password": "short"},
        {"display_name": ""},
        {"display_name": "   "},
    ],
)
def test_register_validates_input(bad):
    assert register(**bad).status_code == 422


def test_wrong_password_for_registered_user_is_rejected():
    register(username="pw_user", email="pw@example.com")
    assert login("pw_user", "wrong-password").status_code == 401
