"""PubChem transport and failure-budget behaviour, with no network access."""

import io
import json
import urllib.error

import pytest
from upath import UPath

from data_curation.core.errors import CurationError
from data_curation.drugs.ontology import pubchem


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pubchem.time, "sleep", lambda _s: None)


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(b""))  # ty: ignore[invalid-argument-type]


_Response = io.BytesIO


def test_request_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pubchem.urllib.request, "urlopen", lambda *_a, **_k: _Response(json.dumps({"a": 1}).encode()))
    assert pubchem._request("http://x") == {"a": 1}


def test_request_raises_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def urlopen(*_args: object, **_kwargs: object) -> None:
        calls.append(1)
        raise _http_error(503)

    monkeypatch.setattr(pubchem.urllib.request, "urlopen", urlopen)
    with pytest.raises(pubchem.PubChemUnavailable, match="503"):
        pubchem._request("http://x")
    assert len(calls) == pubchem._MAX_ATTEMPTS


def test_request_raises_on_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def urlopen(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.URLError("down")

    monkeypatch.setattr(pubchem.urllib.request, "urlopen", urlopen)
    with pytest.raises(pubchem.PubChemUnavailable):
        pubchem._request("http://x")


def test_request_recovers_after_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = [_http_error(503), _Response(b'{"ok": true}')]

    def urlopen(*_args: object, **_kwargs: object) -> _Response:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(pubchem.urllib.request, "urlopen", urlopen)
    assert pubchem._request("http://x") == {"ok": True}


def test_request_not_found_is_empty_and_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def urlopen(*_args: object, **_kwargs: object) -> None:
        calls.append(1)
        raise _http_error(404)

    monkeypatch.setattr(pubchem.urllib.request, "urlopen", urlopen)
    assert pubchem._request("http://x") == {}
    assert len(calls) == 1


def test_request_client_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def urlopen(*_args: object, **_kwargs: object) -> None:
        calls.append(1)
        raise _http_error(400)

    monkeypatch.setattr(pubchem.urllib.request, "urlopen", urlopen)
    with pytest.raises(pubchem.PubChemUnavailable):
        pubchem._request("http://x")
    assert len(calls) == 1


def test_fetch_records_failed_batches_per_cid(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(*_args: object, **_kwargs: object) -> dict:
        raise pubchem.PubChemUnavailable("down")

    monkeypatch.setattr(pubchem, "_request", failing)
    failures: list[tuple[str, str]] = []
    assert pubchem.fetch_inchikeys(["1", "2"], failures) == {}
    assert [query for query, _ in failures] == ["inchikey:1", "inchikey:2"]
    with pytest.raises(pubchem.PubChemUnavailable):
        pubchem.fetch_cross_references(["1"])


def test_budget_at_threshold_passes_and_writes_report(tmp_dir: UPath) -> None:
    report = tmp_dir / "out" / "pubchem_failures.tsv"
    pubchem.enforce_failure_budget([("name:a", "boom")], total=20, max_fraction=0.05, report_path=report)
    assert report.read_text() == "query\treason\nname:a\tboom\n"


def test_budget_just_over_fails_but_still_writes_report(tmp_dir: UPath) -> None:
    report = tmp_dir / "pubchem_failures.tsv"
    failures = [("name:a", "boom"), ("name:b", "multi\tline\nreason")]
    with pytest.raises(CurationError, match=r"2 of 39"):
        pubchem.enforce_failure_budget(failures, total=39, max_fraction=0.05, report_path=report)
    assert report.read_text().splitlines() == ["query\treason", "name:a\tboom", "name:b\tmulti line reason"]


def test_budget_without_failures_writes_header_only(tmp_dir: UPath) -> None:
    report = tmp_dir / "pubchem_failures.tsv"
    pubchem.enforce_failure_budget([], total=0, max_fraction=0.0, report_path=report)
    assert report.read_text() == "query\treason\n"
