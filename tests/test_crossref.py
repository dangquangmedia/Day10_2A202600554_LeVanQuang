from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from core.config import load_settings
from core.utils import read_json, write_json
from ingestion.crossref import PaperRecord, fetch_source_records, load_raw_records, parse_crossref_payload


def _payload() -> dict:
    return {
        "message": {
            "items": [
                {
                    "DOI": "10.1234/Example.DOI",
                    "title": ["  Agentic RAG with Data Quality  "],
                    "abstract": "<jats:p>This paper studies reliable RAG pipelines.</jats:p>",
                    "author": [
                        {"given": "Ada", "family": "Lovelace"},
                        {"name": "Grace Hopper"},
                    ],
                    "subject": ["Artificial Intelligence", "Data Engineering"],
                    "published-print": {"date-parts": [[2025, 5, 9]]},
                    "updated": {"date-parts": [[2025, 6, 1]]},
                    "URL": "https://doi.org/10.1234/example.doi",
                    "link": [{"URL": "https://example.org/paper.pdf", "content-type": "application/pdf"}],
                },
                {
                    "DOI": "10.9999/missing-title",
                    "abstract": "No title should be skipped.",
                },
            ]
        }
    }


def test_parse_crossref_payload_normalizes_valid_items() -> None:
    records = parse_crossref_payload(_payload())

    assert records == [
        PaperRecord(
            paper_id="10.1234/example.doi",
            title="Agentic RAG with Data Quality",
            summary="This paper studies reliable RAG pipelines.",
            authors=["Ada Lovelace", "Grace Hopper"],
            categories=["Artificial Intelligence", "Data Engineering"],
            primary_category="Artificial Intelligence",
            published="2025-05-09",
            updated="2025-06-01",
            abs_url="https://doi.org/10.1234/example.doi",
            pdf_url="https://example.org/paper.pdf",
            comment="",
        )
    ]


def test_load_raw_records_maps_json_to_dataclasses(tmp_path: Path) -> None:
    path = tmp_path / "records.json"
    write_json(
        path,
        [
            {
                "paper_id": "10.1234/example",
                "title": "Example",
                "summary": "A useful abstract.",
                "authors": ["Ada Lovelace"],
                "categories": ["AI"],
                "primary_category": "AI",
                "published": "2025-01-02",
                "updated": "2025-01-03",
                "abs_url": "https://doi.org/10.1234/example",
                "pdf_url": "",
                "comment": "",
            }
        ],
    )

    assert load_raw_records(path)[0] == PaperRecord(
        paper_id="10.1234/example",
        title="Example",
        summary="A useful abstract.",
        authors=["Ada Lovelace"],
        categories=["AI"],
        primary_category="AI",
        published="2025-01-02",
        updated="2025-01-03",
        abs_url="https://doi.org/10.1234/example",
        pdf_url="",
        comment="",
    )


def test_fetch_source_records_retries_and_persists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = load_settings()
    paths = replace(
        settings.paths,
        raw_api_response=tmp_path / "crossref_response.json",
        raw_records_json=tmp_path / "crossref_records.json",
    )
    settings = replace(settings, paths=paths, max_results=1)
    calls: list[dict] = []

    class Response:
        def __init__(self, status_code: int, payload: dict | None = None):
            self.status_code = status_code
            self._payload = payload or {}
            self.text = "temporary failure"

        def json(self) -> dict:
            return self._payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    def fake_get(url: str, params: dict, timeout: int) -> Response:
        calls.append({"url": url, "params": params, "timeout": timeout})
        if len(calls) == 1:
            return Response(503)
        return Response(200, _payload())

    monkeypatch.setattr("ingestion.crossref.requests.get", fake_get)
    monkeypatch.setattr("ingestion.crossref.time.sleep", lambda _: None)

    records = fetch_source_records(settings)

    assert len(calls) == 2
    assert calls[0]["params"]["query"] == settings.source_query
    assert calls[0]["params"]["filter"] == settings.source_filter
    assert calls[0]["params"]["rows"] == 1
    assert records[0].paper_id == "10.1234/example.doi"
    assert read_json(paths.raw_api_response)["message"]["items"][0]["DOI"] == "10.1234/Example.DOI"
    assert read_json(paths.raw_records_json)[0]["paper_id"] == "10.1234/example.doi"
