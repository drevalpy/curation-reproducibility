"""The gene-list fingerprint that makes ``subsample_toy`` depend on drevalpy's lists."""

import json

from upath import UPath

from data_curation.assembly import record_toy_gene_lists as record

LISTS = {"gene_expression": {"a": ("G1", "G2"), "b": ("G3",)}, "mutations": {"a": ("G1", "G2")}}


def test_each_list_is_summarised_by_its_count_and_hash() -> None:
    summary = record.fingerprint(LISTS)

    assert summary["gene_expression"]["a"]["n_genes"] == 2
    assert summary["gene_expression"]["b"]["n_genes"] == 1
    assert summary["gene_expression"]["a"]["sha256"] == summary["mutations"]["a"]["sha256"]
    assert summary["gene_expression"]["a"]["sha256"] != summary["gene_expression"]["b"]["sha256"]


def test_the_hash_depends_on_the_gene_order() -> None:
    forward = record.fingerprint({"v": {"a": ("G1", "G2")}})
    backward = record.fingerprint({"v": {"a": ("G2", "G1")}})

    assert forward["v"]["a"]["sha256"] != backward["v"]["a"]["sha256"]


def test_main_writes_the_fingerprint_where_the_layout_says(monkeypatch, tmp_dir: UPath) -> None:
    target = tmp_dir / "assembled" / "toy_gene_lists.json"
    monkeypatch.setattr(record, "FINGERPRINT_PATH", target)
    monkeypatch.setattr(record, "shipped_gene_lists", lambda: LISTS)

    record.main()

    assert json.loads(target.read_text()) == record.fingerprint(LISTS)
