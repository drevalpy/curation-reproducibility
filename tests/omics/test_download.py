"""Tests for the raw-data download coordination."""

import pytest
from upath import UPath

from data_curation.core import layout
from data_curation.core.errors import CurationError
from data_curation.omics import download

FAKE_SOURCES = {
    "model_list": {"url": "https://x/model_list.csv", "known_hash": "sha256:aa", "fname": "model_list.csv"},
    "rnaseq": {"url": "https://x/rnaseq.zip", "known_hash": "sha256:bb", "keep": ["*tpm*"]},
    "mutations": {"url": "https://x/mutations.zip", "known_hash": "sha256:cc", "keep": ["*mutations_all*"]},
    "cnv": {"url": "https://x/cnv.zip", "known_hash": "sha256:dd", "keep": ["*gistic*"]},
    "proteomics": {"url": "https://x/proteomics.csv.gz", "known_hash": "sha256:ee", "keep": ["*proteomics*"]},
}


@pytest.fixture
def recorded(monkeypatch, tmp_path):
    """Run in a scratch cwd with the network-facing helpers replaced by recorders."""
    monkeypatch.chdir(tmp_path)
    retrieved: list[tuple[list[dict], UPath]] = []
    pruned: list[tuple[UPath, list[str]]] = []
    monkeypatch.setattr(download, "cmp_source", FAKE_SOURCES.__getitem__)
    monkeypatch.setattr(download, "retrieve_file_list", lambda files, out: retrieved.append((files, out)))
    monkeypatch.setattr(download, "prune_unused", lambda out, keep: pruned.append((out, list(keep))))
    return retrieved, pruned


class TestDownloadCmp:
    def test_each_kind_is_downloaded_once_into_its_own_directory(self, recorded) -> None:
        retrieved, _ = recorded

        download.download_cmp()

        assert [out for _, out in retrieved] == [layout.cmp_raw(kind) for kind in layout.CMP_KINDS]
        assert all(out.is_dir() for _, out in retrieved)

    def test_the_pinned_url_hash_and_name_reach_the_retriever(self, recorded) -> None:
        retrieved, _ = recorded

        download.download_cmp(("model_list", "cnv"))

        assert retrieved[0][0] == [
            {"url": "https://x/model_list.csv", "fname": "model_list.csv", "known_hash": "sha256:aa"}
        ]
        assert retrieved[1][0] == [{"url": "https://x/cnv.zip", "fname": None, "known_hash": "sha256:dd"}]

    def test_unused_extracted_files_are_pruned_per_kind(self, recorded) -> None:
        _, pruned = recorded

        download.download_cmp(("cnv", "model_list"))

        assert pruned == [(layout.cmp_raw("cnv"), ["*gistic*"]), (layout.cmp_raw("model_list"), [])]

    def test_a_subset_of_kinds_can_be_requested(self, recorded) -> None:
        retrieved, _ = recorded

        download.download_cmp(("proteomics",))

        assert [out for _, out in retrieved] == [layout.cmp_raw("proteomics")]


class TestDownloadOmicsData:
    @pytest.mark.parametrize("key", ["transcriptomics", "mutations", "cnv", "proteomics"])
    def test_cmp_omics_are_refused_per_dataset(self, key: str, tmp_dir: UPath) -> None:
        with pytest.raises(CurationError, match="download-cmp"):
            download.download_omics_data("GDSC2", key, tmp_dir)

    def test_a_cmp_dataset_still_downloads_its_own_response(self, monkeypatch, tmp_dir: UPath) -> None:
        calls: list[tuple[str, object]] = []
        monkeypatch.setattr(download, "retrieve_file_list", lambda files, out: calls.append(("files", out)))
        monkeypatch.setattr(download, "retrieve_named_sources", lambda params, out: calls.append(("named", out)))
        monkeypatch.setattr(download, "prune_unused", lambda out, keep: calls.append(("prune", out)))

        download.download_omics_data("GDSC2", "response", tmp_dir)

        assert [name for name, _ in calls] == ["files", "named", "prune"]
