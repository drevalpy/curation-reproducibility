"""Tests for the raw-data download coordination."""

import pytest
from upath import UPath

from data_curation.core import layout
from data_curation.core.errors import CurationError
from data_curation.omics import download

FAKE_CCLE = {
    "transcriptomics": {
        "files": [
            {"url": "https://x/tpm.txt.gz", "known_hash": "sha256:11"},
            {"url": "https://x/counts.gct.gz", "known_hash": "sha256:22"},
        ]
    },
    "methylation": {"url": "https://x/rrbs.txt", "known_hash": "sha256:33"},
}

FAKE_SOURCES = {
    "model_list": {"url": "https://x/model_list.csv", "known_hash": "sha256:aa", "fname": "model_list.csv"},
    "gene_identifiers": {"url": "https://x/genes.csv", "known_hash": "sha256:ab", "fname": "gene_identifiers.csv"},
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
    blocks = {"cell_model_passports": FAKE_SOURCES, "ccle": FAKE_CCLE}
    monkeypatch.setattr(download, "shared_source", lambda source, kind: blocks[source][kind])
    monkeypatch.setattr(download, "download_files", lambda files, out: retrieved.append((files, out)))
    monkeypatch.setattr(download, "prune_unused", lambda out, keep: pruned.append((out, list(keep))))
    return retrieved, pruned


class TestDownloadCmp:
    def test_each_kind_is_downloaded_once_into_its_own_directory(self, recorded) -> None:
        retrieved, _ = recorded

        download.download_cmp()

        assert [out for _, out in retrieved] == [layout.cmp_raw(kind) for kind in layout.CMP_KINDS]

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


class TestDownloadShared:
    def test_every_kind_of_the_source_is_downloaded_into_its_directory(self, recorded) -> None:
        retrieved, _ = recorded

        download.download_shared("ccle")

        assert [out for _, out in retrieved] == [
            layout.source_raw("ccle", "transcriptomics"),
            layout.source_raw("ccle", "methylation"),
        ]

    def test_a_files_list_is_passed_on_whole(self, recorded) -> None:
        retrieved, _ = recorded

        download.download_shared("ccle", ("transcriptomics",))

        assert retrieved[0][0] == FAKE_CCLE["transcriptomics"]["files"]

    def test_macos_metadata_from_an_archive_is_removed(self, recorded) -> None:
        junk = layout.source_raw("ccle", "methylation") / "__MACOSX"
        junk.mkdir(parents=True)
        (junk / "._data.txt").write_text("x")

        download.download_shared("ccle", ("methylation",))

        assert not junk.exists()


class TestDownloadOmicsData:
    @pytest.mark.parametrize(
        ("key", "source"),
        [("transcriptomics", "gdsc"), ("methylation", "gdsc"), ("mutations", "cell_model_passports")],
    )
    def test_shared_omics_are_refused_per_dataset(self, key: str, source: str, tmp_dir: UPath) -> None:
        with pytest.raises(CurationError, match=f"download-shared {source}"):
            download.download_omics_data("GDSC2", key, tmp_dir)

    def test_a_cmp_dataset_still_downloads_its_own_response(self, monkeypatch, tmp_dir: UPath) -> None:
        calls: list[tuple[str, object]] = []
        monkeypatch.setattr(download, "download_files", lambda files, out: calls.append(("files", out)))
        monkeypatch.setattr(download, "retrieve_named_sources", lambda params, out: calls.append(("named", out)))
        monkeypatch.setattr(download, "prune_unused", lambda out, keep: calls.append(("prune", out)))

        download.download_omics_data("GDSC2", "response", tmp_dir)

        assert [name for name, _ in calls] == ["files", "named", "prune"]
