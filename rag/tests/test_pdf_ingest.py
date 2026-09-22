import fitz
import pytest

from rag.ingest import ingest, ingestion_fingerprint
from rag.models import ChunkSpec, Manifest, file_sha256
from rag.pdf_parser import ParseReviewRequired, inspect_pdf, load_blocks

from .support import CharacterCounter, FixtureEncoder, doc_spec, embedding_spec, policy


def make_pdf(path):
    doc = fitz.open()
    for number in range(1, 4):
        page = doc.new_page()
        page.insert_text((60, 25), "SYNTHETIC RUNNING HEADER", fontsize=8)
        page.insert_text((280, 820), str(number), fontsize=8)
        if number == 1:
            page.insert_text((60, 95), "1 Introduction", fontsize=16, fontname="hebo")
            page.insert_textbox(
                fitz.Rect(60, 135, 530, 300),
                "SYNTHETIC test document, not a scientific result. " * 10,
                fontsize=11,
            )
            page.insert_text((60, 335), "1.1 Methods", fontsize=13, fontname="hebo")
            page.insert_textbox(
                fitz.Rect(60, 380, 530, 700),
                "SYNTHETIC procedure description for parser testing. " * 12,
                fontsize=11,
            )
        elif number == 2:
            page.insert_text((60, 95), "References", fontsize=16, fontname="hebo")
            page.insert_text(
                (60, 140), "[1] SYNTHETIC excluded bibliography entry.", fontsize=11
            )
        else:
            page.insert_text((60, 95), "Appendix A", fontsize=16, fontname="hebo")
            page.insert_textbox(
                fitz.Rect(60, 135, 530, 400),
                "SYNTHETIC appendix content must remain after references. " * 8,
                fontsize=11,
            )
    doc.save(path)
    doc.close()


def create_manifest(tmp_path):
    pdf = tmp_path / "fixture_tq.pdf"
    make_pdf(pdf)
    spec = doc_spec(sha256=file_sha256(pdf))
    manifest = Manifest(
        schema_version=1,
        purpose="fixture",
        embedding=embedding_spec(),
        chunking=ChunkSpec(logical_tokens=600, embedding_tokens=400, overlap_tokens=60),
        documents=[spec],
    )
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2))
    return pdf, spec, manifest, path


def test_pdf_parse_section_refs_appendix_and_exclusion_audit(tmp_path):
    pdf, spec, _, _ = create_manifest(tmp_path)
    report = inspect_pdf(pdf, spec)
    text = "\n".join(b["text"] for b in report["blocks"])
    assert "appendix content" in text
    assert "excluded bibliography" not in text
    assert "RUNNING HEADER" not in text
    assert any("Methods" in " ".join(b["section"]) for b in report["blocks"])
    assert any(e["reason"] == "repeated_header_footer" for e in report["excluded"])
    assert any(e["reason"] == "page_number" for e in report["excluded"])
    assert report["page_count"] == 3


def test_checksum_and_page_count_errors(tmp_path):
    pdf, spec, _, _ = create_manifest(tmp_path)
    with pytest.raises(ValueError, match="SHA256"):
        inspect_pdf(pdf, doc_spec())
    with pytest.raises(ValueError, match="expected"):
        inspect_pdf(pdf, spec.model_copy(update={"pages": 5}))


def test_missing_text_layer_is_blocked(tmp_path):
    pdf = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    spec = doc_spec(pages=1, sha256=file_sha256(pdf), path="blank.pdf")
    with pytest.raises(ParseReviewRequired) as exc:
        load_blocks(pdf, spec, tmp_path)
    assert "NO_TEXT_LAYER" in [i["code"] for i in exc.value.report["issues"]]


def test_visual_caption_requires_review(tmp_path):
    pdf = tmp_path / "figure.pdf"
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text(
        (50, 100),
        "Figure 1: SYNTHETIC figure caption without reviewed conditions.",
        fontsize=11,
    )
    doc.save(pdf)
    doc.close()
    spec = doc_spec(pages=1, sha256=file_sha256(pdf), path="figure.pdf")
    with pytest.raises(ParseReviewRequired):
        load_blocks(pdf, spec, tmp_path)


def test_ingestion_pipeline_actual_pdf_and_sqlite_with_synthetic_encoder(tmp_path):
    _, _, _, path = create_manifest(tmp_path)
    index, cache = tmp_path / "index.sqlite", tmp_path / "cache.sqlite"
    encoder = FixtureEncoder()
    report = ingest(
        path,
        index,
        CharacterCounter(),
        encoder,
        policy(),
        thread_id="fixture-run",
        cache_path=cache,
        allow_fixture=True,
    )
    assert report["status"] == "created" and report["chunk_count"] > 1
    assert report["pdf_pages"] == 3 and report["max_embedding_tokens"] <= 400
    assert report["purpose"] == "fixture"
    calls = encoder.calls
    reused = ingest(
        path,
        index,
        CharacterCounter(),
        encoder,
        policy(),
        thread_id="fixture-run",
        cache_path=cache,
        allow_fixture=True,
    )
    assert reused["status"] == "reused" and encoder.calls == calls


def test_changed_original_invalidates_before_cached_index_reuse(tmp_path):
    pdf, _, _, path = create_manifest(tmp_path)
    index = tmp_path / "index.sqlite"
    encoder = FixtureEncoder()
    ingest(
        path,
        index,
        CharacterCounter(),
        encoder,
        policy(),
        thread_id="fixture-run",
        allow_fixture=True,
    )
    with pdf.open("ab") as stream:
        stream.write(b"\n% changed source\n")
    with pytest.raises(ValueError, match="checksum"):
        ingest(
            path,
            index,
            CharacterCounter(),
            encoder,
            policy(),
            thread_id="fixture-run",
            allow_fixture=True,
        )


def test_production_cannot_silently_use_fixture(tmp_path):
    _, _, _, path = create_manifest(tmp_path)
    with pytest.raises(ValueError, match="Fixture"):
        ingest(
            path,
            tmp_path / "x.sqlite",
            CharacterCounter(),
            FixtureEncoder(),
            policy(),
            thread_id="x",
        )


def test_changed_chunk_configuration_invalidates_fingerprint(tmp_path):
    _, _, manifest, _ = create_manifest(tmp_path)
    changed = manifest.model_copy(
        update={
            "chunking": ChunkSpec(
                logical_tokens=800, embedding_tokens=400, overlap_tokens=60
            )
        }
    )
    assert ingestion_fingerprint(
        manifest, FixtureEncoder().identity
    ) != ingestion_fingerprint(changed, FixtureEncoder().identity)
