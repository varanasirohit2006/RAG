import asyncio
import gzip
import json
import threading

import pytest
from unstructured.documents.elements import ElementMetadata, NarrativeText

import processing_jobs as jobs
from document_extraction import ExtractionSettings


def test_running_job_exposes_progress_and_tokens_without_blocking():
    entered, release = threading.Event(), threading.Event()
    def task(job):
        job.progress(1, 3, "Generating")
        job.token("Partial answer")
        entered.set()
        assert release.wait(5)
        return "Final answer"
    job = jobs.BackgroundJob(task)
    assert entered.wait(5)
    running = job.snapshot()
    assert running["status"] == "running"
    assert running["partial_text"] == "Partial answer"
    assert running["done"] == 1
    release.set()
    job.future.result(timeout=5)
    assert job.snapshot()["status"] == "succeeded"
    assert job.snapshot()["result"] == "Final answer"


def test_worker_failure_is_reported_as_failure():
    def task(job):
        raise RuntimeError("Provider timed out")
    job = jobs.BackgroundJob(task)
    job.future.result(timeout=5)
    assert job.snapshot()["status"] == "failed"
    assert job.snapshot()["error"] == "Provider timed out"
    assert job.snapshot()["result"] is None


def test_running_job_cancellation_is_not_a_success():
    entered = threading.Event()
    def task(job):
        entered.set()
        assert job.cancellation_event.wait(5)
        job.check_cancelled()
    job = jobs.BackgroundJob(task)
    assert entered.wait(5)
    job.cancel()
    job.future.result(timeout=5)
    assert job.snapshot()["status"] == "cancelled"
    assert job.snapshot()["result"] is None


def test_async_cancellation_finishes_background_job():
    def task(job):
        raise asyncio.CancelledError("Evaluation cancelled")

    job = jobs.BackgroundJob(task)
    job.future.result(timeout=5)
    state = job.snapshot()
    assert state["status"] == "cancelled"
    assert state["error"] == "Evaluation cancelled"
    assert state["result"] is None


def test_completed_extraction_is_reused_across_calls(tmp_path, monkeypatch):
    calls = []
    def extract(*args, **kwargs):
        calls.append(1)
        return [NarrativeText("Verified text 123.45", metadata=ElementMetadata(page_number=2))]
    monkeypatch.setattr(jobs, "extract_document", extract)
    options = dict(cache_dir=tmp_path, settings=ExtractionSettings(), document_hash="a" * 64)
    first, first_hit = jobs.extract_with_cache("source.pdf", **options)
    second, second_hit = jobs.extract_with_cache("source.pdf", **options)
    assert len(calls) == 1
    assert not first_hit and second_hit
    assert second[0].text == first[0].text
    assert second[0].metadata.page_number == 2
    _, different_hit = jobs.extract_with_cache("source.pdf", **{**options, "settings": ExtractionSettings(dpi=250)})
    assert not different_hit and len(calls) == 2


def test_invalid_cache_is_recomputed(tmp_path, monkeypatch):
    settings = ExtractionSettings()
    key = jobs.extraction_cache_key("a" * 64, settings)
    path = tmp_path / f"{key}.json.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("broken JSON")
    monkeypatch.setattr(jobs, "extract_document", lambda *args, **kwargs: [NarrativeText("Recovered")])
    elements, hit = jobs.extract_with_cache("source.pdf", "a" * 64, settings, cache_dir=tmp_path)
    assert not hit and elements[0].text == "Recovered"
    with gzip.open(path, "rt") as stream:
        assert json.load(stream)["cache_key"] == key


def test_cancelled_extraction_is_not_written_as_complete(tmp_path, monkeypatch):
    cancellation = threading.Event()
    def extract(*args, **kwargs):
        cancellation.set()
        return [NarrativeText("Partial")]
    monkeypatch.setattr(jobs, "extract_document", extract)
    with pytest.raises(jobs.JobCancelled):
        jobs.extract_with_cache("source.pdf", "a" * 64, ExtractionSettings(),
            cancellation_event=cancellation, cache_dir=tmp_path)
    assert not list(tmp_path.glob("*.json.gz"))
