"""GPU rules: only allowed GPUs, one GPU when it fits, a split when it does not, honest refusals and always giving the lease back."""

from __future__ import annotations

import pytest

from galton_hoard import gguf_meta, placement
from galton_hoard.errors import GaltonError
from galton_hoard.fakes import FakeGpus
from galton_hoard.gpus import GpuManager


class Settings:
    def __init__(self, **values):
        self.values = {"gpus.allowed": [2, 3], "gpus.reserved": [0, 1], **values}

    def get(self, key):
        return self.values[key]


def manager(fake=None, **settings):
    fake = fake or FakeGpus()
    return GpuManager(Settings(**settings), probe=fake.probe, hub=fake.hub, lease_factory=fake.lease_factory, names=fake.names, sleep=lambda _s: None), fake


def test_state_marks_allowed_and_reserved():
    mgr, _ = manager()
    state = mgr.state()
    rows = {g["index"]: g for g in state["gpus"]}
    assert rows[0]["reserved"] and not rows[0]["allowed"] and rows[2]["allowed"] and not rows[2]["reserved"]
    assert state["allowed"] == [2, 3] and state["reserved"] == [0, 1] and state["hub"] is False and rows[3]["name"].endswith("5060 Ti")


def test_plan_picks_the_emptiest_allowed_gpu_never_a_reserved_one():
    mgr, _ = manager(FakeGpus([(0, 24_000, 0), (1, 24_000, 0), (2, 16_000, 6_000), (3, 16_000, 1_000)]))
    assert mgr.plan(8_000) == {3: 8_000}


def test_plan_splits_when_no_single_gpu_fits():
    mgr, _ = manager()
    plan = mgr.plan(22_000)
    assert set(plan) == {2, 3} and sum(plan.values()) == 22_000


def test_plan_is_none_when_there_is_no_room_right_now():
    mgr, _ = manager(FakeGpus([(2, 16_000, 15_000), (3, 16_000, 15_500)]))
    assert mgr.plan(8_000) is None


def test_plan_ignores_gpus_that_are_not_allowed_even_if_empty():
    mgr, _ = manager(FakeGpus([(0, 24_000, 0), (1, 24_000, 0), (2, 8_000, 0)]))
    assert mgr.plan(12_000) is None


def test_check_possible_explains_each_impossibility():
    mgr, _ = manager()
    mgr.check_possible(20_000)
    with pytest.raises(GaltonError, match="hold"):
        mgr.check_possible(60_000)
    none_present, _ = manager(FakeGpus([(0, 12_000, 0)]))
    with pytest.raises(GaltonError, match="None of the allowed GPUs"):
        none_present.check_possible(1_000)
    empty, fake_empty = manager()
    fake_empty.gpus = []
    with pytest.raises(GaltonError, match="No NVIDIA GPU"):
        empty.check_possible(1_000)


def test_acquire_takes_one_lease_per_gpu_and_releases_them():
    mgr, fake = manager()
    grant = mgr.acquire(22_000, "test")
    assert sorted(grant.gpus) == [2, 3] and len(fake.leases) == 2 and grant.tensor_split and abs(sum(grant.tensor_split) - 1) < 1e-3
    assert grant.cuda_visible_devices in ("2,3", "3,2")
    grant.release()
    grant.release()
    assert len(fake.released) == 2


def test_acquire_single_gpu_has_no_tensor_split():
    mgr, _ = manager()
    grant = mgr.acquire(5_000, "test")
    assert len(grant.gpus) == 1 and grant.tensor_split is None and grant.via == "local"


def test_a_lease_that_lands_on_a_reserved_gpu_is_refused_and_given_back():
    mgr, fake = manager()
    fake.override_gpu = 0
    with pytest.raises(GaltonError) as e:
        mgr.acquire(5_000, "test")
    assert e.value.code == "gpu_not_allowed" and len(fake.released) == 1


def test_acquire_without_waiting_fails_fast_when_full():
    mgr, fake = manager()
    fake.refuse = {2, 3}
    with pytest.raises(GaltonError) as e:
        mgr.acquire(5_000, "test", wait_s=0)
    assert e.value.code == "no_gpu"
    assert mgr.try_acquire(5_000, "test") is None


def test_acquire_waits_then_succeeds_when_room_appears():
    mgr, fake = manager()
    fake.refuse = {2, 3}
    calls = []

    def sleep(_):
        calls.append(1)
        fake.refuse = set()

    mgr._sleep = sleep
    heard = []
    grant = mgr.acquire(5_000, "test", wait_s=100, on_wait=heard.append)
    assert grant.gpus and calls and heard and "waiting" in heard[0]


def test_acquire_cancel_while_waiting():
    mgr, fake = manager()
    fake.refuse = {2, 3}
    with pytest.raises(GaltonError, match="Cancelled"):
        mgr.acquire(5_000, "test", wait_s=100, cancel=lambda: True)


def test_split_lease_failure_gives_back_the_first_lease():
    mgr, fake = manager()
    fake.refuse = {2}  # GPU 3 has more room, so it is leased first and must be given back when GPU 2 refuses
    assert mgr.try_acquire(22_000, "test") is None
    assert [lease.gpu for lease in fake.released] == [3]


def test_allowing_a_gpu_changes_the_plan():
    mgr, _ = manager(FakeGpus([(0, 24_000, 0), (2, 8_000, 0)]))
    assert mgr.plan(12_000) is None
    mgr.settings.values["gpus.allowed"] = [0, 2]
    assert mgr.plan(12_000) == {0: 12_000}


# ------------------------------------------------------------------ placement
def test_context_is_capped_by_what_the_model_was_trained_for():
    assert placement.choose_context(32_768, {"context_length": 8_192}) == 8_192
    assert placement.choose_context(4_096, {"context_length": 8_192}) == 4_096
    assert placement.choose_context(4_096, {}) == 4_096


def test_describe_a_big_gguf_across_two_gpus(svc):
    from helpers import add_gguf
    big = add_gguf(svc, "big", size_gb=22.0)
    info = placement.describe(svc.store, big, 8_192, svc.gpus)
    assert info["runs_on"] == "llama.cpp" and info["fits_one_16gb"] is False and len(info["gpus"]) == 2 and info["vram_mb"] > 22_000


def test_describe_a_small_gguf_on_one_gpu(svc):
    from helpers import add_gguf
    small = add_gguf(svc, "small", size_gb=3.0)
    info = placement.describe(svc.store, small, 8_192, svc.gpus)
    assert info["fits_one_16gb"] is True and len(info["gpus"]) == 1 and info["where"].startswith("GPU ")


def test_describe_a_model_that_cannot_fit_anywhere(svc):
    from helpers import add_gguf
    huge = add_gguf(svc, "huge", size_gb=60.0)
    info = placement.describe(svc.store, huge, 8_192, svc.gpus)
    assert info["warnings"] and "hold" in info["where"]


def test_describe_a_server_does_not_need_memory(svc):
    c = svc.store.create_contestant(key="server:x", kind="server", name="srv", url="http://127.0.0.1:8081", api="openai", model="m", provider="llamacpp", aliases=["srv"], meta={"resident": True})
    info = placement.describe(svc.store, c, 8_192, svc.gpus)
    assert info["runs_on"] == "server" and info["vram_mb"] is None and "resident" in info["where"]


def test_estimate_needs_a_size(svc):
    c = svc.store.create_contestant(key="gguf:/x.gguf", kind="gguf", name="nosize", path="/x.gguf", provider="llamacpp", aliases=["nosize"])
    with pytest.raises(GaltonError, match="size"):
        placement.estimate_mb(svc.store, c, 4_096)


def test_file_factor_and_headroom_are_the_documented_ones():
    assert gguf_meta.FILE_FACTOR == 1.08 and gguf_meta.HEADROOM_MB == 600


def test_the_child_is_shown_the_gpu_the_lease_holds_not_the_one_asked_for():
    mgr, fake = manager()
    fake.override_gpu = 2          # the plan asks for GPU 3 (more free memory); the hub grants an allowed GPU 2 instead
    grant = mgr.acquire(5_000, "test")
    assert grant.gpus == [2] and grant.cuda_visible_devices == "2" and grant.shares_mb == {2: 5_000}
