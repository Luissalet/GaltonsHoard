"""A small GGUF that no allowed GPU can take right now runs on the CPU (runner.cpu_fallback, runner.cpu_max_gb, the run setting ``device``); its
measurements are flagged as CPU and their speed is never mixed with GPU speeds."""

from __future__ import annotations

import pytest

from galton_hoard import cpu as cpu_module, stats
from galton_hoard.errors import GaltonError
from galton_hoard.fakes import FakeGpus
from galton_hoard.hoard_link import GpuMemory
from galton_hoard.runner import normalise_settings
from helpers import add_gguf, build_services, run_inline

FULL = [(0, 12_282, 600), (1, 16_311, 700), (2, 16_311, 16_000), (3, 16_311, 16_100)]       # GPUs 2 and 3 are taken by another workload


@pytest.fixture
def busy(tmp_path, clock):
    svc = build_services(tmp_path, clock=clock, fake_gpus=FakeGpus(FULL))
    yield svc
    svc.db.close()


def free_the_gpus(svc):
    svc.fake_gpus.gpus = [GpuMemory(g.index, g.total_mb, 500) for g in svc.fake_gpus.gpus]


def take_the_gpus(svc):
    svc.fake_gpus.gpus = [GpuMemory(i, total, used) for i, total, used in FULL]


def small(svc, name="pequeno", size_gb=0.5, **kw):
    return add_gguf(svc, name, size_gb=size_gb, **kw)


def rc_of(svc, run, m):
    return svc.store.run_contestant(run["id"], m["id"])


# ------------------------------------------------------------------------------------------------ the decision
def test_a_small_model_runs_on_the_cpu_when_every_allowed_gpu_is_taken(busy):
    m = small(busy)
    run = run_inline(busy, ["rapida"], [m["id"]])
    rc = rc_of(busy, run, m)
    assert run["state"] == "done" and rc["state"] == "done" and rc["device"] == "cpu" and rc["gpus"] == [] and rc["vram_mb"] is None
    assert rc["runs_on"].key == "runs_on_cpu" and rc["runs_on"].params == {"threads": 10}
    assert any(getattr(w, "key", "") == "warn_cpu" for w in rc["warnings"])
    spec = busy.fake_world.started[0]
    assert spec.cpu is True and spec.threads == 10 and spec.grant is None
    assert busy.fake_gpus.leases == [], "no lease: the child never touches a GPU"
    assert all(r["cpu"] for r in busy.store.results(run_id=run["id"])) and busy.fake_world.stopped == ["pequeno"]


def test_it_does_not_wait_for_a_gpu_before_falling_back(busy):
    m = small(busy)
    slept = []
    busy.gpus._sleep = slept.append
    run_inline(busy, ["rapida"], [m["id"]])
    assert slept == [], "a model that may run on the CPU never queues for a GPU"


def test_a_free_gpu_is_still_preferred(svc):
    m = small(svc)
    run = run_inline(svc, ["rapida"], [m["id"]])
    rc = rc_of(svc, run, m)
    assert rc["device"] == "gpu" and rc["gpus"] and svc.fake_world.started[0].cpu is False
    assert not any(r["cpu"] for r in svc.store.results(run_id=run["id"]))


def test_a_big_file_keeps_todays_error_when_no_gpu_is_free(busy):
    m = small(busy, "grande", size_gb=5.0)          # over runner.cpu_max_gb (4)
    run = run_inline(busy, ["rapida"], [m["id"]], wait_s=0)
    rc = rc_of(busy, run, m)
    assert rc["state"] == "failed" and rc["error"].key == "no_gpu_free" and busy.fake_world.started == []


def test_the_limit_is_a_setting(busy):
    busy.settings.set_many({"runner.cpu_max_gb": 8})
    m = small(busy, "mediano", size_gb=5.0)
    assert rc_of(busy, run_inline(busy, ["rapida"], [m["id"]]), m)["device"] == "cpu"


def test_the_fallback_can_be_switched_off(busy):
    busy.settings.set_many({"runner.cpu_fallback": False})
    m = small(busy)
    rc = rc_of(busy, run_inline(busy, ["rapida"], [m["id"]], wait_s=0), m)
    assert rc["state"] == "failed" and rc["error"].key == "no_gpu_free" and busy.fake_world.started == []


def test_the_defaults_are_on_and_four_gb():
    from galton_hoard.settings import SPEC_BY_KEY
    assert SPEC_BY_KEY["runner.cpu_fallback"].default is True and SPEC_BY_KEY["runner.cpu_max_gb"].default == 4.0


def test_device_gpu_behaves_exactly_as_before(busy):
    m = small(busy)
    rc = rc_of(busy, run_inline(busy, ["rapida"], [m["id"]], device="gpu", wait_s=0), m)
    assert rc["state"] == "failed" and rc["error"].key == "no_gpu_free" and busy.fake_world.started == []


def test_device_cpu_forces_the_cpu_even_with_free_gpus(svc):
    m = small(svc)
    run = run_inline(svc, ["rapida"], [m["id"]], device="cpu")
    rc = rc_of(svc, run, m)
    assert rc["device"] == "cpu" and svc.fake_world.started[0].cpu is True and svc.fake_gpus.leases == []


def test_device_cpu_ignores_the_size_limit(svc):
    m = small(svc, "grande", size_gb=6.0)
    assert rc_of(svc, run_inline(svc, ["rapida"], [m["id"]], device="cpu"), m)["device"] == "cpu"


def test_the_device_setting_is_validated():
    assert normalise_settings({})["device"] == "auto"
    assert normalise_settings({"device": "CPU"})["device"] == "cpu"
    with pytest.raises(GaltonError) as exc:
        normalise_settings({"device": "tpu"})
    assert exc.value.key == "setting_choice"


def test_no_allowed_gpu_on_this_computer_also_falls_back(tmp_path, clock):
    svc = build_services(tmp_path, clock=clock, fake_gpus=FakeGpus([(0, 12_282, 600), (1, 16_311, 700)]))      # 2 and 3 do not exist
    m = small(svc)
    assert rc_of(svc, run_inline(svc, ["rapida"], [m["id"]]), m)["device"] == "cpu"
    svc.db.close()


def test_without_enough_free_memory_the_cpu_is_not_an_option(busy):
    busy.runner.ram_free_mb = lambda: 200
    m = small(busy)
    rc = rc_of(busy, run_inline(busy, ["rapida"], [m["id"]], wait_s=0), m)
    assert rc["state"] == "failed" and rc["error"].key == "no_gpu_free" and busy.fake_world.started == []


def test_forcing_the_cpu_without_enough_memory_says_so(busy):
    busy.runner.ram_free_mb = lambda: 200
    m = small(busy)
    rc = rc_of(busy, run_inline(busy, ["rapida"], [m["id"]], device="cpu"), m)
    assert rc["state"] == "failed" and rc["error"].key == "cpu_no_ram" and busy.fake_world.started == []


def test_the_watch_never_puts_a_background_run_on_the_cpu(busy):
    m = small(busy)
    ok, why = busy.watch._eligible(m)
    assert ok is False and "no allowed GPU" in why
    busy.watch.tick()
    assert all(run["settings"]["device"] == "gpu" for run in busy.store.runs(source="watch"))


def test_a_cpu_launch_uses_physical_cores_minus_two(busy):
    busy.runner.cpu_threads = lambda: 2
    m = small(busy)
    run_inline(busy, ["rapida"], [m["id"]])
    assert busy.fake_world.started[0].threads == 2
    assert cpu_module.cpu_threads(12) == 10 and cpu_module.cpu_threads(3) == 2 and cpu_module.cpu_threads(1) == 2


# ------------------------------------------------------------------------------------------------ the plan
def plan_of(svc, m, **settings):
    return next(c for c in svc.plan_run(["rapida"], [m["id"]], settings or None)["contestants"] if c["id"] == m["id"])


def test_the_plan_says_cpu_when_no_gpu_is_free(busy):
    p = plan_of(busy, small(busy))
    assert p["where"].key == "where_cpu" and p["device"] == "cpu" and p["vram_mb"] is None and p["ram_mb"] > 0
    assert busy.plan_run(["rapida"], [small(busy, "otro")["id"]])["problems"] == []


def test_the_plan_says_gpu_when_there_is_room_and_cpu_when_forced(svc):
    m = small(svc)
    assert plan_of(svc, m)["where"].key == "where_gpus" and plan_of(svc, m)["device"] == "gpu"
    forced = plan_of(svc, m, device="cpu")
    assert forced["where"].key == "where_cpu_forced" and forced["device"] == "cpu"


def test_the_plan_of_a_gpu_run_still_waits_or_fails_as_before(busy):
    p = plan_of(busy, small(busy), device="gpu")
    assert p["where"].key == "where_wait" and p["device"] == "gpu"


def test_a_big_file_in_the_plan_does_not_claim_the_cpu(busy):
    assert plan_of(busy, small(busy, "grande", size_gb=5.0))["where"].key == "where_wait"


# ------------------------------------------------------------------------------------------------ the numbers
def two_runs(svc, gpu_tps=80.0, cpu_tps=6.0, suite="rapida"):
    """The same model measured on the CPU (GPUs taken) and then on a GPU."""
    m = small(svc, "doble", tps=cpu_tps)
    run_inline(svc, [suite], [m["id"]])
    free_the_gpus(svc)
    svc.fake_world.tps["doble"] = gpu_tps
    run_inline(svc, [suite], [m["id"]])
    return m


def test_the_leaderboard_never_mixes_cpu_and_gpu_speeds(busy):
    m = two_runs(busy)
    row = next(r for r in busy.board.leaderboard(suite="rapida")["rows"] if r["contestant"] == m["id"])
    assert row["decode_tps"] == 80.0 and row["speed_cpu"] is False and row["decode_tps_cpu"] == 6.0
    assert row["n_results"] == 24, "the scores of both runs count"


def test_only_cpu_measurements_show_the_cpu_speed_with_the_flag(busy):
    m = small(busy, tps=6.0)
    run_inline(busy, ["rapida"], [m["id"]])
    row = busy.board.leaderboard(suite="rapida")["rows"][0]
    assert row["decode_tps"] == 6.0 and row["speed_cpu"] is True and row["decode_tps_cpu"] is None


def test_speed_by_device():
    rows = [{"decode_tps": 90.0, "cpu": 0}, {"decode_tps": 70.0, "cpu": 0}, {"decode_tps": 5.0, "cpu": 1}]
    both = stats.speed_by_device(rows)
    assert both["decode_tps_median"] == 80.0 and both["cpu"] is False and both["cpu_decode_tps_median"] == 5.0 and both["n"] == 2
    only = stats.speed_by_device(rows[2:])
    assert only["decode_tps_median"] == 5.0 and only["cpu"] is True
    empty = stats.speed_by_device([])
    assert empty["decode_tps_median"] is None and empty["cpu"] is False


def test_compare_keeps_devices_apart(busy):
    a = two_runs(busy)
    b = small(busy, "solo-cpu", tps=4.0)
    take_the_gpus(busy)
    run_inline(busy, ["rapida"], [b["id"]])
    out = busy.board.compare(a["id"], b["id"], suite="rapida")
    assert out["a"]["speed"]["decode_tps_median"] == 80.0 and out["a"]["speed"]["cpu"] is False
    assert out["b"]["speed"]["decode_tps_median"] == 4.0 and out["b"]["speed"]["cpu"] is True


def test_routes_use_the_gpu_speed_when_both_exist_and_flag_cpu_only_models(busy):
    both = two_runs(busy, suite="razonamiento")
    only = small(busy, "solo", tps=5.0)
    take_the_gpus(busy)
    run_inline(busy, ["razonamiento"], [only["id"]])
    ev = busy.routes.evaluate("general", {**busy.settings.get("routes.policy"), "min_cases": 1, "min_tok_s": 0}, busy.clock())
    by = {x["contestant"]["id"]: x for x in ev["ranked"]}
    assert by[both["id"]]["tok_s"] == 80.0 and by[both["id"]]["cpu"] is False
    assert by[only["id"]]["tok_s"] == 5.0 and by[only["id"]]["cpu"] is True
    items = {x["contestant"]["id"]: busy.routes._item(x) for x in ev["ranked"]}
    assert "cpu" not in items[both["id"]] and items[only["id"]]["cpu"] is True


def test_a_cpu_speed_never_counts_in_the_speed_term_of_the_ranking(busy):
    busy.settings.set_many({"routes.policy": {"weight_quality": 1, "weight_speed": 5, "min_tok_s": 0, "min_cases": 1}})
    fast_gpu = small(busy, "gpu-lento", tps=20.0, accuracy=1.0)
    free_the_gpus(busy)
    run_inline(busy, ["razonamiento"], [fast_gpu["id"]])
    cpu_only = small(busy, "cpu-rapido", tps=500.0, accuracy=1.0)
    take_the_gpus(busy)
    run_inline(busy, ["razonamiento"], [cpu_only["id"]])
    ev = busy.routes.evaluate("general", busy.settings.get("routes.policy"), busy.clock())
    assert [x["contestant"]["name"] for x in ev["ranked"]][0] == "gpu-lento"


def test_results_cards_and_run_cards_carry_the_flag(busy):
    m = small(busy)
    run = run_inline(busy, ["razonamiento"], [m["id"]])
    card = busy.run_card(run)
    assert card["contestants"][0]["device"] == "cpu"
    rows = busy.store.results(run_id=run["id"])
    names = {m["id"]: m["name"]}
    assert busy.result_card(rows[0], {}, names)["cpu"] is True


def test_model_detail_records_the_measurement_device(busy):
    m = small(busy)
    run_inline(busy, ["rapida"], [m["id"]])
    assert busy.store.measurements(m["id"])[0]["device"] == "cpu"
    assert busy.board.leaderboard(suite="rapida")["rows"][0]["memory_method"] != "measured"      # no VRAM was measured: nothing claims it was


def test_the_cpu_helpers_never_raise():
    assert cpu_module._cores_proc() >= 0
    assert cpu_module._ram_windows() is None or cpu_module._ram_windows() > 0
