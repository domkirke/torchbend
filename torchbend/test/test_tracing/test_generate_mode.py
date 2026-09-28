"""Generate mode: planning combinations of bending values, and running them.

The planning is plain Python (``graph_viewer.generation``); the job runs through
the viewer's endpoints on a toy text-to-audio interface, with a macro and a
callback parameter swept over two prompts, and must leave every value as it
found it.
"""
import json
import os
import random
import time

import pytest
import torch

from torchbend.ui.graph_viewer.generation import (
    MAX_RUNS, Axis, GenerationJob, combine, count_runs, render_name, sample_values)


# ── sampling one parameter ───────────────────────────────────────────────────

@pytest.mark.parametrize("spec, kind, expected", [
    ({"mode": "fixed", "value": 0.5}, "float", [0.5]),
    ({"mode": "values", "values": "0.1, 0.5 2"}, "float", [0.1, 0.5, 2.0]),
    ({"mode": "values", "values": [1, 0]}, "bool", [True, False]),
    ({"mode": "range", "min": 0, "max": 1, "n": 5}, "float", [0.0, 0.25, 0.5, 0.75, 1.0]),
    ({"mode": "range", "min": 1, "max": 100, "n": 3, "scale": "log"}, "float", [1.0, 10.0, 100.0]),
    ({"mode": "range", "min": 0, "max": 3, "n": 10}, "int", [0, 1, 2, 3]),   # rounded, deduplicated
    ({"mode": "range", "min": 0, "max": 1, "n": 7}, "bool", [False, True]),
    ({"mode": "range", "min": 0, "max": 1, "n": 1}, "float", [0.0]),
])
def test_sampling(spec, kind, expected):
    assert sample_values(spec, kind=kind) == expected


def test_random_values_are_reproducible_and_in_range():
    spec = {"mode": "random", "min": -2, "max": 3, "n": 20}
    a = sample_values(spec, rng=random.Random(4))
    assert a == sample_values(spec, rng=random.Random(4))
    assert len(a) == 20 and all(-2 <= v <= 3 for v in a)
    ints = sample_values(spec, kind="int", rng=random.Random(4))
    assert all(isinstance(v, int) and -2 <= v <= 3 for v in ints)


def test_choices_restrict_the_values():
    assert sample_values({"mode": "range", "min": 0, "max": 5, "n": 3}, choices=[1, 2, 8]) == [1, 2]
    assert sample_values({"mode": "fixed", "value": 7}, choices=[1, 2, 8]) == [8]   # nearest


@pytest.mark.parametrize("spec", [
    {"mode": "sideways"}, {"mode": "values", "values": ""}, {"mode": "range", "min": 0, "max": 1, "n": 0},
    {"mode": "range", "min": 0, "max": 1, "n": 3, "scale": "log"},
])
def test_bad_specs_are_refused(spec):
    with pytest.raises(ValueError):
        sample_values(spec)


# ── combining axes ───────────────────────────────────────────────────────────

def _axes():
    return [Axis("macro:a", [1, 2, 3]),
            Axis("binding:b.x", [10, 20], group="g"),
            Axis("binding:c.y", [100, 200], group="g"),
            Axis("seed", [0, 1])]


def test_product_combines_groups_and_zips_within_one():
    runs = combine(_axes(), "product")
    assert len(runs) == count_runs(_axes()) == 3 * 2 * 2
    # b.x and c.y share a group: they move together
    assert all((r["binding:b.x"], r["binding:c.y"]) in ((10, 100), (20, 200)) for r in runs)
    assert len({tuple(sorted(r.items())) for r in runs}) == len(runs)


def test_zip_advances_every_group_together():
    runs = combine(_axes(), "zip")
    assert [r["macro:a"] for r in runs] == [1, 2, 3]
    assert [r["seed"] for r in runs] == [0, 1, 0]          # shorter groups cycle


def test_random_draws_distinct_combinations():
    runs = combine(_axes(), "random", n=5, seed=2)
    assert len(runs) == 5 and runs == combine(_axes(), "random", n=5, seed=2)
    full = [tuple(sorted(r.items())) for r in combine(_axes(), "product")]
    picked = [tuple(sorted(r.items())) for r in runs]
    assert len(set(picked)) == 5 and set(picked) <= set(full)
    assert picked == sorted(picked, key=full.index)       # product order kept
    assert len(combine(_axes(), "random", n=1000)) == 12   # capped at the product


def test_no_axes_is_one_run_and_too_many_is_refused():
    assert combine([], "product") == [{}]
    with pytest.raises(ValueError, match="limit"):
        combine([Axis("a", range(200)), Axis("b", range(200))], "product")
    assert MAX_RUNS < 200 * 200


# ── naming ───────────────────────────────────────────────────────────────────

def test_names():
    fields = {"input": "a caption/with a slash", "gain": 1 / 3, "index": 7, "on": True}
    assert render_name("{input}_{index:04d}", fields) == "a-caption-with-a-slash_0007"
    # floats short by default, formatted on request; booleans as 0/1
    assert render_name("{gain}_{gain:.2f}_{on}", fields) == "0.3333_0.33_1"
    # a slash in the template makes folders; `..` cannot climb out
    assert render_name("../{input}/x_{index}", fields) == "a-caption-with-a-slash/x_7"
    # an unknown field names itself rather than failing the job
    assert render_name("{typo}", fields) == "{typo}"


# ── a job, through the viewer ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def viewer():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "torchbend.ui.graph_viewer.settings")
    import django
    from torchbend.test.test_modules.interfaces import BendedTinyAudioGen
    from torchbend.ui.graph_viewer import _to_registry, set_registry
    iface = BendedTinyAudioGen()
    set_registry(_to_registry({"audio": iface}))
    django.setup()
    from django.test import Client
    client = Client()

    def post(url, body):
        return client.post(url, json.dumps(body), content_type="application/json").json()

    scale = post("/api/bending/", {"fn": "forward", "node": "sum_3", "callback_type": "Scale",
                                   "params": {}})["bindings"][-1]["id"]
    shift = post("/api/bending/", {"fn": "forward", "node": "sum_3", "callback_type": "Scale",
                                   "params": {"scale": 0.8}})["bindings"][-1]["id"]
    post("/api/bending_params/", {"name": "gain", "value": 1.0, "range_min": 0.0, "range_max": 2.0})
    post("/api/bending/%s/link/" % scale, {"param_name": "scale", "bp_name": "gain"})
    client.post("/api/play/compile/", json.dumps({"fn": "forward", "device": "cpu"}),
                content_type="application/json")
    return client, iface, shift


#: The toy model's one output, as an export target.
OUTPUT = [{"node": "tanh"}]


def _form(config):
    return {"input_ids": ["a low rain on water", "bright bells"], "__mode__input_ids": "text",
            "__labels__input_ids": json.dumps(["rain", "bells"]), "config": json.dumps(config)}


def _values(client, binding):
    macros = client.get("/api/bending_params/").json()
    gain = next(p["value"] for p in macros.get("bending_params", macros.get("params", []))
                if p["name"] == "gain")
    scale = next(b["params"]["scale"] for b in client.get("/api/bending/").json()["bindings"]
                 if b["id"] == binding)
    return gain, scale


def test_sweepable_lists_macros_and_undriven_parameters(viewer):
    client, _iface, shift = viewer
    params = client.get("/api/generate/params/?fn=forward").json()["params"]
    keys = {p["key"]: p for p in params}
    assert "macro:gain" in keys
    assert "binding:%s.scale" % shift in keys
    # the first Scale's `scale` is driven by `gain`: varied through it, not listed
    assert sum(1 for k in keys if k.endswith(".scale")) == 1
    assert len({p["field"] for p in params}) == len(params)


def test_a_job_writes_every_run_and_puts_everything_back(viewer, tmp_path):
    client, _iface, shift = viewer
    before = _values(client, shift)
    config = {"fn": "forward",
              "params": [{"key": "macro:gain", "mode": "range", "min": 0.2, "max": 0.8, "n": 3},
                         {"key": "binding:%s.scale" % shift, "mode": "values", "values": "0.5, 1"}],
              "seed": {"base": 3, "repeats": 1}, "combine": {"mode": "product"},
              "export": {"dir": str(tmp_path), "name": "{input}/{gain:.1f}_{index:02d}",
                         "targets": OUTPUT}}

    plan = client.post("/api/generate/plan/", _form(config)).json()
    assert plan["runs"] == 3 * 2 * 2
    assert plan["names"][:2] == ["rain/0.2_00", "bells/0.2_01"]
    assert plan["inputs"] == {"input_ids": ["rain", "bells"]}

    job = client.post("/api/generate/start/", _form(config)).json()
    assert job["state"] == "running"
    for _ in range(600):
        status = client.get("/api/generate/status/?id=" + job["id"]).json()
        if status["state"] != "running":
            break
        time.sleep(0.1)
    assert status["state"] == "done", status
    assert status["done"] == status["total"] == status["files"] == 12

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["state"] == "done" and len(manifest["files"]) == 12
    first = manifest["files"][0]
    assert first["path"] == "rain/0.2_00.wav" and first["source"] == "all"   # prompts are not audio
    assert first["sample_rate"] == 16000 and first["seed"] == 3
    assert "gain" in first["params"] and len(first["params"]) == 2
    import soundfile as sf
    audio, rate = sf.read(tmp_path / first["path"])
    assert rate == 16000 and len(audio) == 8192

    # the same seed and values give the same sound, however the run was reached
    by_params = {}
    for f in manifest["files"]:
        by_params.setdefault((f["inputs"]["input_ids"], tuple(sorted(f["params"].items()))), []).append(f["path"])
    assert all(len(v) == 1 for v in by_params.values())

    assert _values(client, shift) == before


def test_a_second_job_waits_for_the_first(viewer, tmp_path):
    client, _iface, _shift = viewer
    config = {"fn": "forward", "params": [{"key": "macro:gain", "mode": "range", "min": 0,
                                           "max": 1, "n": 40}],
              "export": {"dir": str(tmp_path), "targets": OUTPUT}}
    first = client.post("/api/generate/start/", _form(config)).json()
    second = client.post("/api/generate/start/", _form(config))
    assert second.status_code == 409
    client.post("/api/generate/cancel/", json.dumps({"id": first["id"]}),
                content_type="application/json")
    for _ in range(600):
        status = client.get("/api/generate/status/?id=" + first["id"]).json()
        if status["state"] != "running":
            break
        time.sleep(0.1)
    assert status["state"] in ("cancelled", "done")


def test_the_explorer_reads_a_manifest(viewer, tmp_path):
    client, _iface, shift = viewer
    config = {"fn": "forward",
              "params": [{"key": "macro:gain", "mode": "values", "values": "0.3, 0.9"}],
              "export": {"dir": str(tmp_path), "type": "g{gain}", "targets": OUTPUT}}
    job = client.post("/api/generate/start/", _form(config)).json()
    for _ in range(600):
        if client.get("/api/generate/status/?id=" + job["id"]).json()["state"] != "running":
            break
        time.sleep(0.1)
    from torchbend.ui.audio_explorer import load_manifest
    originals, bended, stem_to_path, names = load_manifest([tmp_path / "manifest.json"])
    # nothing asked to group them (and no input is audio): one space, no original
    assert originals == [] and set(stem_to_path) == {"all"}
    assert len(bended) == 4
    assert {b["bending_type"] for b in bended} == {"g0.3", "g0.9"}
    assert {b["params"]["gain"] for b in bended} == {0.3, 0.9}
    assert names["gain"] == "gain"


# ── pools for the audio explorer ─────────────────────────────────────────────

def _pool_job(tmp_path, export, inputs):
    hooks = {"set": lambda *a: None, "reseed": lambda seed: None,
             "run": lambda kw: [("output", kw["x"] * 2, "out", True)],
             "rate": lambda t, kw, node: 8000, "wav": lambda t, sr: (b"RIFF", sr),
             "restore": lambda: None}
    runs = combine([Axis("input:x", list(range(len(inputs["x"]))))])
    job = GenerationJob(runs, {"input:x": ("input", "x")}, inputs, {},
                        dict(export, dir=str(tmp_path)), hooks, meta={"fields": {}})
    job.start()
    job.join()
    return json.loads((tmp_path / "manifest.json").read_text())


def _audio_inputs():
    return {"x": [("kick", {"x": torch.zeros(1, 64)}, (torch.zeros(1, 64), 8000)),
                  ("snare", {"x": torch.ones(1, 64)}, (torch.ones(1, 64), 8000))],
            "cond": [("c", {"cond": torch.zeros(1)}, None)]}


def test_by_default_every_generation_is_in_one_pool(tmp_path):
    m = _pool_job(tmp_path, {}, _audio_inputs())
    assert {f["source"] for f in m["files"]} == {"all"} and m["sources"] == {}
    assert not (tmp_path / "sources").exists()


def test_input_as_original_pools_by_audio_input(tmp_path):
    m = _pool_job(tmp_path, {"input_as_original": True}, _audio_inputs())
    assert [f["source"] for f in m["files"]] == ["kick", "snare"]
    assert m["sources"] == {"kick": "sources/kick.wav", "snare": "sources/snare.wav"}
    assert (tmp_path / "sources" / "kick.wav").read_bytes() == b"RIFF"


def test_input_as_original_without_an_audio_input_is_one_pool(tmp_path):
    inputs = {"x": [("a", {"x": torch.zeros(1, 4)}, None), ("b", {"x": torch.ones(1, 4)}, None)]}
    m = _pool_job(tmp_path, {"input_as_original": True}, inputs)
    assert {f["source"] for f in m["files"]} == {"all"} and m["sources"] == {}


# ── a missing input is named, not crashed on ────────────────────────────────

def test_a_missing_input_is_named_before_anything_runs(viewer, tmp_path):
    client, _iface, _shift = viewer
    config = {"fn": "forward", "params": [{"key": "macro:gain", "mode": "values", "values": "0.5"}],
              "export": {"dir": str(tmp_path), "targets": OUTPUT}}
    form = {"config": json.dumps(config)}                  # no prompt on the bench
    plan = client.post("/api/generate/plan/", form).json()
    assert "input_ids" in plan["missing"] and "unchecked" in plan["missing"]
    start = client.post("/api/generate/start/", form)
    assert start.status_code == 400 and "no input for input_ids" in start.json()["error"]
    assert not (tmp_path / "manifest.json").exists()
    # play mode says the same, rather than failing on None inside the model
    run = client.post("/api/play/run/", {})
    assert "no input for input_ids" in run.json()["error"]


# ── choosing outputs ─────────────────────────────────────────────────────────

def test_each_output_can_be_switched_and_formatted(tmp_path):
    hooks = {"set": lambda *a: None, "reseed": lambda seed: None,
             "run": lambda kw: [("out[0]", torch.zeros(1, 1, 4096), "a", True),
                                ("out[1]", torch.zeros(1, 16, 1024), "b", False),
                                ("out[2]", torch.zeros(1, 16), "c", False)],
             "rate": lambda t, kw, node: 8000, "wav": lambda t, sr: (b"RIFF", sr),
             "restore": lambda: None}
    export = {"dir": str(tmp_path), "name": "x", "outputs": [0, 1],
              "formats": {"1": "pt"}, "output_labels": {"0": "speech", "1": "latents"}}
    job = GenerationJob([{}], {}, {}, {}, export, hooks, meta={"fields": {}})
    job.start()
    job.join()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json", "x_latents.pt",
                                                          "x_speech.wav"]
    files = json.loads((tmp_path / "manifest.json").read_text())["files"]
    assert [(f["output"], f["format"]) for f in files] == [("speech", "wav"), ("latents", "pt")]


def test_a_sequence_is_not_taken_for_an_image():
    from torchbend.ui.graph_viewer.generation import guess_format
    assert guess_format(torch.zeros(1, 16, 1024)) == "pt"        # [1, frames, features]
    assert guess_format(torch.zeros(1, 3, 64, 64)) == "png"
    assert guess_format(torch.zeros(3, 64, 64)) == "png"
    assert guess_format(torch.zeros(1, 1, 8192)) == "wav"
    assert guess_format(torch.zeros(1, 16), declared_audio=True) == "wav"



# ── export targets ───────────────────────────────────────────────────────────

def _wait(client, job):
    for _ in range(600):
        status = client.get("/api/generate/status/?id=" + job["id"]).json()
        if status["state"] != "running":
            return status
        time.sleep(0.1)
    return status


def test_outputs_are_listed_as_targets(viewer):
    client, _iface, _shift = viewer
    outputs = client.get("/api/generate/params/?fn=forward").json()["outputs"]
    assert [o["node"] for o in outputs] == ["tanh"]
    # not declared audio by the toy interface, but shaped like it
    assert outputs[0]["format"] == "wav"


def test_an_activation_can_be_exported_beside_an_output(viewer, tmp_path):
    client, iface, shift = viewer
    info = client.get("/api/generate/target/?fn=forward&node=sum_3").json()
    assert info["node"] == "sum_3" and info["field"] == "sum_3"
    config = {"fn": "forward", "params": [{"key": "macro:gain", "mode": "values", "values": "0.5"}],
              "export": {"dir": str(tmp_path), "name": "x",
                         "targets": OUTPUT + [{"node": "sum_3", "format": "pt"}]}}
    status = _wait(client, client.post("/api/generate/start/", _form(config)).json())
    assert status["state"] == "done", status
    files = json.loads((tmp_path / "manifest.json").read_text())["files"]
    assert {(f["output"], f["format"]) for f in files} == {("tanh", "wav"), ("sum_3", "pt")}
    # the activation is the bended one: `sum_3` carries two Scale bendings
    saved = torch.load(str(tmp_path / next(f["path"] for f in files if f["output"] == "sum_3")))
    assert saved.shape[-1] == 8192


def test_nothing_to_export_is_refused(viewer, tmp_path):
    client, _iface, _shift = viewer
    config = {"fn": "forward", "export": {"dir": str(tmp_path), "targets": []}}
    r = client.post("/api/generate/start/", _form(config))
    assert r.status_code == 400 and "nothing selected" in r.json()["error"]
    config["export"]["targets"] = [{"node": "no_such_node"}]
    r = client.post("/api/generate/start/", _form(config))
    assert r.status_code == 400 and "no_such_node" in r.json()["error"]
