"""Activation snapshots: named copies of an activation, kept by the viewer.

Saved from a view in the graph editor, recalled to freeze a view on them, and
kept whole on the server -- the material activation interpolation works from.
"""
import json
import os

import pytest
import torch


@pytest.fixture(scope="module")
def viewer(tmp_path_factory):
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "torchbend.ui.graph_viewer.settings")
    import django
    from torchbend.test.test_modules.interfaces import BendedTinyAudioGen
    from torchbend.ui import graph_viewer as gv
    gv.set_registry(gv._to_registry({"audio": BendedTinyAudioGen()}))
    django.setup()
    from django.test import Client
    return Client(), gv


def _form(**extra):
    return dict({"input_ids": "a low rain", "__mode__input_ids": "text",
                 "fn": "forward", "node": "sum_3"}, **extra)


def test_save_recall_delete(viewer):
    client, gv = viewer
    listed = client.get("/api/snapshots/?fn=forward&node=sum_3").json()
    assert listed["snapshots"] == [] and listed["default_name"] == "sum_3_1"

    saved = client.post("/api/snapshots/", _form()).json()["snapshot"]
    assert saved["name"] == "sum_3_1" and saved["shape"] == [1, 1, 8192]
    assert saved["sample_rate"] == 16000
    assert client.get("/api/snapshots/?node=sum_3").json()["default_name"] == "sum_3_2"

    # a name is not reused unless asked to
    again = client.post("/api/snapshots/", _form(name="sum_3_1"))
    assert again.status_code == 409 and again.json()["exists"]
    assert client.post("/api/snapshots/", _form(name="sum_3_1", overwrite="true")).json()["ok"]

    recalled = client.get("/api/snapshots/sum_3_1/").json()
    assert recalled["shape"] == [1, 1, 8192] and recalled["snapshot"]["node"] == "sum_3"

    # kept whole, for Python to use
    tensors = gv.get_snapshots()
    assert set(tensors) == {"sum_3_1"} and tensors["sum_3_1"].shape == (1, 1, 8192)

    assert client.delete("/api/snapshots/sum_3_1/").json()["snapshots"] == []
    assert client.get("/api/snapshots/sum_3_1/").status_code == 404


def test_a_snapshot_does_not_follow_the_graph(viewer):
    """What is saved is what the view showed: the bended value, frozen."""
    client, gv = viewer
    before = client.post("/api/snapshots/", _form(name="before")).json()["snapshot"]
    bid = client.post("/api/bending/", json.dumps({"fn": "forward", "node": "sum_3",
                                                   "callback_type": "Scale",
                                                   "params": {"scale": 3.0}}),
                      content_type="application/json").json()["bindings"][-1]["id"]
    try:
        client.post("/api/snapshots/", _form(name="after"))
        snaps = gv.get_snapshots()
        assert torch.allclose(snaps["after"], snaps["before"] * 3, atol=1e-5)
    finally:
        client.delete("/api/bending/%s/" % bid)
        for name in ("before", "after"):
            client.delete("/api/snapshots/%s/" % name)
    assert before["node"] == "sum_3"


def test_snapshots_need_a_node_and_inputs(viewer):
    client, _gv = viewer
    assert client.post("/api/snapshots/", {"fn": "forward"}).status_code == 400
    r = client.post("/api/snapshots/", {"fn": "forward", "node": "sum_3"})
    assert r.status_code >= 400


def test_snapshots_survive_a_restart_through_sync(tmp_path):
    from torchbend.ui.graph_viewer.bending_session import BendingSession
    from torchbend.ui.graph_viewer.sync import SyncManager
    sm = SyncManager(tmp_path)
    session = BendingSession()
    session.save_snapshot("x", "forward", "relu", torch.arange(6.).reshape(2, 3), sample_rate=None)
    sm.save_snapshots("model", session.snapshots)
    restored = sm.load_snapshots("model")
    assert torch.equal(restored["x"]["tensor"], torch.arange(6.).reshape(2, 3))
    assert restored["x"]["node"] == "relu"
    # deleting the last one removes the file rather than saving an empty one
    sm.save_snapshots("model", {})
    assert sm.load_snapshots("model") == {}


# ── recalling a snapshot into the graph ──────────────────────────────────────
# Recall puts the saved value back in the graph, as a Snapshot bending on the
# node: everything computed after it follows. The toy model's output is
# tanh(sum_3), so with sum_3 recalled the output is the recalled input's.

def _out(client, text):
    d = client.post("/api/activate/forward/", {"input_ids": text, "__mode__input_ids": "text",
                                               "nodes": json.dumps(["tanh"])}).json()
    return torch.tensor(d["tanh"]["batches"])


def _post(client, url, body):
    return client.post(url, json.dumps(body), content_type="application/json").json()


def test_recall_changes_what_follows(viewer):
    client, gv = viewer
    client.post("/api/snapshots/", _form(name="rain"))            # sum_3 on "a low rain"
    rain, bells = _out(client, "a low rain"), _out(client, "bright bells")
    assert not torch.allclose(rain, bells)
    try:
        r = _post(client, "/api/snapshots/rain/recall/", {"fn": "forward", "node": "sum_3"})
        (b,) = r["bindings"]
        assert (b["callback_type"], b["node"], b["snapshot"], b["params"]) == \
            ("Snapshot", "sum_3", "rain", {"mix": 1.0})
        assert torch.allclose(_out(client, "bright bells"), rain, atol=1e-5)
        # mix crossfades back to the live value
        client.patch("/api/bending/%s/" % b["id"], json.dumps({"mix": 0.0}),
                     content_type="application/json")
        assert torch.allclose(_out(client, "bright bells"), bells, atol=1e-5)
        client.patch("/api/bending/%s/" % b["id"], json.dumps({"mix": 1.0}),
                     content_type="application/json")
        # one recall per node: recalling again replaces it
        client.post("/api/snapshots/", _form(name="rain2"))
        r = _post(client, "/api/snapshots/rain2/recall/", {"fn": "forward", "node": "sum_3"})
        assert [x["snapshot"] for x in r["bindings"]] == ["rain2"]
        # "live" takes it off
        r = _post(client, "/api/snapshots/release/", {"fn": "forward", "node": "sum_3"})
        assert r["released"] and r["bindings"] == []
        assert torch.allclose(_out(client, "bright bells"), bells, atol=1e-5)
        # deleting a recalled snapshot takes it off too
        _post(client, "/api/snapshots/rain/recall/", {"fn": "forward", "node": "sum_3"})
        d = client.delete("/api/snapshots/rain/").json()
        assert len(d["released"]) == 1 and d["bindings"] == []
        assert torch.allclose(_out(client, "bright bells"), bells, atol=1e-5)
    finally:
        _post(client, "/api/snapshots/release/", {"fn": "forward", "node": "sum_3"})
        for name in ("rain", "rain2"):
            client.delete("/api/snapshots/%s/" % name)


def test_a_recall_is_saved_and_restored_with_the_session(viewer):
    client, gv = viewer
    client.post("/api/snapshots/", _form(name="kept"))
    _post(client, "/api/snapshots/kept/recall/", {"fn": "forward", "node": "sum_3"})
    session = gv.get_registry()._entries["audio"].session
    saved = session.session_to_json()
    assert saved["bindings"][0]["snapshot"] == "kept"
    module = gv.get_registry()._entries["audio"].module
    try:
        session.session_from_json(module, saved)       # snapshots are already loaded
        assert [(b["snapshot"], b["node"]) for b in session.list_bindings()] == [("kept", "sum_3")]
        # a recall whose snapshot is gone is dropped, not an error
        gone = dict(saved, bindings=[dict(saved["bindings"][0], snapshot="nope")])
        session.session_from_json(module, gone)
        assert session.list_bindings() == []
    finally:
        session.release_snapshot_everywhere(module, "kept")
        client.delete("/api/snapshots/kept/")


def test_the_snapshot_bending_is_not_offered_on_its_own():
    from torchbend.ui.graph_viewer.bending_session import get_available_callbacks
    assert "Snapshot" not in [c["name"] for c in get_available_callbacks()]


@pytest.mark.parametrize("saved, like, expected", [
    ((1, 2, 5), (3, 2, 5), (3, 2, 5)),      # batch of one broadcasts
    ((1, 2, 7), (1, 2, 5), (1, 2, 5)),      # longer: cropped
    ((1, 2, 3), (1, 2, 5), (1, 2, 5)),      # shorter: zero-padded
])
def test_a_snapshot_fits_another_input(saved, like, expected):
    from torchbend.bending.snapshot import fit_to
    out = fit_to(torch.ones(saved), torch.zeros(like))
    assert tuple(out.shape) == expected
    if saved[-1] < like[-1]:
        assert out[..., saved[-1]:].abs().sum() == 0


# ── mixing several snapshots ─────────────────────────────────────────────────

def _two():
    a = torch.tensor([[[1., 1., 1., 1.], [0., 0., 0., 0.]]])     # [1, 2 channels, 4 steps]
    b = torch.tensor([[[0., 0., 0., 0.], [3., 3., 3., 3.]]])
    return a, b


def test_mix_modes():
    from torchbend.bending.snapshot import mix_tensors
    a, b = _two()
    one = torch.ones(2)
    assert torch.allclose(mix_tensors([a, b], one, "linear"), (a + b) / 2)
    assert torch.allclose(mix_tensors([a, b], torch.tensor([3., 1.]), "linear"), (3 * a + b) / 4)
    # slerp keeps the average norm along the dimension instead of shrinking
    c = mix_tensors([a, b], one, "slerp", dim=1)
    assert torch.allclose(c.norm(dim=1), torch.full((1, 4), 2.0), atol=1e-5)
    # cosine is the equal-power crossfade: gains cos(t·π/2), sin(t·π/2)
    import math
    for t in (0., 0.25, 0.5, 1.):
        mixed = mix_tensors([a, b], torch.tensor([1 - t, t]), "cosine")
        expect = math.cos(t * math.pi / 2) * a + math.sin(t * math.pi / 2) * b
        assert torch.allclose(mixed, expect, atol=1e-6)
    half = mix_tensors([a, b], one, "cosine")
    assert torch.allclose(half, (a + b) * math.sqrt(0.5), atol=1e-6)      # no dip in level
    # max / min pick, per channel, the source with the largest / smallest amplitude
    assert mix_tensors([a, b], one, "max", dim=1)[0, :, 0].tolist() == [1., 3.]
    assert mix_tensors([a, b], one, "min", dim=1)[0, :, 0].tolist() == [0., 0.]
    # a weight of 0 drops a source
    assert mix_tensors([a, b], torch.tensor([1., 0.]), "max", dim=1)[0, :, 0].tolist() == [1., 0.]
    # sweep crossfades along the dimension, first source to last
    s = mix_tensors([a, b], one, "sweep", dim=2)
    assert torch.allclose(s[0, 0], torch.tensor([1., 2 / 3, 1 / 3, 0.]))
    assert torch.allclose(s[0, 1], torch.tensor([0., 1., 2., 3.]))
    # every weight 0: the live value
    live = torch.full_like(a, 7.)
    assert torch.equal(mix_tensors([a, b], torch.zeros(2), "linear", live=live), live)
    with pytest.raises(ValueError):
        mix_tensors([a, b], one, "nope")
    with pytest.raises(ValueError):
        mix_tensors([a, b], one, "max", dim=5)


def test_a_mix_has_a_weight_per_source():
    from torchbend.bending.snapshot import Mix
    a, b = _two()
    m = Mix.build([a, None], ["a", "live"], mode="linear", dim=1, weights=[1, 0])
    assert list(m.controllable_params) == ["w_0", "w_1"] and type(m).__name__ == "Mix"
    labels = [p["label"] for p in type(m).ui_descriptor()["params"].values()]
    assert labels == ["w · a", "w · live"]
    # weights 1, 0: the snapshot, whatever the live value
    assert torch.allclose(m.bend_input(b, w_0=torch.tensor(1.), w_1=torch.tensor(0.)), a)


def test_mixing_snapshots_into_the_graph(viewer):
    client, gv = viewer
    for name, text in (("rain", "a low rain"), ("bells", "bright bells")):
        client.post("/api/snapshots/", dict(_form(), input_ids=text, name=name))
    rain, bells = _out(client, "a low rain"), _out(client, "bright bells")
    try:
        r = _post(client, "/api/snapshots/mix/", {"fn": "forward", "node": "sum_3",
                                                  "sources": ["rain", "bells"], "mode": "linear",
                                                  "weights": [1, 0]})
        (b,) = r["bindings"]
        assert b["callback_type"] == "Mix" and b["mix"]["sources"] == ["rain", "bells"]
        assert torch.allclose(_out(client, "something else"), rain, atol=1e-5)
        client.patch("/api/bending/%s/" % b["id"], json.dumps({"w_0": 0, "w_1": 1}),
                     content_type="application/json")
        assert torch.allclose(_out(client, "something else"), bells, atol=1e-5)
        # every mode runs, with the live value as a source; one recall per node
        for mode in ("cosine", "slerp", "max", "min", "sweep"):
            r = _post(client, "/api/snapshots/mix/", {"fn": "forward", "node": "sum_3",
                                                      "sources": ["rain", "bells", "live"],
                                                      "mode": mode, "dim": -1})
            assert len(r["bindings"]) == 1 and r["bindings"][0]["mix"]["mode"] == mode
            assert _out(client, "something else").shape == rain.shape
        # saved and restored with the session
        session = gv.get_registry()._entries["audio"].session
        module = gv.get_registry()._entries["audio"].module
        session.session_from_json(module, session.session_to_json())
        assert [b["mix"]["mode"] for b in session.list_bindings()] == ["sweep"]
        # deleting one of its snapshots takes the mix off
        d = client.delete("/api/snapshots/bells/").json()
        assert d["released"] and d["bindings"] == []
    finally:
        _post(client, "/api/snapshots/release/", {"fn": "forward", "node": "sum_3"})
        for name in ("rain", "bells"):
            client.delete("/api/snapshots/%s/" % name)


def test_a_mix_is_changed_in_place(viewer):
    """Mode and dim change on the existing mix: its weights, and a macro
    driving one of them, are left alone."""
    client, gv = viewer
    for name, text in (("rain", "a low rain"), ("bells", "bright bells")):
        client.post("/api/snapshots/", dict(_form(), input_ids=text, name=name))
    try:
        bid = _post(client, "/api/snapshots/mix/", {"fn": "forward", "node": "sum_3",
                                                    "sources": ["rain", "bells"], "mode": "linear",
                                                    "dim": 1, "weights": [0.3, 1]})["binding"]
        _post(client, "/api/bending_params/", {"name": "morph", "value": 0.5})
        _post(client, "/api/bending/%s/link/" % bid, {"param_name": "w_0", "bp_name": "morph"})
        linear = _out(client, "something else")
        r = _post(client, "/api/snapshots/mix/%s/" % bid, {"mode": "max", "dim": None})
        assert r["mix"] == {"sources": ["rain", "bells"], "mode": "max", "dim": None}
        (b,) = r["bindings"]
        assert b["id"] == bid and b["bp_links"] == {"w_0": "morph"}
        assert not torch.allclose(_out(client, "something else"), linear)   # the new mode runs
        # out of range, or unknown: refused, nothing changed
        bad = client.post("/api/snapshots/mix/%s/" % bid, json.dumps({"dim": 7}),
                          content_type="application/json")
        assert bad.status_code == 400 and "out of range" in bad.json()["error"]
        bad = client.post("/api/snapshots/mix/%s/" % bid, json.dumps({"mode": "nope"}),
                          content_type="application/json")
        assert bad.status_code == 400
        session = gv.get_registry()._entries["audio"].session
        assert session.bindings[bid]["mix"] == {"sources": ["rain", "bells"], "mode": "max", "dim": None}
        assert client.post("/api/snapshots/mix/nope/", json.dumps({"mode": "max"}),
                           content_type="application/json").status_code == 404
    finally:
        _post(client, "/api/snapshots/release/", {"fn": "forward", "node": "sum_3"})
        client.delete("/api/bending_params/morph/")
        for name in ("rain", "bells"):
            client.delete("/api/snapshots/%s/" % name)
