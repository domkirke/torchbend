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
