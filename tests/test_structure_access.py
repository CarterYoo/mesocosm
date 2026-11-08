"""Graph source grants constrain actual service listings and content routes."""
from __future__ import annotations

import pytest
from test_artifact_exchange import artifact_settings, payload
from test_services import client

from mesocosm.core.access import AccessSpec, validate_access
from mesocosm.env.services.app import create_app


async def test_source_specific_artifact_grants_filter_lists_and_deny_known_ids_across_hosts(tmp_path):
    st = artifact_settings(tmp_path)
    st.chat = {"channels": [], "max_chars": None, "max_messages": None}
    st.cache = {"writable": True, "list_recent": True, "seed": {}}
    st.access = AccessSpec(services={"board": {"read": ["agent-1", "agent-2", "agent-3"]}},
                          artifacts={"agent-1": ["agent-3"], "agent-2": ["agent-3"]})
    app = create_app(st)
    async with client(app, "172.30.0.4", "services", api_key="test-control") as admin:
        first = (await admin.post("/admin/artifacts", json=payload())).json()
        second = (await admin.post("/admin/artifacts", json=payload("worker two", source_agent="agent-2"))).json()
    for host in ("board", "wiki", "cache", "chat"):
        async with client(app, "172.30.0.12", host) as denied:
            listing = (await denied.get("/api/artifacts")).json()
            assert listing == {"artifacts": [], "total": 0, "offset": 0}
            for receipt in (first, second):
                assert (await denied.get(f'/api/artifacts/{receipt["id"]}/files/src/answer.py')).status_code == 403
    async with client(app, "172.30.0.13", "board") as reviewer:
        visible = (await reviewer.get("/api/artifacts")).json()
        assert {row["id"] for row in visible["artifacts"]} == {first["id"], second["id"]}
        assert (await reviewer.get(f'/api/artifacts/{first["id"]}/files/src/answer.py')).status_code == 200
    st.access.services["board"].read = ["agent-1"]
    app = create_app(st)
    async with client(app, "172.30.0.13", "board") as reviewer:
        assert (await reviewer.get("/api/artifacts")).status_code == 403
        assert (await reviewer.get(f'/api/artifacts/{first["id"]}/files/src/answer.py')).status_code == 403


@pytest.mark.parametrize("artifacts", [{"unknown": ["agent-1"]}, {"agent-1": ["unknown"]},
                                       {"agent-1": ["agent-2", "agent-2"]}])
def test_artifact_grants_validate_both_endpoints_and_duplicates(artifacts):
    errors = validate_access(AccessSpec(artifacts=artifacts), ["agent-1", "agent-2"], [], [])
    assert errors


async def test_graph_review_assignments_allow_empty_workers_but_deny_unassigned_reviews(tmp_path):
    st = artifact_settings(tmp_path)
    st.chat = {"channels": [], "max_chars": None, "max_messages": None}
    st.peer_review = {"protocol": "graph", "topology": "all"}
    st.access = AccessSpec(services={"chat": {"read": ["agent-1", "agent-2", "agent-3"],
                                            "write": ["agent-1", "agent-2", "agent-3"]}})
    app = create_app(st)
    async with client(app, "172.30.0.4", "services", api_key="test-control") as admin:
        response = await admin.post("/admin/epoch", json={"epoch": "task", "assignments": {
            "agent-1": [], "agent-2": [], "agent-3": ["agent-1", "agent-2"]}})
        assert response.status_code == 200
    async with client(app, "172.30.0.11", "chat") as worker:
        assert (await worker.post("/api/verdicts", json={"about": "agent-2", "verdict": "accept"})).status_code == 403
    async with client(app, "172.30.0.13", "chat") as reviewer:
        assert (await reviewer.post("/api/verdicts", json={"about": "agent-1", "verdict": "reject"})).status_code == 201
