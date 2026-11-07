"""Durability, identity and independent audit software contracts."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from mesocosm.reproduction.store import Store, digest, write_json


def test_private_root_and_exclusive_writes(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="private"):
        Store(shared)
    root = tmp_path / "private"
    store = Store(root)
    identifier = store.put("record", {"x": 1})
    assert store.get(identifier) == {"x": 1}
    target = root / "record.json"
    write_json(target, {"x": 1})
    with pytest.raises(FileExistsError):
        write_json(target, {"x": 2})
    link = tmp_path / "link"
    link.symlink_to(root)
    with pytest.raises(ValueError, match="symlink"):
        Store(link)


def test_one_claim_under_concurrency_and_changed_inputs_rejected(tmp_path):
    store = Store(tmp_path / "private")

    def claim(_):
        try:
            store.begin("same-key", "training", {"input": 1})
            return "claimed"
        except ValueError:
            return "already-running"

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(claim, range(8)))
    assert results.count("claimed") == 1
    store.finish("same-key", {"job_id": "one"})
    assert store.begin("same-key", "training", {"input": 1}) == {"job_id": "one"}
    with pytest.raises(ValueError, match="different inputs"):
        store.begin("same-key", "training", {"input": 2})
    assert len(store.audit_records()) == 2


def test_object_and_audit_tampering_detected(tmp_path):
    store = Store(tmp_path / "private")
    identifier = store.put("record", {"x": 1})
    store.audit("first", {"record_id": identifier})
    store.audit("second", {})
    with store.connection() as db:
        db.execute("UPDATE objects SET body=? WHERE id=?", ('{"x":2}', identifier))
    with pytest.raises(ValueError, match="hash"):
        store.get(identifier)
    with store.connection() as db:
        db.execute("UPDATE audit SET sha256=? WHERE sequence=1", (digest({}),))
    with pytest.raises(ValueError, match="chain"):
        store.audit_records()
