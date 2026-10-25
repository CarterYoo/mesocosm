"""Regression contract for OpenCode v1.19.3 permission precedence."""
import fnmatch
import json

import pytest

from mesocosm.adapters.opencode.client import render_opencode_config


def config(tools):
    return render_opencode_config(gateway_url="http://gateway:8080", model_id="model", api_key="test", tools=tools)


def effective(permission, tool):
    # OpenCode v1.19.3 uses the last matching permission rule. JSON order matters.
    rules = {"*": permission} if isinstance(permission, str) else permission
    return next(value for pattern, value in reversed(list(rules.items())) if fnmatch.fnmatchcase(tool, pattern))


def test_explicit_disables_override_default_allow_after_json_roundtrip():
    tools = {"task": False, "question": False, "todowrite": False}
    value = json.loads(json.dumps(config(tools)))
    assert list(value["permission"])[0] == "*"
    assert all(effective(value["permission"], name) == "deny" for name in tools)
    assert effective(value["permission"], "bash") == "allow"
    assert "tools" not in value


def test_enabled_tools_and_default_contract_are_preserved():
    assert config(None)["permission"] == config({})["permission"] == "allow"
    value = config({"task": False, "bash": True})
    assert effective(value["permission"], "task") == "deny"
    assert effective(value["permission"], "bash") == "allow"


@pytest.mark.parametrize("alias", ["write", "edit", "patch"])
def test_legacy_edit_aliases_use_the_official_permission_name(alias):
    value = config({alias: False})
    assert effective(value["permission"], "edit") == "deny"
    assert "tools" not in value


def test_explicit_wildcard_and_more_specific_allow_keep_input_order():
    value = config({"*": False, "bash": True})
    assert effective(value["permission"], "task") == "deny"
    assert effective(value["permission"], "bash") == "allow"
