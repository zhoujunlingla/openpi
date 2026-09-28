"""Check that two-checkpoint mode sends generated text to the action policy."""

# ruff: noqa: SLF001 - tests deliberately exercise private routing internals

import numpy as np
import pytest

from openpi.policies.pi05_text_debug import Pi05TextDebugPolicy


class FakePolicy:
    def __init__(self):
        self.calls = []
        self.reset_count = 0

    def infer(self, obs, *, noise=None):
        self.calls.append((obs, noise))
        return {"actions": np.zeros((10, 7), dtype=np.float32)}

    def reset(self):
        self.reset_count += 1


def make_wrapper(mode="condition"):
    wrapper = object.__new__(Pi05TextDebugPolicy)
    wrapper._policy = FakePolicy()
    wrapper._action_policy = FakePolicy()
    wrapper._mode = mode
    wrapper._log_path = None
    wrapper._pending = {}
    return wrapper


def test_condition_routes_text_to_action_policy():
    wrapper = make_wrapper()
    plan = {"subtask": "pick up the black bowl", "text_unmapped_ids": [], "text_nonlanguage_ids": []}
    obs = {"prompt": "put the black bowl on the plate"}
    result = wrapper._act(obs, plan)
    assert result["action_prompt"] == plan["subtask"]
    assert wrapper._action_policy.calls[0][0]["prompt"] == plan["subtask"]
    assert wrapper._policy.calls == []
    assert obs["prompt"] == "put the black bowl on the plate"


def test_invalid_text_stops_before_action():
    wrapper = make_wrapper()
    plan = {"subtask": "", "text_unmapped_ids": [], "text_nonlanguage_ids": []}
    with pytest.raises(ValueError, match="No fallback"):
        wrapper._act({"prompt": "original task"}, plan)
    assert wrapper._action_policy.calls == []


def test_observe_keeps_original_prompt_and_resets_both_policies():
    wrapper = make_wrapper(mode="observe")
    plan = {"subtask": "pick up the bowl", "text_unmapped_ids": [], "text_nonlanguage_ids": []}
    result = wrapper._act({"prompt": "original task"}, plan)
    assert result["action_prompt"] == "original task"
    wrapper.reset()
    assert wrapper._policy.reset_count == 1
    assert wrapper._action_policy.reset_count == 1
