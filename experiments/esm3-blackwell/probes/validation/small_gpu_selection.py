"""Explicitly select GPU unit/kernel tests that do not require published checkpoints."""

import json
import os
from pathlib import Path

import pytest

# tests/conftest.py identifies these three published-weight fixtures. The
# adapter checkpoint fixture is also excluded to avoid downloading another model.
WEIGHTS_FIXTURES = {
    "esmc_300m_dir",
    "esmfold2_dir",
    "esmfold2_experimental_dir",
    "hf_port_checkpoint_dir",
}


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    selected, excluded = [], []
    for item in items:
        weights = sorted(WEIGHTS_FIXTURES.intersection(item.fixturenames))
        if weights:
            excluded.append((item, weights))
        else:
            selected.append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=[item for item, _ in excluded])
    Path(os.environ["ESM_SELECTION_REPORT"]).write_text(
        json.dumps(
            {
                "scope": "GPU-marked tests without published-model fixture dependencies; full-weight GPU tests are excluded, not passed.",
                "selected": [item.nodeid for item in selected],
                "excluded": [
                    {"nodeid": item.nodeid, "weight_fixtures": weights}
                    for item, weights in excluded
                ],
            },
            indent=2,
        )
        + "\n"
    )
