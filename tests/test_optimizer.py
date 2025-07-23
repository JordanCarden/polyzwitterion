import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "optimize_cma", Path("2_scripts/optimize_cma.py")
)
optimize_cma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimize_cma)


def test_parse_parameters():
    lines = optimize_cma._load_lines(Path("0_parameters/base.par"))
    params = optimize_cma._parse_parameters(lines)
    assert params.shape == (10,)
    assert np.isclose(params[0], -2.02657743786)


def test_update_roundtrip(tmp_path):
    lines = optimize_cma._load_lines(Path("0_parameters/base.par"))
    params = optimize_cma._parse_parameters(lines)
    new_params = params + 0.1
    updated = optimize_cma._update_parameters(lines, new_params)
    parsed = optimize_cma._parse_parameters(updated)
    assert np.allclose(parsed, new_params)


def test_map_to_valid():
    valid = np.array([-2.0, -1.3, -0.5])
    values = np.array([-1.4, -0.55])
    mapped = optimize_cma._map_to_valid(values, valid)
    assert np.allclose(mapped, [-1.3, -0.5])
