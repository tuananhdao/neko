import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest
import conftest

from testlib import get_genmeshbox, get_makeneko


TEST_DIR = Path(__file__).resolve().parent
BASELINE_PATTERN = re.compile(r"EULER_IDP_BASELINE\s+\d+\s+(.+)")


def runtime_environment():
    environment = os.environ.copy()
    if sys.platform != "darwin":
        return environment

    library_dirs = []
    for package in ("json-fortran", "hdf5_fortran"):
        result = subprocess.run(
            ["pkg-config", "--variable=libdir", package],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            library_dirs.append(result.stdout.strip())
    if environment.get("DYLD_LIBRARY_PATH"):
        library_dirs.append(environment["DYLD_LIBRARY_PATH"])
    environment["DYLD_LIBRARY_PATH"] = ":".join(library_dirs)
    environment["OMPI_MCA_mca_base_env_list"] = (
        "DYLD_LIBRARY_PATH=" + environment["DYLD_LIBRARY_PATH"]
    )
    return environment


def generate_mesh(work_dir, environment):
    result = subprocess.run(
        [
            get_genmeshbox(),
            "0", "1", "0", "1", "0", "0.25",
            "4", "4", "1",
            ".true.", ".true.", ".true.",
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def euler_idp_directory(tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("euler_idp")
    environment = runtime_environment()
    result = subprocess.run(
        [get_makeneko(), str(TEST_DIR / "euler_idp_baseline.f90")],
        cwd=work_dir,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    generate_mesh(work_dir, environment)
    return work_dir


def load_case():
    return json.loads(
        (TEST_DIR / "euler_idp_baseline.case").read_text(encoding="utf-8")
    )


def run_case(
    launcher_script,
    work_dir,
    case_data,
    name,
    conserved_dump=None,
):
    case_file = work_dir / f"{name}.case"
    case_file.write_text(json.dumps(case_data), encoding="utf-8")
    environment = runtime_environment()
    environment.pop("NEKO_TEST_IDP_DUMP", None)
    if conserved_dump is not None:
        conserved_dump.unlink(missing_ok=True)
        environment["NEKO_TEST_IDP_DUMP"] = str(conserved_dump.resolve())
    return subprocess.run(
        [
            str(Path(launcher_script).resolve()),
            "1",
            str(case_file),
            str(work_dir / "neko"),
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        env=environment,
    )


def baseline_values(output):
    match = BASELINE_PATTERN.search(output)
    assert match is not None, output
    return np.fromstring(match.group(1), sep=" ")


def baseline_tolerances():
    if conftest.RP == "sp":
        return 2.0e-6, 2.0e-10
    return 5.0e-12, 5.0e-13


def recorded_baseline(time_order):
    baselines = json.loads(
        (TEST_DIR / "baselines.json").read_text(encoding="utf-8")
    )
    return np.asarray(
        baselines["baselines"][conftest.RP][f"time_order_{time_order}"]
    )


@pytest.mark.parametrize("time_order", [1, 3])
@pytest.mark.parametrize("idp_setting", ["omitted", "false"])
def test_disabled_euler_baseline(
    launcher_script, euler_idp_directory, backend, time_order, idp_setting
):
    case_data = load_case()
    case_data["case"]["numerics"]["time_order"] = time_order
    if idp_setting == "false":
        case_data["case"]["numerics"]["euler_idp"] = {"enabled": False}

    result = run_case(
        launcher_script,
        euler_idp_directory,
        case_data,
        f"baseline_rk{time_order}_{idp_setting}",
    )
    assert result.returncode == 0, result.stdout + result.stderr

    if backend == "cpu":
        expected = recorded_baseline(time_order)
        rtol, atol = baseline_tolerances()
        np.testing.assert_allclose(
            baseline_values(result.stdout), expected, rtol=rtol, atol=atol
        )


@pytest.mark.parametrize("time_order", [1, 3, 4])
def test_enabled_euler_idp_matches_legacy_advance(
    launcher_script, euler_idp_directory, backend, time_order
):
    legacy_case = load_case()
    legacy_case["case"]["numerics"]["time_order"] = time_order
    legacy_dump = euler_idp_directory / f"legacy_parity_rk{time_order}.bin"
    legacy_result = run_case(
        launcher_script,
        euler_idp_directory,
        legacy_case,
        f"legacy_parity_rk{time_order}",
        legacy_dump,
    )
    assert legacy_result.returncode == 0, (
        legacy_result.stdout + legacy_result.stderr
    )

    enabled_case = load_case()
    enabled_case["case"]["numerics"]["time_order"] = time_order
    enabled_case["case"]["numerics"]["euler_idp"] = {"enabled": True}
    enabled_dump = euler_idp_directory / f"enabled_parity_rk{time_order}.bin"
    enabled_result = run_case(
        launcher_script,
        euler_idp_directory,
        enabled_case,
        f"enabled_parity_rk{time_order}",
        enabled_dump,
    )
    output = enabled_result.stdout + enabled_result.stderr
    assert enabled_result.returncode == 0, output
    legacy_values = baseline_values(legacy_result.stdout)
    enabled_values = baseline_values(enabled_result.stdout)
    np.testing.assert_array_equal(enabled_values, legacy_values)
    assert legacy_dump.stat().st_size > 0
    assert enabled_dump.read_bytes() == legacy_dump.read_bytes()
    if backend == "cpu" and time_order in (1, 3):
        expected = recorded_baseline(time_order)
        rtol, atol = baseline_tolerances()
        np.testing.assert_allclose(
            enabled_values, expected, rtol=rtol, atol=atol
        )


def test_enabled_euler_idp_requires_compressible_scheme(
    launcher_script, euler_idp_directory
):
    case_data = load_case()
    case_data["case"]["numerics"]["euler_idp"] = {"enabled": True}
    case_data["case"]["fluid"]["scheme"] = "pnpn"

    result = run_case(
        launcher_script,
        euler_idp_directory,
        case_data,
        "invalid_scheme",
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Euler IDP requires case.fluid.scheme = compressible" in output
