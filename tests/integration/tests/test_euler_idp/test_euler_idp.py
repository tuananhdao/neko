"""End-to-end tests for the invariant-domain-preserving Euler path."""

import json
import math
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import conftest
from testlib import configure_nprocs, get_genmeshbox, get_makeneko


HERE = Path(__file__).resolve().parent
USER_FILE = HERE / "euler_idp_integration.f90"
N_STAGES = 3
GRAPH_CFL_TOL = 5.0e-6
NEAR_VACUUM_FLOOR = 1.0e-12


def _resolve_executable(command):
    """Return an absolute executable path before changing directories."""
    path = Path(command)
    if path.is_file():
        return path.resolve()
    resolved = shutil.which(command)
    if resolved is None:
        pytest.fail(f"Could not locate required executable: {command}")
    return Path(resolved).resolve()


def _tail(path, line_count=60):
    """Return the end of a log in an assertion message."""
    if not path.is_file():
        return "(log file was not created)"
    return "".join(path.read_text(errors="replace").splitlines(True)[-line_count:])


@pytest.fixture(scope="session")
def euler_idp_runtime(tmp_path_factory):
    """Build one instrumented solver and one checked periodic box mesh."""
    work_dir = tmp_path_factory.mktemp("euler_idp")
    makeneko = _resolve_executable(get_makeneko())
    genmeshbox = _resolve_executable(get_genmeshbox())
    mesh_checker = _resolve_executable(genmeshbox.with_name("mesh_checker"))

    environment = os.environ.copy()
    install_lib = makeneko.parent.parent / "lib"
    if platform.system().lower() == "darwin":
        variable = "DYLD_LIBRARY_PATH"
        value = os.pathsep.join(
            filter(None, (str(install_lib), environment.get(variable, "")))
        )
        environment[variable] = value
        environment["OMPI_MCA_mca_base_env_list"] = f"{variable}={value}"
    else:
        variable = "LD_LIBRARY_PATH"
        environment[variable] = os.pathsep.join(
            filter(None, (str(install_lib), environment.get(variable, "")))
        )

    compile_result = subprocess.run(
        [str(makeneko), str(USER_FILE)],
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=environment,
    )
    assert compile_result.returncode == 0, (
        "makeneko failed for the Euler IDP integration driver:\n"
        + compile_result.stdout
    )

    mesh_result = subprocess.run(
        [
            str(genmeshbox),
            "0", "1", "0", "1", "0", "1",
            "4", "4", "1", ".true.", ".true.", ".true.",
        ],
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=environment,
    )
    assert mesh_result.returncode == 0, (
        "genmeshbox failed for the periodic Euler IDP mesh:\n"
        + mesh_result.stdout
    )

    mesh = work_dir / "box.nmsh"
    check_result = subprocess.run(
        [str(mesh_checker), mesh.name],
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=environment,
    )
    assert check_result.returncode == 0, (
        "mesh_checker rejected the periodic Euler IDP mesh:\n"
        + check_result.stdout
    )

    return {
        "directory": work_dir,
        "environment": environment,
        "mesh": mesh,
        "neko": work_dir / "neko",
    }


def _case(problem, runtime, polynomial_order, steps, timestep, gamma=1.4):
    """Build a short, output-free, fully periodic SSPRK3 case."""
    output_dir = runtime["directory"] / (
        f"output_{problem}_p{polynomial_order}"
    )
    output_dir.mkdir(exist_ok=True)
    floor = NEAR_VACUUM_FLOOR
    return {
        "version": 1.0,
        "test": {"problem": problem},
        "case": {
            "mesh_file": str(runtime["mesh"]),
            "output_directory": str(output_dir),
            "output_at_end": False,
            "output_boundary": False,
            "output_checkpoints": False,
            "time": {
                "end_time": steps * timestep,
                "timestep": timestep,
                "variable_timestep": False,
            },
            "numerics": {
                "time_order": 3,
                "polynomial_order": polynomial_order,
                "c_avisc_low": 0.5,
                "c_avisc_entropy": 1.0,
                "euler_idp": {
                    "enabled": True,
                    "low_order_only": False,
                    "relax_density_bounds": False,
                    "limit_internal_energy": True,
                    "limit_entropy": True,
                    "internal_energy_floor": floor,
                    "diagnostics_level": "full",
                    "diagnostics_interval": 1,
                },
            },
            "fluid": {
                "scheme": "compressible",
                "gamma": gamma,
                "initial_condition": {"type": "user"},
                "output_control": "never",
            },
        },
    }


def _parse_summary(log_path):
    """Parse the rank-zero summary emitted by the user file."""
    tags = {
        "EULER_IDP_RESULT": None,
        "EULER_IDP_ERROR": None,
        "EULER_IDP_DRIFT": None,
        "EULER_IDP_LIMITER": None,
        "EULER_IDP_STATE": None,
        "EULER_IDP_BOUNDS": None,
        "EULER_IDP_CONSERVATION": None,
    }
    for line in log_path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        for tag in tags:
            if stripped.startswith(tag + " "):
                tags[tag] = stripped.split()[1:]

    missing = [tag for tag, values in tags.items() if values is None]
    assert not missing, f"Missing {missing} in log:\n{_tail(log_path)}"

    result = tags["EULER_IDP_RESULT"]
    summary = {
        "problem": result[0],
        "polynomial_order": int(result[1]),
        "mpi_ranks": int(result[2]),
        "steps": int(result[3]),
        "stage_count": int(result[4]),
        "finite": result[5] == "T",
        "errors": np.asarray(tags["EULER_IDP_ERROR"], dtype=float),
        "drifts": np.asarray(tags["EULER_IDP_DRIFT"], dtype=float),
        "limiter": np.asarray(tags["EULER_IDP_LIMITER"], dtype=float),
        "state": np.asarray(tags["EULER_IDP_STATE"], dtype=float),
        "bounds": np.asarray(tags["EULER_IDP_BOUNDS"], dtype=float),
        "stage_conservation": np.asarray(
            tags["EULER_IDP_CONSERVATION"], dtype=float
        ),
    }
    return summary


def _run_case(
    launcher_script,
    log_file,
    runtime,
    problem,
    polynomial_order,
    steps,
    timestep,
    mpi_ranks=1,
    gamma=1.4,
):
    """Write, run, and parse one generated Euler IDP case."""
    case_data = _case(
        problem, runtime, polynomial_order, steps, timestep, gamma
    )
    suffix = f"{problem}_p{polynomial_order}_n{mpi_ranks}"
    case_file = runtime["directory"] / f"{suffix}.case"
    case_file.write_text(json.dumps(case_data, indent=2) + "\n")

    log_path = Path(log_file).with_name(f"{suffix}.log")
    command = [
        str(_resolve_executable(launcher_script)),
        str(mpi_ranks),
        str(case_file),
        str(runtime["neko"]),
    ]
    with log_path.open("w") as output:
        result = subprocess.run(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            env=runtime["environment"],
        )
    assert result.returncode == 0, (
        f"Euler IDP case {suffix} failed with code {result.returncode}:\n"
        + _tail(log_path)
    )
    summary = _parse_summary(log_path)
    assert summary["problem"] == problem
    assert summary["polynomial_order"] == polynomial_order
    assert summary["mpi_ranks"] == mpi_ranks
    assert summary["steps"] == steps
    assert summary["stage_count"] == N_STAGES * steps
    assert summary["finite"]
    for key in (
        "errors",
        "drifts",
        "limiter",
        "state",
        "bounds",
        "stage_conservation",
    ):
        assert np.all(np.isfinite(summary[key])), f"non-finite {key}: {summary[key]}"
    return summary


def _tolerances():
    """Working-precision tolerances for roundoff and global reductions."""
    if conftest.RP == "sp":
        return {
            "roundoff": 3.0e-5,
            "conservation": 3.0e-5,
            "bounds": 3.0e-5,
            "rank_rtol": 8.0e-5,
            "rank_atol": 8.0e-6,
        }
    return {
        "roundoff": 2.0e-12,
        "conservation": 2.0e-12,
        "bounds": 2.0e-12,
        "rank_rtol": 2.0e-11,
        "rank_atol": 2.0e-12,
    }


@pytest.mark.skipif(conftest.USES_DEVICE, reason="Euler IDP is CPU-only")
def test_idp_free_stream(launcher_script, log_file, euler_idp_runtime):
    """A uniform periodic Euler state stays constant to roundoff."""
    summary = _run_case(
        launcher_script,
        log_file,
        euler_idp_runtime,
        "free_stream",
        polynomial_order=4,
        steps=12,
        timestep=2.0e-4,
    )
    tolerance = _tolerances()

    assert np.max(summary["errors"]) <= tolerance["roundoff"]
    assert np.max(summary["drifts"]) <= tolerance["conservation"]
    assert summary["limiter"][1] == 0.0
    assert math.isclose(
        summary["limiter"][0], 1.0, abs_tol=tolerance["roundoff"]
    )


@pytest.mark.skipif(conftest.USES_DEVICE, reason="Euler IDP is CPU-only")
def test_idp_smooth_transport(launcher_script, log_file, euler_idp_runtime):
    """The periodic density wave converges under p-refinement."""
    tolerance = _tolerances()
    runs = {}
    for polynomial_order in (2, 4):
        runs[polynomial_order] = _run_case(
            launcher_script,
            log_file,
            euler_idp_runtime,
            "smooth_transport",
            polynomial_order=polynomial_order,
            steps=100,
            timestep=5.0e-4,
        )
        summary = runs[polynomial_order]
        assert np.min(summary["state"][[0, 2]]) > 0.0
        assert np.min(summary["state"][[1, 3]]) >= NEAR_VACUUM_FLOOR
        assert np.max(summary["bounds"]) <= tolerance["bounds"]
        assert summary["limiter"][2] <= 1.0 + GRAPH_CFL_TOL

    assert np.max(runs[4]["errors"]) < np.max(runs[2]["errors"])
    assert runs[4]["errors"][0] < runs[2]["errors"][0]


@pytest.mark.skipif(conftest.USES_DEVICE, reason="Euler IDP is CPU-only")
def test_idp_periodic_discontinuity_activates_limiter(
    launcher_script, log_file, euler_idp_runtime
):
    """A periodic pair of Sod jumps activates the limiter without drift."""
    summary = _run_case(
        launcher_script,
        log_file,
        euler_idp_runtime,
        "periodic_discontinuity",
        polynomial_order=4,
        steps=12,
        timestep=1.0e-4,
    )
    tolerance = _tolerances()

    assert summary["limiter"][1] > 0.0
    assert summary["limiter"][0] < 1.0
    assert np.min(summary["state"][[0, 2]]) > 0.0
    assert np.min(summary["state"][[1, 3]]) >= NEAR_VACUUM_FLOOR
    assert np.max(summary["bounds"]) <= tolerance["bounds"]
    assert np.max(summary["drifts"]) <= tolerance["conservation"]


@pytest.mark.skipif(conftest.USES_DEVICE, reason="Euler IDP is CPU-only")
def test_idp_near_vacuum_evolution(
    launcher_script, log_file, euler_idp_runtime
):
    """Leblanc-type data evolves safely and has rank-invariant diagnostics."""
    if configure_nprocs(2) < 2:
        pytest.skip("This test requires two MPI ranks")
    assert NEAR_VACUUM_FLOOR < 1.0e-10

    runs = []
    for mpi_ranks in (1, 2):
        summary = _run_case(
            launcher_script,
            log_file,
            euler_idp_runtime,
            "near_vacuum",
            polynomial_order=4,
            steps=8,
            timestep=1.0e-4,
            mpi_ranks=mpi_ranks,
            gamma=5.0 / 3.0,
        )
        assert summary["stage_count"] > N_STAGES
        assert np.min(summary["state"][[0, 2]]) > 0.0
        assert np.min(summary["state"][[1, 3]]) >= NEAR_VACUUM_FLOOR
        assert summary["limiter"][2] <= 1.0 + GRAPH_CFL_TOL
        runs.append(summary)

    tolerance = _tolerances()
    diagnostic_keys = (
        "drifts",
        "limiter",
        "state",
        "bounds",
        "stage_conservation",
    )
    for key in diagnostic_keys:
        np.testing.assert_allclose(
            runs[0][key],
            runs[1][key],
            rtol=tolerance["rank_rtol"],
            atol=tolerance["rank_atol"],
            err_msg=f"1-rank and 2-rank {key} diagnostics differ",
        )
