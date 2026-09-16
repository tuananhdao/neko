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
    """Build one instrumented solver and the meshes used by the IDP tests."""
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

    utility_environment = environment.copy()
    if conftest.USES_DEVICE:
        utility_environment["CUDA_VISIBLE_DEVICES"] = "0"

    compile_result = subprocess.run(
        [str(makeneko), str(USER_FILE)],
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=utility_environment,
    )
    assert compile_result.returncode == 0, (
        "makeneko failed for the Euler IDP integration driver:\n"
        + compile_result.stdout
    )

    def generate_mesh(name, elements, periodic):
        mesh_result = subprocess.run(
            [
                str(genmeshbox),
                "0", "1", "0", "1", "0", "1",
                *(str(value) for value in elements),
                *(".true." if value else ".false." for value in periodic),
            ],
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            env=utility_environment,
        )
        assert mesh_result.returncode == 0, (
            f"genmeshbox failed for {name}:\n" + mesh_result.stdout
        )

        mesh = work_dir / f"{name}.nmsh"
        (work_dir / "box.nmsh").replace(mesh)
        check_result = subprocess.run(
            [str(mesh_checker), mesh.name],
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            env=utility_environment,
        )
        assert check_result.returncode == 0, (
            f"mesh_checker rejected {name}:\n" + check_result.stdout
        )
        return mesh

    periodic_mesh = generate_mesh(
        "periodic_box", (4, 4, 1), (True, True, True)
    )
    bounded_x_mesh = generate_mesh(
        "bounded_x_box", (4, 2, 1), (False, True, True)
    )
    bounded_xy_mesh = generate_mesh(
        "bounded_xy_box", (4, 2, 1), (False, False, True)
    )

    return {
        "directory": work_dir,
        "environment": environment,
        "mesh": periodic_mesh,
        "bounded_x_mesh": bounded_x_mesh,
        "bounded_xy_mesh": bounded_xy_mesh,
        "neko": work_dir / "neko",
    }


def _case(
    problem,
    runtime,
    polynomial_order,
    steps,
    timestep,
    gamma=1.4,
    mesh=None,
    boundary_conditions=None,
    internal_energy_floor=NEAR_VACUUM_FLOOR,
    limit_entropy=True,
    time_order=3,
):
    """Build a short, output-free Euler IDP case."""
    output_dir = runtime["directory"] / (
        f"output_{problem}_p{polynomial_order}_r{time_order}"
    )
    output_dir.mkdir(exist_ok=True)
    case_data = {
        "version": 1.0,
        "test": {"problem": problem},
        "case": {
            "mesh_file": str(mesh or runtime["mesh"]),
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
                "time_order": time_order,
                "polynomial_order": polynomial_order,
                "c_avisc_low": 0.5,
                "c_avisc_entropy": 1.0,
                "euler_idp": {
                    "enabled": True,
                    "low_order_only": False,
                    "relax_density_bounds": False,
                    "limit_internal_energy": True,
                    "limit_entropy": limit_entropy,
                    "internal_energy_floor": internal_energy_floor,
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
    if boundary_conditions is not None:
        case_data["case"]["fluid"]["boundary_conditions"] = (
            boundary_conditions
        )
    return case_data


def _parse_summary(log_path):
    """Parse the rank-zero summary emitted by the user file."""
    tags = {
        "EULER_IDP_RESULT": None,
        "EULER_IDP_ERROR": None,
        "EULER_IDP_DRIFT": None,
        "EULER_IDP_LIMITER": None,
        "EULER_IDP_LIMITER_STATS": None,
        "EULER_IDP_LIMITER_COUNTS": None,
        "EULER_IDP_STATE": None,
        "EULER_IDP_BOUNDS": None,
        "EULER_IDP_CONSERVATION": None,
        "EULER_IDP_CORRECTION": None,
        "EULER_IDP_BOUNDARY": None,
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
        "limiter_stats": np.asarray(
            tags["EULER_IDP_LIMITER_STATS"], dtype=float
        ),
        "limiter_counts": np.asarray(
            tags["EULER_IDP_LIMITER_COUNTS"], dtype=int
        ),
        "state": np.asarray(tags["EULER_IDP_STATE"], dtype=float),
        "bounds": np.asarray(tags["EULER_IDP_BOUNDS"], dtype=float),
        "stage_conservation": np.asarray(
            tags["EULER_IDP_CONSERVATION"], dtype=float
        ),
        "correction": np.asarray(tags["EULER_IDP_CORRECTION"], dtype=float),
        "boundary": np.asarray(tags["EULER_IDP_BOUNDARY"], dtype=float),
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
    mesh=None,
    boundary_conditions=None,
    internal_energy_floor=NEAR_VACUUM_FLOOR,
    limit_entropy=True,
    time_order=3,
):
    """Write, run, and parse one generated Euler IDP case."""
    case_data = _case(
        problem,
        runtime,
        polynomial_order,
        steps,
        timestep,
        gamma,
        mesh,
        boundary_conditions,
        internal_energy_floor,
        limit_entropy,
        time_order,
    )
    suffix = (
        f"{problem}_p{polynomial_order}_r{time_order}_n{mpi_ranks}"
    )
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
    assert summary["stage_count"] == time_order * steps
    assert summary["finite"]
    for key in (
        "errors",
        "drifts",
        "limiter",
        "limiter_stats",
        "state",
        "bounds",
        "stage_conservation",
        "correction",
        "boundary",
    ):
        assert np.all(np.isfinite(summary[key])), f"non-finite {key}: {summary[key]}"
    assert np.all(summary["limiter_counts"] >= 0)
    limiter_min, limiter_mean, limiter_max, limited_fraction = (
        summary["limiter_stats"]
    )
    tolerance = _tolerances()["roundoff"]
    assert -tolerance <= limiter_min <= limiter_mean + tolerance
    assert limiter_mean <= limiter_max + tolerance
    assert limiter_max <= 1.0 + tolerance
    assert -tolerance <= limited_fraction <= 1.0 + tolerance
    assert np.max(summary["correction"]) <= _tolerances()["correction"]
    return summary


def _primitive_boundary_conditions(boundary_type, density=1.4):
    """Return the case entries for one isolated x-boundary contract."""
    if boundary_type == "prescribed":
        return [
            {
                "type": "velocity_value",
                "zone_indices": [1, 2],
                "value": [0.8, -0.2, 0.1],
            },
            {
                "type": "density_value",
                "zone_indices": [1, 2],
                "value": density,
            },
            {
                "type": "pressure_value",
                "zone_indices": [1, 2],
                "value": 1.0,
            },
        ]
    return [{"type": boundary_type, "zone_indices": [1, 2]}]


def _assert_boundary_contract(summary, floor=NEAR_VACUUM_FLOOR):
    """Check the synthetic boundary map and evolved state."""
    tolerance = _tolerances()
    assert np.max(summary["boundary"][:7]) <= tolerance["roundoff"]
    assert summary["boundary"][7] >= floor
    assert np.min(summary["state"][[0, 2]]) > 0.0
    assert np.min(summary["state"][[1, 3]]) >= floor


def _tolerances():
    """Working-precision tolerances for roundoff and global reductions."""
    if conftest.RP == "sp":
        return {
            "roundoff": 3.0e-5,
            "conservation": 3.0e-5,
            "correction": 3.0e-5,
            "bounds": 3.0e-5,
            "rank_rtol": 8.0e-5,
            "rank_atol": 8.0e-6,
            "stat_rtol": 2.0e-4,
            "stat_atol": 2.0e-5,
        }
    return {
        "roundoff": 2.0e-12,
        "conservation": 2.0e-12,
        "correction": 2.0e-11,
        "bounds": 2.0e-12,
        "rank_rtol": 2.0e-11,
        "rank_atol": 2.0e-12,
        "stat_rtol": 2.0e-8,
        "stat_atol": 2.0e-10,
    }


@pytest.mark.parametrize("time_order", (1, 3))
def test_idp_free_stream(
    launcher_script, log_file, euler_idp_runtime, time_order
):
    """A uniform periodic Euler state stays constant to roundoff."""
    summary = _run_case(
        launcher_script,
        log_file,
        euler_idp_runtime,
        "free_stream",
        polynomial_order=4,
        steps=12,
        timestep=2.0e-4,
        time_order=time_order,
    )
    tolerance = _tolerances()

    assert np.max(summary["errors"]) <= tolerance["roundoff"]
    assert np.max(summary["drifts"]) <= tolerance["conservation"]
    assert summary["limiter"][1] == 0.0
    assert math.isclose(
        summary["limiter"][0], 1.0, abs_tol=tolerance["roundoff"]
    )


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


@pytest.mark.parametrize("time_order", (1, 3))
def test_idp_periodic_discontinuity_activates_limiter(
    launcher_script, log_file, euler_idp_runtime, time_order
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
        time_order=time_order,
    )
    tolerance = _tolerances()

    assert summary["limiter"][1] > 0.0
    assert summary["limiter"][0] < 1.0
    assert np.min(summary["state"][[0, 2]]) > 0.0
    assert np.min(summary["state"][[1, 3]]) >= NEAR_VACUUM_FLOOR
    assert np.max(summary["bounds"]) <= tolerance["bounds"]
    assert np.max(summary["drifts"]) <= tolerance["conservation"]


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
        assert summary["stage_count"] > 3
        assert np.min(summary["state"][[0, 2]]) > 0.0
        assert np.min(summary["state"][[1, 3]]) >= NEAR_VACUUM_FLOOR
        assert summary["limiter"][2] <= 1.0 + GRAPH_CFL_TOL
        runs.append(summary)

    tolerance = _tolerances()
    diagnostic_keys = (
        "drifts",
        "limiter",
        "limiter_stats",
        "limiter_counts",
        "state",
        "bounds",
        "stage_conservation",
        "correction",
    )
    for key in diagnostic_keys:
        if key == "limiter_counts":
            np.testing.assert_array_equal(
                runs[0][key],
                runs[1][key],
                err_msg="1-rank and 2-rank limiter counts differ",
            )
            continue
        relative_tolerance = tolerance["rank_rtol"]
        absolute_tolerance = tolerance["rank_atol"]
        if key == "limiter_stats":
            relative_tolerance = tolerance["stat_rtol"]
            absolute_tolerance = tolerance["stat_atol"]
        np.testing.assert_allclose(
            runs[0][key],
            runs[1][key],
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            err_msg=f"1-rank and 2-rank {key} diagnostics differ",
        )


def test_idp_rejects_unsupported_gamma(
    launcher_script, log_file, euler_idp_runtime
):
    """The IDP wave-speed guarantee is restricted to gamma <= 5/3."""
    with pytest.raises(
        AssertionError,
        match=r"Euler IDP requires finite gamma with 1 < gamma <= 5/3",
    ):
        _run_case(
            launcher_script,
            log_file,
            euler_idp_runtime,
            "free_stream",
            polynomial_order=2,
            steps=1,
            timestep=1.0e-4,
            gamma=1.8,
        )


@pytest.mark.parametrize(
    "boundary_type",
    ("prescribed", "symmetry", "slip", "outflow", "normal_outflow"),
)
def test_idp_boundary_map_contracts(
    launcher_script, log_file, euler_idp_runtime, boundary_type
):
    """Each supported Euler boundary map preserves its primitive contract."""
    summary = _run_case(
        launcher_script,
        log_file,
        euler_idp_runtime,
        f"boundary_{boundary_type}",
        polynomial_order=3,
        steps=2,
        timestep=1.0e-6,
        mesh=euler_idp_runtime["bounded_x_mesh"],
        boundary_conditions=_primitive_boundary_conditions(boundary_type),
    )
    _assert_boundary_contract(summary)


def test_idp_outflow_respects_internal_energy_floor(
    launcher_script, log_file, euler_idp_runtime
):
    """Pressure reconstruction honors a configured floor above 1e-12."""
    floor = 1.0e-8
    summary = _run_case(
        launcher_script,
        log_file,
        euler_idp_runtime,
        "boundary_outflow_floor",
        polynomial_order=3,
        steps=2,
        timestep=1.0e-6,
        mesh=euler_idp_runtime["bounded_x_mesh"],
        boundary_conditions=_primitive_boundary_conditions("outflow"),
        internal_energy_floor=floor,
    )
    _assert_boundary_contract(summary, floor=floor)


def test_idp_rejects_nonpositive_prescribed_boundary_density(
    launcher_script, log_file, euler_idp_runtime
):
    """An inadmissible prescribed primitive state fails at the boundary map."""
    with pytest.raises(AssertionError, match=r"Euler IDP.*density"):
        _run_case(
            launcher_script,
            log_file,
            euler_idp_runtime,
            "boundary_invalid_density",
            polynomial_order=2,
            steps=1,
            timestep=1.0e-6,
            mesh=euler_idp_runtime["bounded_x_mesh"],
            boundary_conditions=_primitive_boundary_conditions(
                "prescribed", density=-1.0
            ),
        )


def test_idp_mixed_boundaries_are_rank_invariant(
    launcher_script, log_file, euler_idp_runtime
):
    """Inlet, slip walls, and outflow compose through SSPRK3 on 1/2 ranks."""
    if configure_nprocs(2) < 2:
        pytest.skip("This test requires two MPI ranks")

    boundary_conditions = [
        {
            "type": "velocity_value",
            "zone_indices": [1],
            "value": [0.8, 0.0, 0.1],
        },
        {"type": "density_value", "zone_indices": [1], "value": 1.4},
        {"type": "pressure_value", "zone_indices": [1], "value": 1.0},
        {"type": "outflow", "zone_indices": [2]},
        {"type": "slip", "zone_indices": [3, 4]},
    ]
    runs = []
    for mpi_ranks in (1, 2):
        summary = _run_case(
            launcher_script,
            log_file,
            euler_idp_runtime,
            "boundary_mixed",
            polynomial_order=3,
            steps=3,
            timestep=1.0e-9,
            mpi_ranks=mpi_ranks,
            mesh=euler_idp_runtime["bounded_xy_mesh"],
            boundary_conditions=boundary_conditions,
            limit_entropy=False,
        )
        _assert_boundary_contract(summary)
        runs.append(summary)

    tolerance = _tolerances()
    for key in (
        "limiter",
        "state",
        "bounds",
        "stage_conservation",
        "boundary",
    ):
        np.testing.assert_allclose(
            runs[0][key],
            runs[1][key],
            rtol=tolerance["rank_rtol"],
            atol=tolerance["rank_atol"],
            err_msg=f"1-rank and 2-rank {key} diagnostics differ",
        )
