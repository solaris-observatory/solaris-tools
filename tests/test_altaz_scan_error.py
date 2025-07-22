import pytest
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u
from hypothesis import given, settings, strategies as st
import sys
import os
import subprocess

from pointing.altaz_scan_error import (
    compute_ideal_trajectory,
    compute_real_trajectory,
    compute_errors,
    AzimuthError,
)


@pytest.fixture
def base_params():
    return {
        "location": EarthLocation(
            lat=-75.1 * u.deg, lon=123.35 * u.deg, height=3233 * u.m
        ),
        "observation_time": Time("2025-12-21T02:00:00"),
        "scan_duration_sec": 15.0,
        "scan_length_deg": 1.5,
        "delay_time": 0.0,
        "num_samples": 30,
    }


def test_zero_error_with_max_interpolation(base_params):
    n_interp = base_params["num_samples"] - 2
    max_error, mean_error = compute_errors(**base_params, n_interp=n_interp)
    assert max_error < 1e-6 * u.arcsec
    assert mean_error < 1e-7 * u.arcsec


def test_error_decreases_with_more_points(base_params):
    values = []
    interp_points = [0, 2, 5, 10, base_params["num_samples"] - 2]
    for n_interp in interp_points:
        max_error, mean_error = compute_errors(**base_params, n_interp=n_interp)
        values.append(max_error.value)
    assert all(x >= y for x, y in zip(values, values[1:]))


def test_large_error_with_few_points(base_params):
    max_error, mean_error = compute_errors(**base_params, n_interp=0)
    assert max_error > 0.01 * u.arcsec


def test_ideal_and_real_identical_with_max_points(base_params):
    n_interp = base_params["num_samples"] - 2
    ideal = compute_ideal_trajectory(
        base_params["location"],
        base_params["observation_time"],
        base_params["scan_duration_sec"],
        base_params["scan_length_deg"],
        base_params["num_samples"],
    )
    real = compute_real_trajectory(
        base_params["location"],
        base_params["observation_time"],
        base_params["scan_duration_sec"],
        base_params["scan_length_deg"],
        base_params["delay_time"],
        n_interp,
        base_params["num_samples"],
    )
    az_diff = np.abs((ideal.az - real.az).to(u.arcsec).value)
    alt_diff = np.abs((ideal.alt - real.alt).to(u.arcsec).value)
    assert np.all(az_diff < 1e-6)
    assert np.all(alt_diff < 1e-6)


@given(
    scan_length_deg=st.floats(0.1, 5.0),
    scan_duration_sec=st.floats(1.0, 30.0),
    delay_time=st.floats(0, 2.0),
    num_samples=st.integers(10, 40),
    n_interp=st.integers(0, 38),
)
@settings(max_examples=5)
def test_hypothesis_various_parameters(
    scan_length_deg, scan_duration_sec, delay_time, num_samples, n_interp
):
    n_interp = min(n_interp, num_samples - 2)
    if n_interp < 0:
        n_interp = 0
    location = EarthLocation(lat=-75.1 * u.deg, lon=123.35 * u.deg, height=3233 * u.m)
    observation_time = Time("2025-12-21T02:00:00")
    max_error, mean_error = compute_errors(
        location,
        observation_time,
        scan_duration_sec,
        scan_length_deg,
        delay_time,
        n_interp,
        num_samples,
    )
    assert max_error >= 0 * u.arcsec
    assert mean_error >= 0 * u.arcsec


def test_error_increases_with_delay(base_params):
    n_interp = base_params["num_samples"] - 2
    _, mean_err_no_delay = compute_errors(**base_params, n_interp=n_interp)
    params_with_delay = dict(base_params)
    params_with_delay["delay_time"] = 2.0
    _, mean_err_delay = compute_errors(**params_with_delay, n_interp=n_interp)
    assert mean_err_delay > mean_err_no_delay


def test_error_with_changed_scan_length(base_params):
    params_short = dict(base_params)
    params_short["scan_length_deg"] = 0.2
    params_long = dict(base_params)
    params_long["scan_length_deg"] = 4.0
    max_error_short, _ = compute_errors(**params_short, n_interp=0)
    max_error_long, _ = compute_errors(**params_long, n_interp=0)
    assert max_error_long > max_error_short


def test_error_with_changed_samples(base_params):
    params_low = dict(base_params)
    params_low["num_samples"] = 15
    params_high = dict(base_params)
    params_high["num_samples"] = 35
    max_error_low, _ = compute_errors(**params_low, n_interp=0)
    max_error_high, _ = compute_errors(**params_high, n_interp=0)
    assert max_error_high <= max_error_low * 1.5


def test_real_traj_with_location_testa_grigia(base_params):
    params = dict(base_params)
    params["location"] = EarthLocation(
        lat=45.8309 * u.deg, lon=7.7864 * u.deg, height=3315 * u.m
    )
    max_error, mean_error = compute_errors(**params, n_interp=5)
    assert max_error > 0 * u.arcsec


def test_traj_with_minimum_points(base_params):
    # Use only 2 samples, degenerate case
    params = dict(base_params)
    params["num_samples"] = 2
    max_error, mean_error = compute_errors(**params, n_interp=0)
    assert max_error >= 0 * u.arcsec


def test_AzimuthError_is_raised(monkeypatch, base_params):
    """Force max_error > 100000 arcsec to trigger AzimuthError."""

    def always_big(arr, *args, **kwargs):
        if hasattr(arr, "unit"):
            return 2e5 * arr.unit
        return 2e5

    import pointing.altaz_scan_error

    monkeypatch.setattr(pointing.altaz_scan_error.np, "max", always_big)
    n_interp = base_params["num_samples"] - 2
    with pytest.raises(AzimuthError):
        compute_errors(**base_params, n_interp=n_interp)


def test_AzimuthError_repr_and_doc():
    # Check the exception class can be constructed and has correct message
    err = AzimuthError()
    assert isinstance(err, Exception)
    assert "azimuth" in str(err).lower()


def test_module_docstrings_and_function_docs():
    import pointing.altaz_scan_error as mod

    # All docstrings present
    assert mod.__doc__ is not None
    assert mod.compute_ideal_trajectory.__doc__ is not None
    assert mod.compute_real_trajectory.__doc__ is not None
    assert mod.compute_errors.__doc__ is not None
    assert mod.altaz_to_vec.__doc__ is not None
    assert mod.vec_to_altaz.__doc__ is not None
    assert mod.slerp_vec.__doc__ is not None


def test_command_line_interface(tmp_path):
    """Test the CLI: the script should run and print something sensible."""
    import sys
    import subprocess

    # Trova il path assoluto allo script
    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "pointing", "altaz_scan_error.py")
    )
    result = subprocess.run(
        [
            sys.executable,
            script_path,
            "-p",
            "concordia",
            "-d",
            "10",
            "-l",
            "1",
            "-i",
            "5",
            "--num_samples",
            "10",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    assert "Maximum angular error" in result.stdout
    assert "Mean angular error" in result.stdout
    assert result.returncode == 0
    # Test testa_grigia
    result = subprocess.run(
        [
            sys.executable,
            script_path,
            "-p",
            "testa_grigia",
            "-d",
            "10",
            "-l",
            "1",
            "-i",
            "5",
            "--num_samples",
            "10",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    assert "Maximum angular error" in result.stdout
    assert "Mean angular error" in result.stdout
    assert result.returncode == 0


def test_command_line_interface_with_invalid_location(tmp_path):
    """Test CLI with wrong location value, should fail with error."""
    import sys
    import subprocess

    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "pointing", "altaz_scan_error.py")
    )
    result = subprocess.run(
        [sys.executable, script_path, "-p", "not_a_place"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    assert result.returncode != 0
    assert "error" in result.stderr.lower() or "invalid" in result.stderr.lower()


def test_command_line_interface_with_default_args(tmp_path):
    """Test CLI with no arguments (all defaults)."""
    import sys
    import subprocess

    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "pointing", "altaz_scan_error.py")
    )
    result = subprocess.run(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    assert "Maximum angular error" in result.stdout
    assert "Mean angular error" in result.stdout
    assert result.returncode == 0


def test_slerp_vec_identical_vectors():
    from pointing.altaz_scan_error import slerp_vec
    import numpy as np

    v0 = np.array([1.0, 0.0, 0.0])
    v1 = np.array([1.0, 0.0, 0.0])
    t_arr = np.array([0.0, 0.5, 1.0])
    result = slerp_vec(v0, v1, t_arr)
    expected = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert np.allclose(result, expected)


def test_command_line_concordia(tmp_path):
    import sys
    import os

    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "pointing", "altaz_scan_error.py")
    )
    result = subprocess.run(
        [
            sys.executable,
            script_path,
            "-p",
            "concordia",
            "-d",
            "10",
            "-l",
            "1",
            "-i",
            "5",
            "--num_samples",
            "10",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    assert "Maximum angular error" in result.stdout
    assert result.returncode == 0
