# tests/test_altaz_scan_error.py

import pytest
from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u
from pointing.altaz_scan_error import compute_errors, compute_real_trajectory, compute_ideal_trajectory
from hypothesis import given, strategies as st

# Fixed test cases with pytest

def test_compute_errors_basic():
    location = EarthLocation(lat=45.0 * u.deg, lon=7.0 * u.deg, height=1000 * u.m)
    observation_time = Time("2025-12-21T00:00:00")
    max_error, mean_error = compute_errors(location, observation_time, 15, 1.5, 1.0, 0, num_samples=10)
    assert max_error >= 0 * u.arcsec
    assert mean_error >= 0 * u.arcsec


def test_trajectories_shape_consistency():
    location = EarthLocation(lat=45.0 * u.deg, lon=7.0 * u.deg, height=1000 * u.m)
    observation_time = Time("2025-12-21T00:00:00")
    ideal = compute_ideal_trajectory(location, observation_time, 15, 1.5, num_samples=20)
    real = compute_real_trajectory(location, observation_time, 15, 1.5, 1.0, 3, num_samples=20)
    assert len(ideal) == len(real)

# Property-based tests with Hypothesis

@given(
    scan_length=st.floats(min_value=0.1, max_value=5.0),
    delay_time=st.floats(min_value=0.0, max_value=5.0),
    n_interp=st.integers(min_value=0, max_value=5)
)
def test_error_properties_with_hypothesis(scan_length, delay_time, n_interp):
    location = EarthLocation(lat=0 * u.deg, lon=0 * u.deg, height=0 * u.m)
    observation_time = Time("2025-12-21T00:00:00")
    max_error, mean_error = compute_errors(location, observation_time, 15, scan_length, delay_time, n_interp, num_samples=20)
    assert max_error >= 0 * u.arcsec
    assert mean_error >= 0 * u.arcsec
    assert max_error >= mean_error
