# Copyright (c) 2025 Marco Buttu
# Author: Marco Buttu - marco.buttu@inaf.it
# License: MIT License

"""
This module computes the maximum and mean angular pointing errors, in horizontal
(AltAz) coordinates, that occur when a telescope scans the Sun along Right Ascension (RA).
It simulates both the ideal scan and a real scan that approximates the ideal trajectory
using a finite number of control points and interpolation. The real scan uses spherical
linear interpolation (slerp) between control points for a more realistic simulation.

Main features:
- Calculates the ideal trajectory in AltAz (theoretical best path) for a scan along RA.
- Simulates the real trajectory, which uses interpolation in AltAz between a user-defined
  number of control points.
- Computes the maximum and mean pointing errors between the ideal and real trajectories.

This is useful for telescope control software developers and astronomers who want
to understand the effect of trajectory approximation on pointing precision.

Usage:
- Import the module and use the functions in your own scripts.
- Or run it from the command line to test with different parameters.
"""

import argparse
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u
import numpy as np


class AzimuthError(Exception):
    """
    Exception raised if the trajectory wraps around the azimuth (i.e., passes
    through a full 360° turn), which is not handled by this script.
    """

    def __init__(self):
        super().__init__("Trajectory crosses a full azimuth turn.")


def altaz_to_vec(az_deg, alt_deg):
    """
    Convert azimuth and altitude angles (in degrees) to a 3D unit vector.

    Parameters:
        az_deg (float or array): Azimuth angle(s) in degrees [0-360], measured from North.
        alt_deg (float or array): Altitude angle(s) in degrees [−90 (horizon) to +90 (zenith)].

    Returns:
        numpy.ndarray: Array of shape (..., 3) with the [x, y, z] unit vectors.
    """
    az_rad = np.radians(az_deg)
    alt_rad = np.radians(alt_deg)
    x = np.cos(alt_rad) * np.cos(az_rad)
    y = np.cos(alt_rad) * np.sin(az_rad)
    z = np.sin(alt_rad)
    return np.stack((x, y, z), axis=-1)


def vec_to_altaz(vec):
    """
    Convert a 3D unit vector to azimuth and altitude angles (in degrees).

    Parameters:
        vec (numpy.ndarray): Array of 3D unit vectors (..., 3).

    Returns:
        tuple:
            az (numpy.ndarray): Azimuth angle(s) in degrees [0-360].
            alt (numpy.ndarray): Altitude angle(s) in degrees [−90 to +90].
    """
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    r = np.hypot(x, y)
    az = (np.degrees(np.arctan2(y, x))) % 360
    alt = np.degrees(np.arctan2(z, r))
    return az, alt


def slerp_vec(v0, v1, t_arr):
    """
    Perform spherical linear interpolation (slerp) between two 3D vectors.

    Slerp smoothly interpolates between v0 and v1 along the shortest path
    on the unit sphere.

    Parameters:
        v0 (numpy.ndarray): Starting vector (shape (3,))
        v1 (numpy.ndarray): Ending vector (shape (3,))
        t_arr (numpy.ndarray): Array of interpolation parameters, in [0, 1].
            0 returns v0, 1 returns v1.

    Returns:
        numpy.ndarray: Interpolated vectors (shape (len(t_arr), 3))
    """
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    dot = np.clip(np.dot(v0, v1), -1, 1)
    omega = np.arccos(dot)
    if np.isclose(omega, 0):
        # The two vectors are almost identical, so just return copies of v0
        return np.outer(np.ones_like(t_arr), v0)
    sin_omega = np.sin(omega)
    return (
        np.sin((1 - t_arr) * omega)[:, None] * v0 + np.sin(t_arr * omega)[:, None] * v1
    ) / sin_omega


def compute_ideal_trajectory(
    location: EarthLocation,
    observation_time: Time,
    scan_duration_sec: float,
    scan_length_deg: float,
    num_samples: int = 100,
) -> SkyCoord:
    """
    Compute the ideal scanning trajectory in AltAz coordinates.

    The ideal trajectory is defined by moving linearly in RA (with Dec fixed to
    the Sun's center) over the scan length, converting each sampled position
    to AltAz.

    Parameters:
        location (EarthLocation): Observer's location on Earth.
        observation_time (Time): Start time of the scan (UTC).
        scan_duration_sec (float): Total duration of the scan, in seconds.
        scan_length_deg (float): Total length of the scan, in degrees (in RA).
        num_samples (int): Number of points in the trajectory.

    Returns:
        SkyCoord: The ideal trajectory, as AltAz positions at the right times.
    """
    # Get the Sun's center coordinates at the observation time
    sun_coord = get_sun(observation_time).transform_to("icrs")
    central_ra = sun_coord.ra.deg
    central_dec = sun_coord.dec.deg

    # Calculate starting and ending RA, keeping Dec fixed
    start_ra = central_ra - scan_length_deg / (2 * np.cos(np.radians(central_dec)))
    end_ra = central_ra + scan_length_deg / (2 * np.cos(np.radians(central_dec)))

    # Generate num_samples evenly spaced RA values along the scan
    ideal_ras = np.linspace(start_ra, end_ra, num_samples)
    ideal_coords = SkyCoord(ra=ideal_ras * u.deg, dec=central_dec * u.deg, frame="icrs")

    # Generate the corresponding times for each sample point
    ideal_times = (
        observation_time + np.linspace(0, scan_duration_sec, num_samples) * u.second
    )
    # Convert the ideal RA/Dec samples to AltAz
    ideal_altaz = ideal_coords.transform_to(
        AltAz(obstime=ideal_times, location=location)
    )
    return ideal_altaz


def compute_real_trajectory(
    location: EarthLocation,
    observation_time: Time,
    scan_duration_sec: float,
    scan_length_deg: float,
    delay_time: float,
    n_interp: int = 0,
    num_samples: int = 100,
) -> SkyCoord:
    """
    Compute the "real" scan trajectory in AltAz coordinates.

    The real trajectory is simulated by:
      - Generating a set of (n_interp + 2) control points in RA, evenly spaced
        between the start and end of the scan, and at evenly spaced times.
      - Converting each control point to AltAz.
      - Using spherical linear interpolation (slerp) in AltAz between the control
        points to generate a trajectory with exactly num_samples points.

    This method guarantees that when n_interp = num_samples - 2, the real
    trajectory matches the ideal one (error = 0).

    Parameters:
        location (EarthLocation): Observer's location on Earth.
        observation_time (Time): Start time of the scan (UTC).
        scan_duration_sec (float): Total duration of the scan, in seconds.
        scan_length_deg (float): Total length of the scan, in degrees (in RA).
        delay_time (float): How much the scan is delayed (in seconds).
        n_interp (int): Number of *intermediate* control points (total control points = n_interp + 2).
        num_samples (int): Number of points in the output trajectory.

    Returns:
        SkyCoord: The real (approximated) trajectory as AltAz positions at the right times.
    """
    # Get the Sun's center coordinates at the observation time
    sun_coord = get_sun(observation_time).transform_to("icrs")
    central_ra = sun_coord.ra.deg
    central_dec = sun_coord.dec.deg

    # Calculate starting and ending RA
    start_ra = central_ra - scan_length_deg / (2 * np.cos(np.radians(central_dec)))
    end_ra = central_ra + scan_length_deg / (2 * np.cos(np.radians(central_dec)))
    real_start_time = observation_time + delay_time * u.second

    # Generate control points in RA, evenly spaced, and their times
    cpoints_ra = np.linspace(start_ra, end_ra, n_interp + 2)
    cpoints_times = (
        real_start_time + np.linspace(0, scan_duration_sec, n_interp + 2) * u.second
    )
    cpoints_icrs = SkyCoord(
        ra=cpoints_ra * u.deg, dec=central_dec * u.deg, frame="icrs"
    )
    # Convert control points to AltAz coordinates
    cpoints_altaz = cpoints_icrs.transform_to(
        AltAz(obstime=cpoints_times, location=location)
    )
    ctrl_vecs = altaz_to_vec(cpoints_altaz.az.deg, cpoints_altaz.alt.deg)
    ctrl_times_jd = cpoints_altaz.obstime.jd

    # Create num_samples points, distributed proportionally along the full scan (from 0 to 1)
    t_global = np.linspace(0, 1, num_samples)
    n_segments = len(ctrl_vecs) - 1
    seg_edges = np.linspace(0, 1, n_segments + 1)  # Edges of the segments

    # For each sample t_global, find which segment it is in, and the local position in the segment (t_local)
    seg_idx = np.searchsorted(seg_edges, t_global, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, n_segments - 1)
    t_local = (t_global - seg_edges[seg_idx]) / (seg_edges[1] - seg_edges[0])

    # Slerp for each segment
    traj_vecs = np.array(
        [
            slerp_vec(ctrl_vecs[i], ctrl_vecs[i + 1], np.array([tl]))[0]
            for i, tl in zip(seg_idx, t_local)
        ]
    )
    # Interpolate time as well (linear between segment endpoints)
    traj_times = (
        ctrl_times_jd[seg_idx] * (1 - t_local) + ctrl_times_jd[seg_idx + 1] * t_local
    )
    traj_az, traj_alt = vec_to_altaz(traj_vecs)
    traj_times_isot = Time(traj_times, format="jd", scale="utc").isot

    return SkyCoord(
        az=traj_az * u.deg,
        alt=traj_alt * u.deg,
        frame=AltAz(obstime=traj_times_isot, location=location),
    )


def compute_errors(
    location: EarthLocation,
    observation_time: Time,
    scan_duration_sec: float,
    scan_length_deg: float,
    delay_time: float,
    n_interp: int = 0,
    num_samples: int = 100,
) -> tuple[u.Quantity, u.Quantity]:
    """
    Compute the maximum and mean angular pointing errors (in arcseconds) between
    the ideal and real scan trajectories in AltAz coordinates.

    Parameters:
        location (EarthLocation): Observer's location.
        observation_time (Time): Start time of the scan.
        scan_duration_sec (float): Scan duration, in seconds.
        scan_length_deg (float): Scan length, in degrees (RA).
        delay_time (float): Delay in seconds (between ideal and real scan).
        n_interp (int): Number of intermediate control points (see compute_real_trajectory).
        num_samples (int): Number of points in each trajectory.

    Returns:
        tuple: (max_error, mean_error), both as astropy Quantity in arcseconds.
            - max_error: The highest error during the scan.
            - mean_error: The average error during the scan.
    """
    ideal_altaz = compute_ideal_trajectory(
        location, observation_time, scan_duration_sec, scan_length_deg, num_samples
    )
    real_altaz = compute_real_trajectory(
        location,
        observation_time,
        scan_duration_sec,
        scan_length_deg,
        delay_time,
        n_interp,
        num_samples,
    )
    az_error = (real_altaz.az - ideal_altaz.az).to(u.arcsec)
    el_error = (real_altaz.alt - ideal_altaz.alt).to(u.arcsec)
    angular_error = np.sqrt(az_error**2 + el_error**2)
    max_error = np.max(angular_error)
    if max_error > 100000 * u.arcsec:
        raise AzimuthError()
    mean_error = np.mean(angular_error)
    return max_error, mean_error


def main():
    parser = argparse.ArgumentParser(
        description="Compute pointing error in AltAz due to delay and path interpolation. "
        "The real trajectory uses spherical interpolation between AltAz control points."
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=15.0,
        help="Scan duration in seconds (default: 15.0)",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=float,
        default=1.5,
        help="Scan length in degrees (default: 1.5)",
    )
    parser.add_argument(
        "-p",
        "--place",
        "--location",
        dest="location",
        type=str,
        choices=["concordia", "testa_grigia"],
        default="concordia",
        help="Observing location. Use 'concordia' or 'testa_grigia'.",
    )
    parser.add_argument(
        "-t",
        "--observation_time",
        type=str,
        default="2025-12-21T02:00:00",
        help="Observation time in UTC, format: YYYY-MM-DDTHH:MM:SS",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Total number of samples in the scan trajectory (default: 100)",
    )
    parser.add_argument(
        "-s",
        "--shift",
        "--delay",
        type=float,
        dest="delay",
        default=1.0,
        help="Delay in seconds between the ideal and real scan (default: 1.0)",
    )
    parser.add_argument(
        "-i",
        "--interpolation_points",
        type=int,
        default=0,
        help="Number of intermediate interpolation (control) points (default: 0). "
        "Total control points = i + 2",
    )
    args = parser.parse_args()

    # Set the observing location based on user selection
    location = EarthLocation(lat=-75.1 * u.deg, lon=123.35 * u.deg, height=3233 * u.m)
    if args.location == "testa_grigia":
        location = EarthLocation(
            lat=45.8309 * u.deg, lon=7.7864 * u.deg, height=3315 * u.m
        )

    # Compute and print errors
    max_error, mean_error = compute_errors(
        location,
        Time(args.observation_time),
        args.duration,
        args.length,
        args.delay,
        args.interpolation_points,
        args.num_samples,
    )

    print(f"Maximum angular error in AltAz during scan: {max_error:.2f}")
    print(f"Mean angular error in AltAz during scan: {mean_error:.2f}")


if __name__ == "__main__":
    main()
