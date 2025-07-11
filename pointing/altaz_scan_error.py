# Copyright (c) 2025 Marco Buttu
# Author: Marco Buttu - marco.buttu@inaf.it
# License: MIT License

"""
This module computes the maximum angular pointing error in AltAz coordinates
introduced during a scan in Right Ascension (RA) under two combined sources
of error:

1. Timing delay: the telescope moves with a fixed delay with respect to
   the ideal tracking trajectory.

2. Interpolation in AltAz: instead of following the ideal scanning path in
   equatorial coordinates (RA/Dec), the movement is approximated in horizontal
   coordinates (Az/El) using a polyline with interpolation points.

Functions provided:
- compute_ideal_trajectory: Computes the ideal AltAz scanning trajectory.
- compute_real_trajectory: Computes the real (interpolated) scanning trajectory.
- compute_errors: Computes maximum and mean angular errors between ideal and real trajectories.

Usage:
- Import the module and use the functions in your own scripts.
- Run as a script for command-line usage.
"""

import argparse
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u
import numpy as np


def compute_ideal_trajectory(
    location: EarthLocation,
    observation_time: Time,
    scan_duration_sec: float,
    scan_length_deg: float,
    num_samples: int = 100,
) -> SkyCoord:
    """
    Compute the ideal scanning trajectory in AltAz coordinates by calculating
    the apparent position of the Sun in equatorial coordinates (RA/Dec) along
    a linear scan path defined by a given length and duration, then transforming
    these positions into the horizontal coordinate system (Azimuth/Altitude)
    for a specified observing location and time.

    Parameters:
        location (EarthLocation): Observing location.
        observation_time (Time): Start time of observation.
        scan_duration_sec (float): Scan duration in seconds.
        scan_length_deg (float): Scan length in degrees.
        num_samples (int): Number of samples along the trajectory.

    Returns:
        SkyCoord: Ideal trajectory as SkyCoord object in AltAz frame.
    """
    sun_coord = get_sun(observation_time).transform_to("icrs")
    central_ra = sun_coord.ra.deg
    central_dec = sun_coord.dec.deg

    start_ra = central_ra - scan_length_deg / (2 * np.cos(np.radians(central_dec)))
    end_ra = central_ra + scan_length_deg / (2 * np.cos(np.radians(central_dec)))

    ideal_ras = np.linspace(start_ra, end_ra, num_samples)
    ideal_coords = SkyCoord(ra=ideal_ras * u.deg, dec=central_dec * u.deg, frame="icrs")
    ideal_times = (
        observation_time + np.linspace(0, scan_duration_sec, num_samples) * u.second
    )
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
    n_interp: int,
    num_samples: int = 100,
) -> SkyCoord:
    """
    Compute the real scanning trajectory in AltAz coordinates by simulating
    a delayed and interpolated version of the ideal path. The function applies
    a fixed timing delay to the scan start time and approximates the trajectory
    using a specified number of interpolation points in RA. These interpolated
    positions are then converted into the horizontal coordinate system
    (Azimuth/Altitude) for the given location and observation time.

    Parameters:
        location (EarthLocation): Observing location.
        observation_time (Time): Start time of observation.
        scan_duration_sec (float): Scan duration in seconds.
        scan_length_deg (float): Scan length in degrees.
        delay_time (float): Delay in seconds.
        n_interp (int): Number of interpolation points.
        num_samples (int): Number of samples along the trajectory.

    Returns:
        SkyCoord: Real trajectory as SkyCoord object in AltAz frame.
    """
    sun_coord = get_sun(observation_time).transform_to("icrs")
    central_ra = sun_coord.ra.deg
    central_dec = sun_coord.dec.deg

    start_ra = central_ra - scan_length_deg / (2 * np.cos(np.radians(central_dec)))
    end_ra = central_ra + scan_length_deg / (2 * np.cos(np.radians(central_dec)))

    real_start_time = observation_time + delay_time * u.second
    real_end_time = real_start_time + scan_duration_sec * u.second

    if n_interp == 0:
        start_coord = SkyCoord(
            ra=start_ra * u.deg, dec=central_dec * u.deg, frame="icrs"
        )
        end_coord = SkyCoord(ra=end_ra * u.deg, dec=central_dec * u.deg, frame="icrs")
        start_altaz = start_coord.transform_to(
            AltAz(obstime=real_start_time, location=location)
        )
        end_altaz = end_coord.transform_to(
            AltAz(obstime=real_end_time, location=location)
        )
        interp_az = np.linspace(start_altaz.az.deg, end_altaz.az.deg, num_samples)
        interp_el = np.linspace(start_altaz.alt.deg, end_altaz.alt.deg, num_samples)
        interp_times = (
            real_start_time + np.linspace(0, scan_duration_sec, num_samples) * u.second
        )
    else:
        segment_ras = np.linspace(start_ra, end_ra, n_interp + 2)
        segment_times_all = np.linspace(
            real_start_time.unix, real_end_time.unix, n_interp + 2
        )
        segment_times_all = Time(segment_times_all, format="unix")
        times_for_interpolation = (
            real_start_time + np.linspace(0, scan_duration_sec, num_samples) * u.second
        )
        ras_for_interpolation = np.interp(
            times_for_interpolation.jd, segment_times_all.jd, segment_ras
        )
        interp_coords = SkyCoord(
            ra=ras_for_interpolation * u.deg, dec=central_dec * u.deg, frame="icrs"
        )
        interp_altaz = interp_coords.transform_to(
            AltAz(obstime=times_for_interpolation, location=location)
        )
        interp_az = interp_altaz.az.deg
        interp_el = interp_altaz.alt.deg
        interp_times = times_for_interpolation

    return SkyCoord(
        az=interp_az * u.deg,
        alt=interp_el * u.deg,
        frame=AltAz(obstime=interp_times, location=location),
    )


def compute_errors(
    location: EarthLocation,
    observation_time: Time,
    scan_duration_sec: float,
    scan_length_deg: float,
    delay_time: float,
    n_interp: int,
    num_samples: int = 100,
) -> tuple[u.Quantity, u.Quantity]:
    """
    Compute maximum and mean angular errors between ideal and real scanning
    trajectories.

    This function calculates both the maximum and the mean angular pointing error
    in AltAz coordinates introduced by a fixed timing delay and path interpolation.
    The ideal trajectory is computed as a continuous scan in RA/Dec converted to AltAz.
    The real trajectory is simulated by delaying the scan start time and approximating
    the path using a specified number of interpolation points in RA.

    Parameters:
        location (EarthLocation): Observing location.
        observation_time (Time): Start time of observation.
        scan_duration_sec (float): Scan duration in seconds.
        scan_length_deg (float): Scan length in degrees.
        delay_time (float): Delay in seconds.
        n_interp (int): Number of interpolation points in RA.
        num_samples (int): Number of samples along the trajectory.

    Returns:
        tuple: (max_error, mean_error), both as astropy Quantity in arcseconds.
        max_error represents the highest angular deviation between the ideal and real trajectory.
        mean_error represents the average angular deviation over all sampled points.
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
    mean_error = np.mean(angular_error)

    return max_error, mean_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute pointing error in AltAz due to delay and path interpolation."
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
        help="Observing location",
    )
    parser.add_argument(
        "-t",
        "--observation_time",
        type=str,
        default="2025-12-21T02:00:00",
        help="Observation time in UTC",
    )
    parser.add_argument(
        "-s",
        "--shift",
        "--delay",
        type=float,
        dest="delay",
        default=1.0,
        help="Delay in seconds (default: 1.0)",
    )
    parser.add_argument(
        "-n",
        "--interpolation_points",
        type=int,
        default=0,
        help="Number of intermediate interpolation points (default: 0)",
    )
    args = parser.parse_args()

    location = EarthLocation(lat=-75.1 * u.deg, lon=123.35 * u.deg, height=3233 * u.m)
    if args.location == "testa_grigia":
        location = EarthLocation(
            lat=45.8309 * u.deg, lon=7.7864 * u.deg, height=3315 * u.m
        )

    max_error, mean_error = compute_errors(
        location,
        Time(args.observation_time),
        args.duration,
        args.length,
        args.delay,
        args.interpolation_points,
    )

    print(f"Maximum angular error in AltAz during scan: {max_error:.2f}")
    print(f"Mean angular error in AltAz during scan: {mean_error:.2f}")
