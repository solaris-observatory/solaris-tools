#!/usr/bin/env python3
# Copyright (c) 2025 Marco Buttu
# Author: Marco Buttu - marco.buttu@inaf.it
# License: MIT License

"""
This script samples the pointing error in AltAz coordinates over a specified time range,
at a given sampling frequency. It leverages the functions from the altaz_scan_error module.

Parameters:
- --start-time / -a: ISO format start time of the range (e.g., 2025-12-21T00:00:00).
- --end-time / -b: ISO format end time of the range (e.g., 2025-12-21T03:00:00).
- --frequency / -f: Sampling frequency (default: 30min).
- --show-max-error: Include maximum error in output.
- --show-mean-error: Include mean error in output.
- --plot: Show the plot (only for selected error types).
- --save-plot filename.png: Save the plot to the specified file (extension must be supported by matplotlib).
- --output filename.csv: Save results as CSV file.
- Other parameters are the same as altaz_scan_error CLI arguments.

Example usage:
python check_errors_over_time.py -a "2025-12-21T00:00:00" -b "2025-12-21T03:00:00" --frequency "30min" --show-max-error --show-mean-error --plot --save-plot result.png --output result.csv
"""

import argparse
import os
from astropy.time import Time, TimeDelta
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import csv
from pointing.altaz_scan_error import compute_errors, AzimuthError
from astropy.coordinates import EarthLocation


VALID_PLOT_EXTENSIONS = {".png", ".pdf", ".svg", ".jpg", ".jpeg"}


def parse_args():
    """Setup command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sample pointing error over a time range at given frequency."
    )
    parser.add_argument(
        "-a",
        "--start-time",
        required=True,
        help="ISO format start time (e.g., 2025-12-21T00:00:00)",
    )
    parser.add_argument(
        "-b",
        "--end-time",
        required=True,
        help="ISO format end time (e.g., 2025-12-21T03:00:00)",
    )
    parser.add_argument(
        "-f",
        "--frequency",
        default="30min",
        help="Sampling frequency with astropy units (default: '30min')",
    )
    parser.add_argument(
        "--show-max-error", action="store_true", help="Include max error in output"
    )
    parser.add_argument(
        "--show-mean-error", action="store_true", help="Include mean error in output"
    )
    parser.add_argument("--plot", action="store_true", help="Show plot")
    parser.add_argument("--save-plot", type=str, help="Save plot to specified file")
    parser.add_argument("--output", type=str, help="Save results as CSV file")
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
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    if args.save_plot:
        _, ext = os.path.splitext(args.save_plot)
        if ext.lower() not in VALID_PLOT_EXTENSIONS:
            raise ValueError(
                f"Unsupported plot file extension: {ext}. Supported: {VALID_PLOT_EXTENSIONS}"
            )

    if not args.show_max_error and not args.show_mean_error:
        print(
            "No output type selected: use --show-max-error, --show-mean-error or both."
        )
        return

    start_time = Time(args.start_time)
    end_time = Time(args.end_time)
    freq_quantity = u.Quantity(args.frequency)

    total_seconds = (end_time - start_time).sec
    step_seconds = freq_quantity.to(u.second).value
    sample_count = int(np.floor(total_seconds / step_seconds)) + 1

    sampling_times = start_time + np.arange(sample_count) * freq_quantity

    location = EarthLocation(lat=-75.1 * u.deg, lon=123.35 * u.deg, height=3233 * u.m)
    if args.location == "testa_grigia":
        location = EarthLocation(
            lat=45.8309 * u.deg, lon=7.7864 * u.deg, height=3315 * u.m
        )

    results = []
    for t in sampling_times:
        try:
            max_err, mean_err = compute_errors(
                location,
                t,
                args.duration,
                args.length,
                args.delay,
                args.interpolation_points,
            )
            results.append((t.iso, max_err.value, mean_err.value))
        except AzimuthError:
            continue

    max_of_max_errors = max(r[1] for r in results)
    mean_of_mean_errors = np.mean([r[2] for r in results])

    print(f"\nSummary:")
    if args.show_max_error:
        print(f"Maximum of max errors: {max_of_max_errors:.2f} arcsec")
    if args.show_mean_error:
        print(f"Mean of mean errors: {mean_of_mean_errors:.2f} arcsec")

    if args.output:
        header = ["Time"]
        if args.show_max_error:
            header.append("Max Error (arcsec)")
        if args.show_mean_error:
            header.append("Mean Error (arcsec)")
        header.append(
            f"# Context: duration={args.duration}, length={args.length}, "
            f"delay={args.delay}, location={args.location}, "
            f"interpolation_points={args.interpolation_points}, "
            f"frequency={args.frequency}"
        )
        with open(args.output, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for row in results:
                output_row = [row[0]]
                if args.show_max_error:
                    output_row.append(row[1])
                if args.show_mean_error:
                    output_row.append(row[2])
                writer.writerow(output_row)

    if args.plot or args.save_plot:
        times = [Time(r[0]).datetime for r in results]
        plt.figure()
        plotted = False
        if args.show_max_error:
            max_errors = [r[1] for r in results]
            plt.plot(times, max_errors, marker="o", label="Max Error")
            plotted = True
        if args.show_mean_error:
            mean_errors = [r[2] for r in results]
            plt.plot(times, mean_errors, marker="x", label="Mean Error")
            plotted = True
        if plotted:
            plt.xlabel("Time (UTC)")
            plt.ylabel("Errors (arcsec)")
            plt.grid(True)
            plt.legend()
            context_text = (
                f"{args.location}, {args.start_time} -> {args.end_time}, "
                f"every {args.frequency},\n"
                f"scan duration={args.duration}, length={args.length}, "
                f"delay={args.delay}, points={args.interpolation_points}."
            )
            plt.gcf().text(
                0.02,
                0.02,
                context_text,
                fontsize=8,
                va="bottom",
                ha="left",
                bbox=dict(facecolor="white", edgecolor="black"),
            )
            if args.save_plot:
                plt.savefig(args.save_plot)
            if args.plot:
                plt.show()


if __name__ == "__main__":
    main()
