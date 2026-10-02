#!/usr/bin/env python3
"""Live plot of the mean episode reward from a training log CSV.

    python -m src.live_plot checkpoints/bc_rl/training_log_e2e.csv

Started automatically by src.train_marl (disable with --no-live-plot). It runs
in its own process and only reads the CSV, so closing the window never affects
training.
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def read_rewards(path: str, column: str) -> tuple[list[int], list[float]]:
    """Return (update, value) for rows that already have finished episodes."""
    updates, values = [], []
    try:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                try:
                    update, value = int(row['update']), float(row[column])
                    if int(row['episodes']) == 0:
                        continue
                except (KeyError, TypeError, ValueError):
                    continue  # partially written last line
                updates.append(update)
                values.append(value)
    except FileNotFoundError:
        pass
    return updates, values


def ema(values: list[float], alpha: float) -> list[float]:
    out, acc = [], None
    for v in values:
        acc = v if acc is None else alpha * v + (1.0 - alpha) * acc
        out.append(acc)
    return out


def main():
    parser = argparse.ArgumentParser(description='Live plot of a training log CSV')
    parser.add_argument('log', help='Path to the training log CSV')
    parser.add_argument('--column', default='mean_ep_reward')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='Seconds between checks for new rows')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='EMA weight of the newest point (1 disables smoothing)')
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.canvas.manager.set_window_title(f'Training: {os.path.dirname(os.path.abspath(args.log))}')
    raw_line, = ax.plot([], [], color='tab:blue', alpha=0.35, linewidth=1, label='raw')
    smooth_line, = ax.plot([], [], color='tab:blue', linewidth=2, label='EMA')
    ax.set_xlabel('update')
    ax.set_ylabel(args.column)
    ax.set_title(args.column)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    last_size = [-1]

    def refresh(_frame):
        try:
            size = os.path.getsize(args.log)
        except OSError:
            return raw_line, smooth_line
        if size == last_size[0]:
            return raw_line, smooth_line
        last_size[0] = size
        updates, values = read_rewards(args.log, args.column)
        if not updates:
            return raw_line, smooth_line
        raw_line.set_data(updates, values)
        smooth_line.set_data(updates, ema(values, args.smoothing))
        ax.set_title(f'{args.column}  (update {updates[-1]}: {values[-1]:.2f})')
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()
        return raw_line, smooth_line

    _anim = FuncAnimation(fig, refresh, interval=int(args.interval * 1000),
                          cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
