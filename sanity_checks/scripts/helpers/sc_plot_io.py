"""
Plot I/O helpers for sanity-check scripts.

REASONING:
- Purpose: centralize figure save behavior and output-dir creation.
- Reproducibility: shared save options make QC figure output consistent.
"""

from __future__ import annotations


def save_figure(fig, output_dir, filename, *, dpi=150, bbox_inches="tight", **savefig_kwargs):
    """Ensure output_dir exists, then save figure and return full path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / filename
    fig.savefig(plot_path, dpi=dpi, bbox_inches=bbox_inches, **savefig_kwargs)
    return plot_path

