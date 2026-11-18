"""Offline evaluation of claim extraction.

- `dataset`: loads and validates labelled evaluation datasets (read-only).
- `runner`: runs the existing extraction pipeline on every case.
- `predictions`: the stored run format, saved and reloaded without any LLM call.

Scoring (claim matching and metrics) is a separate, later step that reads saved runs.
"""
