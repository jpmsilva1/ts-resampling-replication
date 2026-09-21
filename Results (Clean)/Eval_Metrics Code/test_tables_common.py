#!/usr/bin/env python3
"""Guards on the shared constants every table generator depends on.

These exist because the same class of bug has now been found four times in
this project: a container sized for 20 datasets that silently drops DS21-24
rather than failing (rdata_to_csv.R's `for (i in 1:20)`, generate_paper_
comparison's `N_OURS = 20`, causal_effects' SOURCE_MAP, and DS_LABELS here).
A missing label does not raise -- the dataset just vanishes from the table --
so the check has to be explicit.
"""
import csv

from paths import MANIFEST, ROOT
from tables_common import DS_LABELS, STRATEGIES, parse_workflow


def _manifest_ids() -> list[str]:
    with open(MANIFEST, newline="") as fh:
        return [row["dataset_id"] for row in csv.DictReader(fh)]


def test_ds_labels_cover_every_dataset_in_the_manifest():
    missing = [d for d in _manifest_ids() if d not in DS_LABELS]
    assert not missing, f"DS_LABELS is missing {missing} -- they would be dropped from every table"


def test_ds_labels_have_no_extras():
    """An id in DS_LABELS but not the manifest means a stale or typo'd key."""
    extra = sorted(set(DS_LABELS) - set(_manifest_ids()))
    assert not extra, f"DS_LABELS has ids absent from the manifest: {extra}"


def test_root_is_derived_not_hardcoded():
    """paths.ROOT must point at a real tree containing this code, so a clone
    on another machine resolves correctly (see paths.py docstring)."""
    assert (ROOT / "Results (Clean)/Eval_Metrics Code/paths.py").exists()


def test_parse_workflow_round_trips_every_strategy():
    for fam in ("lm", "svm", "mars", "rf", "rpart"):
        for strat in STRATEGIES:
            wf = f"mc.{fam}" if strat == "baseline" else f"mc.{fam}_{strat}"
            assert parse_workflow(wf) == (fam, strat)
