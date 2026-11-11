"""Skeleton check: every subpackage and module imports cleanly."""

import importlib

import pytest

MODULES = [
    "debias_distill",
    "debias_distill.utils.config",
    "debias_distill.utils.seed",
    "debias_distill.utils.device",
    "debias_distill.utils.runlog",
    "debias_distill.utils.manifest",
    "debias_distill.utils.hub",
    "debias_distill.data.swaps",
    "debias_distill.data.templates",
    "debias_distill.data.corpus",
    "debias_distill.data.names",
    "debias_distill.teacher.align",
    "debias_distill.teacher.cache_targets",
    "debias_distill.distill.losses",
    "debias_distill.distill.train",
    "debias_distill.measure.bias",
    "debias_distill.measure.capability",
    "debias_distill.measure.benchmarks",
    "debias_distill.measure.probe",
    "debias_distill.loop.hotspots",
    "debias_distill.loop.reweight",
    "debias_distill.loop.run_loop",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)
