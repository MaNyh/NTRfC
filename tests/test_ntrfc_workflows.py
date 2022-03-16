#!/usr/bin/env python

"""Tests for `ntrfc` package."""

from pathlib import Path
import snakemake
from py._path.local import LocalPath as TmpDir


def test_snakefile(tmpdir: TmpDir) -> None:
    """
    this test is working, but its not suitable for long-term-development
    it copies (deploys) the workflow into TmpDir and it is executing a snakemake dry-run
    """
    # Create input files
    import os
    from ntrfc.utils.snake_utils.deployment import deploy

    workdir: Path = Path(tmpdir)
    # Get the path to the snakefile
    deploy("case_creation", workdir)
    # Run Snakemake
    result: bool = snakemake.snakemake(
        snakefile=str(os.path.join(tmpdir, "Snakefile")),
        resources={"mem_gb": 8},
        workdir=str(workdir),
        # lint=True,
        dryrun=True,
        quiet=False,
        verbose=True,
        # log_handler=[logger.log_handler],
        ignore_ambiguity=True,
    )

    # Check the results
    assert result, "Snakemake did not complete successfully"
