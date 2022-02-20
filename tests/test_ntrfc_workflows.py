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
    import shutil
    import os

    workdir: Path = Path(tmpdir) #/ "working"
    #workdir.mkdir()

    # Get the path to the snakefile
    src_dir: Path = Path(__file__).absolute().parent.parent
    configfile: Path = src_dir / "examples" / "gwk_compressor_casegeneration" / "casesettings.yaml"
    paramfile: Path = src_dir / "examples" / "gwk_compressor_casegeneration" / "caseparams.tsv"
    paramschema: Path = src_dir / "examples" / "gwk_compressor_casegeneration" / "config.schema.yaml"
    snakefile: Path = src_dir / "examples" / "gwk_compressor_casegeneration" / "Snakefile"
    workflowpath: Path = src_dir / "examples" / "gwk_compressor_casegeneration"
    shutil.copy(configfile, tmpdir)
    shutil.copy(paramfile, tmpdir)
    shutil.copy(paramschema, tmpdir)
    shutil.copy(snakefile, tmpdir)

    # Run Snakemake
    result: bool = snakemake.snakemake(
        snakefile=str(os.path.join(tmpdir,"Snakefile")),
        resources={"mem_gb": 8},
        workdir=str(workdir),
        #lint=True,
        dryrun=True,
        quiet=False,
        #log_handler=[logger.log_handler],
        ignore_ambiguity=True,
    )

    # Check the results
    assert result, "Snakemake did not complete successfully"

