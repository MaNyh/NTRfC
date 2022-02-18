#!/usr/bin/env python

"""Tests for `ntrfc` package."""

import pytest
import pyvista as pv
import numpy as np

def test_create_case(tmpdir):
    from ntrfc.preprocessing.case_creation.create_case import create_case
    from ntrfc.database.case_templates import list_templates

    templates = list_templates()
    print(templates)
