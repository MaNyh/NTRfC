import numpy as np
import pytest

from ntrfc.turbo.cascade_case.utils.probecontainer import ProbeContainer


@pytest.fixture
def probe_container():
    container = ProbeContainer()
    container.add_probe(
        position=np.array([0, 0, 0]),
        values={'value1': np.array([1, 2, 3]), 'value2': np.array([[1, 2, 3], [4, 5, 6], [3, 2, 1]])},
        groupname='Group1',
        timesteps=np.array([0, 1, 2])
    )
    container.add_probe(
        position=np.array([1, 1, 1]),
        values={'value2': np.array([[1, 2, 3], [4, 2, 4], [7, 8, 9]]), 'value1': np.array([4, 5, 6])},
        groupname='Group2',
        timesteps=np.array([0, 1, 2])
    )
    container.add_probe(
        position=np.array([0, 0, 0]),
        values={'value2': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 'value1': np.array([2, 2, 2])},
        groupname='Group2',
        timesteps=np.array([0, 1, 2])
    )
    return container


def test_add_probe(probe_container):
    assert len(probe_container.probes) == 3


def test_filter_probes_by_groupname(probe_container):
    filtered_probes = probe_container.filter_probes(by_groupname='Group1')
    assert len(filtered_probes) == 1
    assert filtered_probes[0].groupname == 'Group1'


def test_filter_probes_by_position(probe_container):
    filtered_probes = probe_container.filter_probes(by_position=np.array([0, 0, 0]))
    assert len(filtered_probes) == 2
    assert np.array_equal(filtered_probes[0].position, np.array([0, 0, 0]))
    assert np.array_equal(filtered_probes[1].position, np.array([0, 0, 0]))


def test_plot_probes(probe_container):
    with pytest.raises(AssertionError):
        probe_container.plot_probes()  # value not specified

    # Test plotting with value='value1'
    probe_container.plot_probes(value='value1')

    # Test plotting with value='value2'
    probe_container.plot_probes(value='value2')

    # Add more specific tests based on your requirements
