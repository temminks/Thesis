import pytest

from thesis.Run import Runner


def test_runner_init():
    """Tests for correct parameter ranges."""
    with pytest.raises(ValueError):
        Runner(model=None, episodes=-1)

    with pytest.raises(ValueError):
        Runner(model=None, gamma=-1)

    with pytest.raises(ValueError):
        Runner(model=None, gamma=1.1)

    assert Runner(model=None, gamma=0.5) is not False
