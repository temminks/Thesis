import pytest

from thesis.Run import ForwardSarsaLambdaRunner


def test_runner_init():
    """Tests for correct parameter ranges."""
    with pytest.raises(ValueError):
        ForwardSarsaLambdaRunner(model=None, episodes=-1)

    with pytest.raises(ValueError):
        ForwardSarsaLambdaRunner(model=None, gamma=-1)

    with pytest.raises(ValueError):
        ForwardSarsaLambdaRunner(model=None, gamma=1.1)

    assert ForwardSarsaLambdaRunner(model=None, gamma=0.5) is not False
