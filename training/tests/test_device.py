import pytest

from lasdaicvi_kws.train import parse_args, resolve_device


@pytest.mark.parametrize("arch", ["cnn", "gru", "crnn"])
def test_auto_runs_every_model_on_cpu(arch):
    # tensorflow-metal mis-trains both the GRU and the CNN, so every model defaults to the CPU.
    assert resolve_device("auto", arch) == "/CPU:0"


@pytest.mark.parametrize("arch", ["cnn", "gru", "crnn"])
def test_explicit_choice_wins(arch):
    assert resolve_device("cpu", arch) == "/CPU:0"
    assert resolve_device("gpu", arch) == "/GPU:0"


def test_device_flag_defaults_to_auto():
    assert parse_args(["--task", "interaction", "--arch", "gru"]).device == "auto"
