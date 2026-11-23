"""T4: architectures and parameter counts (12 classes)."""
import pytest

from lasdaicvi_kws import models


@pytest.mark.parametrize("arch, params", [
    ("cnn", 2_769_420), ("gru", 593_836), ("crnn", 467_772),
])
def test_param_counts(arch, params):
    m = models.build_model(arch, n_classes=12)
    assert m.count_params() == params
    assert m.output_shape == (None, 12)


def test_layer_names_follow_the_original_models():
    names = [l.name for l in models.build_model("crnn", n_classes=3).layers]
    assert names == ["input", "conv_1", "conv_2", "reshape", "gru", "dropout",
                     "dense_linear", "dense", "output"]
    cnn = [l.name for l in models.build_model("cnn", n_classes=3).layers]
    assert cnn[1:8] == [f"conv_{i}" for i in range(1, 8)]
    assert [l.name for l in models.build_model("gru", n_classes=3).layers][:3] == [
        "input", "reshape", "gru"]


def test_unknown_arch_raises():
    with pytest.raises(KeyError):
        models.build_model("lstm", 12)
