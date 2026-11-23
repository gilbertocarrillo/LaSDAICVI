"""T6: experiments.yaml and CLI wiring."""
import pytest

from lasdaicvi_kws import config

COMMANDS = ["si", "no", "verdadero", "falso", "detente", "siguiente",
            "aprender", "repite", "jugar", "continúa"]


def test_ten_tasks_and_sixty_rows():
    assert sorted(config.load_tasks()) == sorted(
        [f"table{i}" for i in range(2, 10)] + ["interaction", "pepe"])
    assert len(config.experiment_rows()) == 60


def test_interaction_labels_are_the_ten_commands():
    t = config.get_task("interaction")
    assert t.labels == COMMANDS
    assert t.unknown == ["pepe"] + [str(d) for d in range(10)]


def test_pepe_unknown_lists_fixed_and_thesis():
    fixed = config.get_task("pepe", thesis=False).unknown
    thesis = config.get_task("pepe", thesis=True).unknown
    assert "7" in fixed and "8" in fixed and "78" not in fixed
    assert "78" in thesis and "7" not in thesis and "8" not in thesis
    assert len(thesis) == 19 and len(fixed) == 20


def test_other_tasks_have_a_single_unknown_list():
    assert config.get_task("table2", thesis=True).unknown == config.get_task("table2").unknown


def test_epochs_per_arch_and_mode():
    assert config.get_task("interaction").epochs("gru", aug=True) == 300
    assert config.get_task("table4").epochs("crnn", aug=False) == 400
    assert config.get_task("table9").epochs("gru", aug=False) == 400
    assert config.get_task("pepe").epochs("crnn", aug=False) == 400
    assert config.get_task("table4").epochs("crnn", aug=True) == 300


def test_experiment_names():
    names = {config.experiment_name(*row) for row in config.experiment_rows()}
    assert "interaction_gru_aug" in names and "table6_cnn_base" in names and len(names) == 60


def test_unknown_task_raises():
    with pytest.raises(KeyError):
        config.get_task("table1")
