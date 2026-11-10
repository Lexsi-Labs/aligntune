from aligntune.data.classification import ClassificationLabelEncoder


def encode(values):
    encoder = ClassificationLabelEncoder().fit(values)
    return [encoder.encode(v) for v in values]


def test_text_labels_get_sorted_ids():
    assert encode(["positive", "negative", "positive"]) == [1, 0, 1]
    assert encode(["0", "1", "1"]) == [0, 1, 1]


def test_one_indexed_integers_shift_the_same_in_every_batch():
    encoder = ClassificationLabelEncoder().fit([1, 2, 3, 1])
    assert [encoder.encode(v) for v in [1, 2]] == [0, 1]
    assert [encoder.encode(v) for v in [3, 1]] == [2, 0]
    assert encode([0, 1, 2]) == [0, 1, 2]


def test_trainers_no_longer_shift_per_batch():
    import inspect

    from aligntune.backends.trl.sft import Classification_trainer, sft_classification

    for module in (Classification_trainer, sft_classification):
        assert "min(labels)" not in inspect.getsource(module)
