"""Checks on the two downloaded data sets. Skipped when data/ has not been fetched."""
import numpy as np
import pytest

from mlscratch.datasets import BBC_DIR, SPEECH_CACHE, SPEECH_DIR, load_bbc, load_speech_emotion, log_mel

needs_bbc = pytest.mark.skipif(not (BBC_DIR / "bbc.mtx").exists(), reason="run scripts/fetch_data.py bbc")
needs_speech = pytest.mark.skipif(not (SPEECH_CACHE.exists() or SPEECH_DIR.exists()),
                                  reason="run scripts/fetch_data.py ravdess")


@pytest.fixture(scope="module")
def bbc():
    return load_bbc()


@needs_bbc
def test_bbc_sizes(bbc):
    counts, labels, class_names, terms, docs = bbc
    assert counts.shape == (2225, 9635)
    assert len(terms) == 9635 and len(docs) == 2225
    assert class_names == ["business", "entertainment", "politics", "sport", "tech"]
    assert np.bincount(labels).tolist() == [510, 386, 417, 511, 401]
    assert int((counts > 0).sum()) == 286774      # the number of entries announced in bbc.mtx


@needs_bbc
def test_bbc_matrix_is_read_with_the_right_orientation(bbc):
    # Line 812 of bbc.mtx is "2 528 5.0": term 2 ("sale") occurs 5 times in document 528
    # ("entertainment.018"). The file counts from 1 and lists the term first.
    counts, labels, class_names, terms, docs = bbc
    with open(BBC_DIR / "bbc.mtx") as handle:
        assert handle.readlines()[812 - 1].split() == ["2", "528", "5.0"]
    assert terms[2 - 1] == "sale"
    assert docs[528 - 1] == "entertainment.018"
    assert counts[528 - 1, 2 - 1] == 5.0
    assert class_names[labels[528 - 1]] == "entertainment"


@needs_bbc
def test_both_classifiers_reach_the_recorded_accuracy_on_bbc(bbc):
    from mlscratch.gaussian_classifier import GaussianClassConditional
    from mlscratch.model_selection import stratified_split
    from mlscratch.naive_bayes import BernoulliNaiveBayes
    from sklearn.naive_bayes import BernoulliNB

    counts, labels, *_ = bbc
    binary = (counts > 0).astype(np.float32)
    train, test = stratified_split(labels, test_size=0.3, seed=0)
    assert (len(train), len(test)) == (1558, 667)

    naive = BernoulliNaiveBayes().fit(binary[train], labels[train])
    reference = BernoulliNB(alpha=1.0, binarize=None).fit(binary[train], labels[train])
    np.testing.assert_array_equal(naive.predict(binary[test]), reference.predict(binary[test]))
    assert naive.score(binary[test], labels[test]) > 0.95

    gaussian = GaussianClassConditional(var_smoothing=3e-3).fit(counts[train], labels[train])
    assert gaussian.score(counts[test], labels[test]) > 0.94


@needs_speech
def test_speech_features_have_the_expected_layout():
    X, y, info = load_speech_emotion()
    # 24 actors x 4 emotions x 2 statements x 2 repetitions x 2 intensities = 768 clips.
    assert X.shape == (768, 140)
    assert np.bincount(y).tolist() == [384, 384]
    assert np.isfinite(X).all() and (X > 0).all()
    assert sorted(np.unique(info["actor"])) == list(range(1, 25))
    assert set(info["emotion"]) == {"calm", "happy", "angry", "fearful"}
    assert list(info["filename"]) == sorted(info["filename"], key=lambda name: (name.split("-")[6], name))
    # Stereo clips are mixed down, not dropped or zeroed.
    assert info["stereo"].sum() == 4


@needs_speech
def test_log_mel_changes_only_the_mel_bands():
    X, _, _ = load_speech_emotion()
    Z = log_mel(X)
    np.testing.assert_array_equal(Z[:, :12], X[:, :12])
    np.testing.assert_allclose(Z[:, 12:], np.log(X[:, 12:].astype(np.float64) + 1e-10))
    # The band energies span many orders of magnitude; their logarithms do not.
    assert X[:, 12:].max() / X[:, 12:].min() > 1e8
    assert Z[:, 12:].max() - Z[:, 12:].min() < 40
