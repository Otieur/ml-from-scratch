"""Loaders for the data sets used in the experiments.

- Graduate admissions (400 applicants, 7 features): ships with the package.
- BBC news (2,225 articles, 9,635 terms) and RAVDESS speech (1,440 clips): downloaded into
  data/ by scripts/fetch_data.py.
"""
import glob
import os
from pathlib import Path

import numpy as np

PACKAGE_DATA = Path(__file__).resolve().parent / "data"
DOWNLOADS = Path(__file__).resolve().parent.parent / "data"
BBC_DIR = DOWNLOADS / "bbc"
SPEECH_DIR = DOWNLOADS / "Audio_Speech_Actors_01-24"
SPEECH_CACHE = DOWNLOADS / "speech_features.npz"


# ------------------------------------------------------------------ graduate admissions

def load_admissions(path=PACKAGE_DATA / "admissions.csv"):
    """Chance of admission to a graduate program and seven applicant features.

    Output:
    - X (ndarray (Shape: (400, 7))), y (ndarray (Shape: (400,))), names (list of 7 str).
    """
    import pandas as pd

    table = pd.read_csv(path)
    # Two headers in the file carry a trailing space ("LOR " and "Chance of Admit ").
    table.columns = [name.strip() for name in table.columns]
    target = "Chance of Admit"
    names = [name for name in table.columns if name not in ("Serial No.", target)]
    return table[names].to_numpy(dtype=np.float64), table[target].to_numpy(dtype=np.float64), names


# ------------------------------------------------------------------ BBC news

def load_bbc(data_dir=BBC_DIR):
    """The pre-processed BBC news data set.

    Files:
    - bbc.mtx      term-document matrix in MatrixMarket coordinate format. After the two header
                   lines, each line "t d f" says that term t occurs f times in document d;
                   both indices start at 1.
    - bbc.terms    one stemmed term per line; line t is term t.
    - bbc.docs     one document name per line, e.g. "business.001".
    - bbc.classes  after four header lines, each line "d c" gives the class c of document d;
                   both indices start at 0. The header names the classes in order.

    Output:
    - counts (ndarray (Shape: (N, D), dtype float32)): counts[d, t] is the frequency of term t
      in document d.
    - labels (ndarray (Shape: (N,))): class index of each document.
    - class_names (list of str), terms (list of D str), docs (list of N str).
    """
    data_dir = Path(data_dir)
    terms = (data_dir / "bbc.terms").read_text().split()
    docs = (data_dir / "bbc.docs").read_text().split()

    with open(data_dir / "bbc.mtx") as handle:
        header = handle.readline()
        assert header.startswith("%%MatrixMarket matrix coordinate"), f"Unexpected header: {header!r}"
        n_terms, n_docs, n_entries = (int(value) for value in handle.readline().split())
        entries = np.loadtxt(handle)
    assert entries.shape == (n_entries, 3)
    assert (n_terms, n_docs) == (len(terms), len(docs))
    counts = np.zeros((n_docs, n_terms), dtype=np.float32)
    # The file is indexed (term, document) from 1; the matrix is indexed [document, term] from 0.
    counts[entries[:, 1].astype(int) - 1, entries[:, 0].astype(int) - 1] = entries[:, 2]

    class_names = None
    labels = np.full(n_docs, -1)
    for line in (data_dir / "bbc.classes").read_text().splitlines():
        if line.startswith("%Clusters"):
            class_names = line.split()[2].split(",")
        elif line and not line.startswith("%"):
            doc, label = (int(value) for value in line.split())
            labels[doc] = label
    assert class_names is not None and (labels >= 0).all()
    # Cross-check the mapping between the two files: every document name starts with its class name.
    for name, label in zip(docs, labels):
        assert name.split(".")[0] == class_names[label], f"{name} is labelled {class_names[label]}"
    return counts, labels, class_names, terms, docs


# ------------------------------------------------------------------ RAVDESS speech

# Third field of a RAVDESS file name, e.g. 03-01-06-01-02-02-05.wav is "fearful", actor 5.
EMOTIONS = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
            "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
POSITIVE, NEGATIVE = ("calm", "happy"), ("angry", "fearful")
N_CHROMA, N_MELS = 12, 128


def clip_features(path):
    """Describe one audio clip by 140 numbers.

    Output:
    - ndarray (Shape: (140,)): 12 chroma values (energy per pitch class) followed by 128 Mel
      band energies, each averaged over the duration of the clip.
    - stereo (bool): whether the file had two channels (they are averaged into one).
    """
    import librosa
    import soundfile

    waveform, sample_rate = soundfile.read(path, dtype="float32")
    stereo = waveform.ndim == 2
    if stereo:
        waveform = waveform.mean(axis=1)
    mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=N_MELS, fmax=sample_rate / 2.0)
    chroma = librosa.feature.chroma_stft(S=np.abs(librosa.stft(waveform)), sr=sample_rate)
    return np.concatenate((chroma.mean(axis=1), mel.mean(axis=1))), stereo


def load_speech_emotion(audio_dir=SPEECH_DIR, cache_path=SPEECH_CACHE):
    """Features and labels for the calm, happy, angry and fearful clips of RAVDESS.

    Output:
    - X (ndarray (Shape: (768, 140))): one row per clip, see clip_features.
    - y (ndarray (Shape: (768,))): 0 for a positive emotion (calm, happy), 1 for a negative
      one (angry, fearful).
    - info (dict of arrays): "actor" (1 to 24), "emotion", "filename", "stereo".

    The files are read in sorted order so that the rows, and every later random split, are
    the same on every machine. Extracting the features takes about 15 seconds; the result is
    cached in data/.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["X"], cached["y"], {key: cached[key] for key in ("actor", "emotion", "filename", "stereo")}

    rows, labels = [], []
    info = {"actor": [], "emotion": [], "filename": [], "stereo": []}
    for path in sorted(glob.glob(str(audio_dir) + "/Actor_*/*.wav")):
        name = os.path.basename(path)
        fields = name[:-len(".wav")].split("-")
        emotion = EMOTIONS[fields[2]]
        if emotion not in POSITIVE + NEGATIVE:
            continue
        features, stereo = clip_features(path)
        rows.append(features)
        labels.append(0 if emotion in POSITIVE else 1)
        info["actor"].append(int(fields[6]))
        info["emotion"].append(emotion)
        info["filename"].append(name)
        info["stereo"].append(stereo)
    assert rows, f"No audio found under {audio_dir}. Run scripts/fetch_data.py first."
    X, y = np.array(rows), np.array(labels)
    info = {key: np.array(value) for key, value in info.items()}
    np.savez_compressed(cache_path, X=X, y=y, **info)
    return X, y, info


def log_mel(X):
    """The same features with the Mel band energies replaced by their logarithm.

    The band energies span about ten orders of magnitude, so a Euclidean distance on them
    is decided by the few loudest bands; the logarithm puts the bands on a comparable scale.
    """
    Z = np.array(X, dtype=np.float64)
    Z[:, N_CHROMA:] = np.log(Z[:, N_CHROMA:] + 1e-10)
    return Z
