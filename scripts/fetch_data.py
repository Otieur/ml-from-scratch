"""Download the three public data sets used by the experiments into data/ (not committed).

Usage (from the repository root):
    python scripts/fetch_data.py            # all three
    python scripts/fetch_data.py bbc        # one of: bbc, ravdess, movielens

Sources
- BBC news, pre-processed term-document matrix (0.9 MB):
  http://mlg.ucd.ie/datasets/bbc.html
  D. Greene and P. Cunningham, "Practical Solutions to the Problem of Diagonal Dominance
  in Kernel Document Clustering", ICML 2006. Free for non-commercial and research use.
- RAVDESS, speech audio only (208 MB):
  https://zenodo.org/records/1188976
  S. R. Livingstone and F. A. Russo, "The Ryerson Audio-Visual Database of Emotional
  Speech and Song (RAVDESS)", PLoS ONE 13(5), 2018. CC BY-NC-SA 4.0.
- MovieLens 100K, 100,000 film ratings (4.9 MB):
  https://grouplens.org/datasets/movielens/100k/
  F. M. Harper and J. A. Konstan, "The MovieLens Datasets: History and Context", ACM
  Transactions on Interactive Intelligent Systems 5(4), 2015. Research use; not to be
  redistributed.
"""
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

SOURCES = {
    "bbc": {
        "url": "http://mlg.ucd.ie/files/datasets/bbc.zip",
        "archive": "bbc.zip",
        "folder": "bbc",
        "sha256": "976a091f7579f65c20b2738829e63a6cbae988ca7af1d7a4dfadefcefd460ca6",
    },
    "ravdess": {
        "url": "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip",
        "archive": "Audio_Speech_Actors_01-24.zip",
        "folder": "Audio_Speech_Actors_01-24",
        "sha256": "5d208e01632cc3e5242106fa2af3273e6dc5239fb8143131979ac74c4aa40657",
    },
    "movielens": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "archive": "ml-100k.zip",
        "folder": "movielens",
        "sha256": "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229",
    },
}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name):
    source = SOURCES[name]
    DATA.mkdir(exist_ok=True)
    archive = DATA / source["archive"]
    folder = DATA / source["folder"]
    if folder.exists() and any(folder.iterdir()):
        print(f"{name}: already present in {folder}")
        return
    if not archive.exists():
        print(f"{name}: downloading {source['url']}")
        urllib.request.urlretrieve(source["url"], archive)
    digest = sha256_of(archive)
    print(f"{name}: {archive.name}, {archive.stat().st_size} bytes, sha256 {digest}")
    if source["sha256"] is not None and digest != source["sha256"]:
        raise SystemExit(f"{name}: checksum mismatch, expected {source['sha256']}")
    folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(folder)
    print(f"{name}: extracted to {folder}")


if __name__ == "__main__":
    for name in (sys.argv[1:] or list(SOURCES)):
        fetch(name)
