# Simple helper to grab a ~3 MB tarball hosted on GitHub Releases
import urllib.request, tarfile, pathlib, shutil, sys

URL = "https://github.com/Manas2006/smoke-sentinel/releases/download/v0.1/sample_data.tar.gz"
DEST = pathlib.Path("data/raw_sample")

DEST.mkdir(parents=True, exist_ok=True)
tgz_path = DEST / "sample_data.tar.gz"
print("Downloading sample…")
urllib.request.urlretrieve(URL, tgz_path)
with tarfile.open(tgz_path, "r:gz") as tf:
    tf.extractall(DEST)
tgz_path.unlink()
print("Done! Extracted to", DEST) 