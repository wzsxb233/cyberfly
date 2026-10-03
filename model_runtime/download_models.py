"""Download the public official GGUF bundle with resume and LFS SHA256 checks.

Uses no Hugging Face client, credential store, token or cloud inference API.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(os.environ.get("CYBERFLY_MINICPM_ROOT", "/root/.cache/cyberfly/minicpm-runtime"))
DEST = ROOT / "models"
REPO = "openbmb/MiniCPM-o-4_5-gguf"
CHUNK = 8 * 1024 * 1024
SOURCE = os.environ.get("CYBERFLY_MODEL_SOURCE", "modelscope")


def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as source:
        return json.load(source)


def digest(path):
    check = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(32 * 1024 * 1024):
            check.update(block)
    return check.hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "models-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        revision = get_json(f"https://huggingface.co/api/models/{REPO}")["sha"]
        entries = get_json(f"https://huggingface.co/api/models/{REPO}/tree/{revision}?recursive=true&expand=false")
        files = [entry for entry in entries if entry["type"] == "file" and (
            entry["path"] == "MiniCPM-o-4_5-Q4_K_M.gguf" or
            entry["path"].endswith(".gguf") and "/" in entry["path"]
        )]
        manifest = {"repo": REPO, "revision": revision, "files": files}
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print("Pinned revision", manifest["revision"], "bundle bytes", sum(f["size"] for f in manifest["files"]), flush=True)
    for entry in manifest["files"]:
        target = DEST / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = entry["lfs"]["oid"]
        if target.exists() and target.stat().st_size == entry["size"] and digest(target) == expected:
            print("VERIFIED existing", entry["path"], flush=True)
            continue
        partial = target.with_suffix(".gguf.part")
        progress = target.with_suffix(".gguf.chunks.json")
        complete = set(json.loads(progress.read_text())) if progress.exists() and partial.exists() else set()
        fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(fd, entry["size"])
        total_chunks = (entry["size"] + CHUNK - 1) // CHUNK
        started = time.monotonic()
        last_log = started

        def fetch(index):
            start = index * CHUNK
            end = min(entry["size"], start + CHUNK) - 1
            # ModelScope is OpenBMB's public mirror. Its mutable master is always
            # checked against the immutable HF revision's LFS SHA256 below.
            if SOURCE == "modelscope":
                url = f"https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf/resolve/master/{entry['path']}?part={start}"
            else:
                url = f"https://huggingface.co/{REPO}/resolve/{manifest['revision']}/{entry['path']}?download=true&part={start}"
            failure = None
            for attempt in range(4):
                try:
                    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(req, timeout=60) as response:
                        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{entry['size']}":
                            raise RuntimeError("Unexpected HTTP range response")
                        data = response.read(end - start + 2)
                    if len(data) != end - start + 1:
                        raise RuntimeError("Short range response")
                    written = 0
                    while written < len(data):
                        written += os.pwrite(fd, data[written:], start + written)
                    return index
                except Exception as exc:
                    failure = exc
                    time.sleep(min(2**attempt, 8))
            raise RuntimeError(f"Failed chunk {index} for {entry['path']}: {failure}")

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                pending = [pool.submit(fetch, n) for n in range(total_chunks) if n not in complete]
                for future in concurrent.futures.as_completed(pending):
                    complete.add(future.result())
                    temp = progress.with_suffix(".tmp")
                    temp.write_text(json.dumps(sorted(complete)))
                    temp.replace(progress)
                    if time.monotonic() - last_log > 20:
                        print(entry["path"], f"{len(complete)}/{total_chunks} chunks", f"{time.monotonic()-started:.0f}s", flush=True)
                        last_log = time.monotonic()
            os.fsync(fd)
        finally:
            os.close(fd)
        actual = digest(partial)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {entry['path']}: {actual}")
        partial.replace(target)
        progress.unlink(missing_ok=True)
        print("VERIFIED", entry["path"], entry["size"], flush=True)
    print("ALL OFFICIAL GGUF FILES VERIFIED", flush=True)


if __name__ == "__main__":
    main()
