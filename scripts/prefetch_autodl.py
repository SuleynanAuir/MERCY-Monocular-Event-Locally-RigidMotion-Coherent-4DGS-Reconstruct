"""Prefetch pinned packages with curl/HTTP2 when AutoDL throttles pip/HTTP1."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(requirement, destination, tag_order):
    try:
        if requirement.specifier.contains(metadata.version(requirement.name)):
            print(f"Already installed: {requirement}", flush=True)
            return
    except metadata.PackageNotFoundError:
        pass
    pins = list(requirement.specifier)
    if len(pins) != 1 or pins[0].operator != "==":
        raise ValueError(f"Expected an exact version pin: {requirement}")
    curl = ["curl", "--http2", "-fsSL", "--retry", "2", "--connect-timeout", "15", "--max-time", "300"]
    release = json.loads(subprocess.check_output(
        curl + [f"https://pypi.org/pypi/{requirement.name}/{pins[0].version}/json"]))
    candidates = []
    for asset in release["urls"]:
        if asset["packagetype"] == "bdist_wheel":
            tags = parse_wheel_filename(asset["filename"])[3]
            rank = min((tag_order[t] for t in tags if t in tag_order), default=None)
            if rank is not None:
                candidates.append((rank, asset))
        elif asset["packagetype"] == "sdist":
            candidates.append((len(tag_order), asset))
    if not candidates:
        raise ValueError(f"No compatible distribution for {requirement}")
    asset = min(candidates, key=lambda candidate: candidate[0])[1]
    filename = asset["filename"]
    if Path(filename).name != filename:
        raise ValueError("Unsafe distribution filename")
    target = destination / filename
    expected = asset["digests"]["sha256"]
    if target.exists():
        if sha256(target) != expected:
            raise ValueError(f"Existing package checksum mismatch: {target}")
        return
    partial = target.with_name(filename + ".part")
    # This mirror negotiates HTTP/2; the original CDN may fall back to HTTP/1.
    # The digest still comes from the official PyPI release metadata above.
    url = asset["url"].replace("https://files.pythonhosted.org/packages/",
                               "https://mirrors.aliyun.com/pypi/packages/")
    subprocess.run(curl + [url, "-o", str(partial)], check=True)
    if sha256(partial) != expected:
        raise ValueError(f"Downloaded package checksum mismatch: {partial}")
    partial.replace(target)
    print(f"Verified: {filename}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=Path("requirements-autodl.txt"))
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    requirements = [Requirement(line.strip()) for line in args.requirements.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
    tag_order = {tag: rank for rank, tag in enumerate(sys_tags())}
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda requirement: fetch(requirement, args.destination, tag_order), requirements))


if __name__ == "__main__":
    main()
