#!/usr/bin/env python3
"""Validate and package the static project page; no third-party dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def local_file(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Missing or unsafe local resource: {relative}")
    return path


def parts_manifest(relative):
    manifest_path = ROOT / (relative + ".parts.json")
    if not manifest_path.is_file():
        return None
    manifest = json.loads(local_file(relative + ".parts.json").read_text())
    if manifest["target"] != relative or not manifest["parts"]:
        raise ValueError(f"Invalid split resource: {relative}")
    digest = hashlib.sha256()
    size = 0
    for part in manifest["parts"]:
        with local_file(part).open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    if size != manifest["size"] or digest.hexdigest() != manifest["sha256"]:
        raise ValueError(f"Split resource checksum mismatch: {relative}")
    return manifest


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = set()
        self.ids = set()
        self.anchors = set()
        self.object_count = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            if attrs["id"] in self.ids:
                raise ValueError(f"Duplicate HTML id: {attrs['id']}")
            self.ids.add(attrs["id"])
        if "error-object" in attrs.get("class", "").split():
            self.object_count += 1
        for key in ("href", "src", "data-gif", "data-poster"):
            value = attrs.get(key, "")
            url = urlsplit(value)
            if url.scheme or url.netloc or not value:
                continue
            if value.startswith("#"):
                self.anchors.add(value[1:])
            elif url.path:
                self.references.add(unquote(url.path))


def source_files():
    page = PageParser()
    page.feed((ROOT / "index.html").read_text())
    if page.object_count != 9:
        raise ValueError(f"Expected nine object visualizations, got {page.object_count}")
    missing_anchors = page.anchors - page.ids
    if missing_anchors:
        raise ValueError(f"Broken section links: {sorted(missing_anchors)}")
    paths = {"index.html", ".nojekyll", "assets/web/visualizations.json"} | page.references
    for relative in sorted(paths):
        if not (ROOT / relative).resolve().is_relative_to(ROOT):
            raise ValueError(f"Unsafe resource: {relative}")
        if parts_manifest(relative) is None:
            local_file(relative)
    rows = json.loads((ROOT / "assets/web/results.json").read_text())
    if len(rows) != 9 or any(len(row) != 13 for row in rows):
        raise ValueError("Results must contain nine objects and twelve metrics per object")
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Destination for the deployable static site")
    parser.add_argument("--manifest", type=Path, help="Write the exact list of required source files")
    args = parser.parse_args()
    files = source_files()
    if args.manifest:
        sources = []
        for relative in files:
            manifest = parts_manifest(relative)
            sources.extend([relative + ".parts.json", *manifest["parts"]] if manifest else [relative])
        args.manifest.write_text(json.dumps(sources, indent=2) + "\n")
    if args.output:
        output = args.output.resolve()
        if output == ROOT or output in ROOT.parents:
            raise ValueError("Output must not overwrite the source directory or an ancestor")
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Output is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        for relative in files:
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            manifest = parts_manifest(relative)
            if manifest:
                with destination.open("wb") as target:
                    for part in manifest["parts"]:
                        with local_file(part).open("rb") as source:
                            shutil.copyfileobj(source, target)
            else:
                shutil.copy2(ROOT / relative, destination)
    total = sum(manifest["size"] if (manifest := parts_manifest(name)) else local_file(name).stat().st_size for name in files)
    print(f"Validated {len(files)} public files ({total / 1_000_000:.1f} MB), nine objects and all local links.")
    if args.output:
        print(f"GitHub Pages artifact: {args.output.resolve()}")


if __name__ == "__main__":
    main()
