from __future__ import annotations

import argparse
import sys
from pathlib import Path


def collect_files(directory: Path, recursive: bool, extensions: list[str] | None) -> list[Path]:
    """Return the files to rename, skipping directories and hidden entries."""
    entries = directory.rglob("*") if recursive else directory.iterdir()

    files = []
    for path in entries:
        if not path.is_file() or path.name.startswith("."):
            continue
        if extensions and path.suffix.lower() not in extensions:
            continue
        files.append(path)

    # Sort for a deterministic, reviewable dry-run listing.
    return sorted(files)


def plan_renames(files: list[Path], prefix: str, skip_prefixed: bool) -> list[tuple[Path, Path]]:
    """Build the (source, destination) pairs, refusing anything unsafe."""
    plan: list[tuple[Path, Path]] = []
    targets: set[Path] = set()

    for src in files:
        if skip_prefixed and src.name.startswith(prefix):
            print(f"[skip] already prefixed: {src.name}")
            continue

        dst = src.with_name(prefix + src.name)

        # An existing destination would be silently overwritten by Path.rename
        # on POSIX, so refuse instead.
        if dst.exists():
            print(f"[skip] destination exists: {dst.name}")
            continue

        # Two different sources cannot map onto the same destination.
        if dst in targets:
            print(f"[skip] duplicate destination: {dst.name}")
            continue

        targets.add(dst)
        plan.append((src, dst))

    return plan


def main() -> None:
    p = argparse.ArgumentParser(description="Add a prefix to every file name in a directory")
    p.add_argument("directory", type=Path, help="Directory to process")
    p.add_argument("prefix", type=str, help="String to prepend to each file name")
    p.add_argument("--recursive", action="store_true", help="Descend into subdirectories")
    p.add_argument("--ext", nargs="+", default=None, metavar="EXT",
                   help="Only rename these extensions, e.g. --ext .bmp .png")
    p.add_argument("--dry-run", action="store_true", help="Print the plan without renaming")
    p.add_argument("--force", action="store_true",
                   help="Also rename files that already start with the prefix")
    args = p.parse_args()

    if not args.directory.is_dir():
        sys.exit(f"Not a directory: {args.directory}")
    if not args.prefix:
        sys.exit("Prefix must not be empty.")

    # Path.with_name rejects separators, but fail with a clear message instead.
    if any(sep in args.prefix for sep in ("/", "\\")):
        sys.exit("Prefix must not contain a path separator.")

    extensions = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext] if args.ext else None

    files = collect_files(args.directory, args.recursive, extensions)
    if not files:
        sys.exit(f"No matching files found in {args.directory}")

    plan = plan_renames(files, args.prefix, skip_prefixed=not args.force)
    if not plan:
        sys.exit("Nothing to rename.")

    if args.dry_run:
        for src, dst in plan:
            print(f"{src.name}  ->  {dst.name}")
        print(f"\n{len(plan)} file(s) would be renamed. Re-run without --dry-run to apply.")
        return

    renamed = 0
    for src, dst in plan:
        try:
            src.rename(dst)
            renamed += 1
        except OSError as exc:
            print(f"[error] {src.name}: {exc}")

    print(f"Renamed {renamed}/{len(plan)} file(s).")


if __name__ == "__main__":
    main()