from __future__ import annotations

import argparse
from pathlib import Path


def _write_manifest(root: Path, name: str, pattern: str) -> Path:
    files = sorted(path for path in root.glob(pattern) if path.is_file())
    manifest = root / f"{name.upper()}_MANIFEST.txt"
    manifest.write_text(
        f"{name} cache prepared by GitHub Actions\n"
        f"Root: {root}\n"
        f"Files: {len(files)}\n"
        f"Bytes: {sum(path.stat().st_size for path in files)}\n\n"
        + "\n".join(f"{path.stat().st_size} {path.relative_to(root)}" for path in files)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def ensure_cifar10(root: Path) -> Path:
    from torchvision import datasets

    for train in (True, False):
        dataset = datasets.CIFAR10(root=root, train=train, download=True)
        split = "train" if train else "test"
        print(f"CIFAR-10 {split} cache ready: {len(dataset)} examples")
    return _write_manifest(root, "cifar10", "cifar-10-batches-py/*")


def ensure_stl10(root: Path) -> Path:
    from torchvision import datasets

    for split in ("train+unlabeled", "test"):
        dataset = datasets.STL10(root=root, split=split, download=True)
        print(f"STL-10 {split} cache ready: {len(dataset)} examples")
    return _write_manifest(root, "stl10", "stl10_binary/*")


def ensure_celeba64(root: Path, allow_download: bool) -> Path:
    from torchvision import datasets

    def check_cache() -> bool:
        ok = True
        for split in ("train", "valid"):
            try:
                dataset = datasets.CelebA(root=root, split=split, target_type="attr", download=False)
                print(f"CelebA {split} cache hit: {len(dataset)} examples")
            except RuntimeError as exc:
                print(f"CelebA {split} cache miss: {exc}")
                ok = False
        return ok

    if not check_cache():
        if not allow_download:
            raise RuntimeError(
                "CelebA cache is missing and downloads are disabled. "
                "Stage the dataset in DATA_DIR or rerun with downloads enabled."
            )
        for split in ("train", "valid"):
            datasets.CelebA(root=root, split=split, target_type="attr", download=True)
        if not check_cache():
            raise RuntimeError(f"CelebA cache was not valid after download at {root}")

    return _write_manifest(root, "celeba64", "celeba/**/*")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensure a torchvision dataset cache exists.")
    parser.add_argument("--dataset", required=True, choices=("cifar10", "stl10", "celeba64"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    root = args.data_root.expanduser()
    root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "cifar10":
        manifest = ensure_cifar10(root)
    elif args.dataset == "stl10":
        manifest = ensure_stl10(root)
    else:
        manifest = ensure_celeba64(root, allow_download=args.allow_download)

    print(f"{args.dataset} cache ready at {root}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
