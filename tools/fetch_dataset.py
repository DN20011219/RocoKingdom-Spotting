"""从 Gitee 发行版拉取数据集到 datasets/yolo_dataset/。

数据集不随仓库分发：
  https://gitee.com/dinning/RocoKingdom-Spotting-dataset/releases
下载后先比对 SHA-256，再解压到临时目录、整体移动到目标位置，避免留下半套数据。
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

RELEASES = "https://gitee.com/dinning/RocoKingdom-Spotting-dataset/releases/download/assets-v1"
DATASET_URL = f"{RELEASES}/RocoKingdom-Spotting-yolo-dataset.zip"
SHA256_URL = f"{RELEASES}/RocoKingdom-Spotting-yolo-dataset.zip.sha256"

# 已知版本的校验值，取不到 sha256 文件时兜底
FALLBACK_SHA256 = "1261aaa3bd0e728128c83401f459456ba125d7861740dc685a06ba1197fdc294"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = PROJECT_ROOT / "datasets" / "yolo_dataset"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_with_retry(target, timeout: int, tries: int = 3):
    """国内网络偶发连不上发行版站点，失败后退避重试几次。"""
    last_error: Exception | None = None
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(target, timeout=timeout)
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
            if attempt + 1 < tries:
                time.sleep(2 * (attempt + 1))
    raise last_error  # type: ignore[misc]


def expected_sha256() -> str:
    try:
        with open_with_retry(SHA256_URL, timeout=10) as resp:
            text = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError):
        return FALLBACK_SHA256
    match = re.search(r"\b[0-9a-f]{64}\b", text)
    return match.group(0) if match else FALLBACK_SHA256


def download(url: str, dest: Path) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "RocoKingdom-Spotting-fetch-dataset"}
    )
    started = time.monotonic()
    with open_with_retry(request, timeout=30) as resp, dest.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        received = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
            received += len(chunk)
            if total:
                pct = received * 100 // total
                speed = received / max(time.monotonic() - started, 1e-6) / (1 << 20)
                print(
                    f"\r  {pct:3d}%  {received / (1 << 20):5.1f}"
                    f"/{total / (1 << 20):.1f} MB  {speed:4.1f} MB/s",
                    end="",
                    flush=True,
                )
    print()


def extract_archive(archive: Path, workdir: Path) -> Path:
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(workdir)
    entries = [p for p in workdir.iterdir() if p.is_dir()]
    if len(entries) != 1:
        raise ValueError(f"压缩包结构异常，期望单个顶层目录，实际 {len(entries)} 个")
    return entries[0]


def has_dataset_data(dest: Path) -> bool:
    """只有 classes.txt 的空目录不算已有数据 —— 那是仓库里带过来的注册表。"""
    for sub in ("images", "labels"):
        root = dest / sub
        if root.is_dir() and any(p.is_file() for p in root.rglob("*")):
            return True
    return (dest / "train.txt").is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 YOLO 数据集到 datasets/yolo_dataset")
    parser.add_argument("--url", default=DATASET_URL, help="数据集压缩包地址")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="解压目标目录")
    parser.add_argument("--force", action="store_true", help="目标目录已有数据时备份后重新拉取")
    args = parser.parse_args()

    dest: Path = args.dest.resolve()
    if has_dataset_data(dest):
        if not args.force:
            print(f"目标目录已有数据：{dest}", file=sys.stderr)
            print("确认要重新拉取请加 --force（原目录会改名保留为 .bak-时间戳）", file=sys.stderr)
            return 1
        backup = dest.with_name(f"{dest.name}.bak-{datetime.now():%Y%m%d-%H%M%S}")
        dest.rename(backup)
        print(f"已备份原目录：{backup}")

    # 仓库里跟踪的 classes.txt 是 class id 注册表，优先级高于压缩包里的同名文件
    local_classes = dest / "classes.txt"
    keep_classes = local_classes.read_bytes() if local_classes.is_file() else None

    expected = expected_sha256()
    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".fetch-dataset-") as tmp:
        tmpdir = Path(tmp)
        archive = tmpdir / "dataset.zip"

        print(f"下载 {args.url}")
        try:
            download(args.url, archive)
        except (urllib.error.URLError, OSError) as exc:
            print(f"下载失败：{exc}", file=sys.stderr)
            return 1

        actual = sha256_of(archive)
        if actual != expected:
            print(
                f"SHA-256 校验失败\n  期望 {expected}\n  实际 {actual}\n"
                "可能是发版更新了数据集但校验文件没同步",
                file=sys.stderr,
            )
            return 1

        print("解压 ...")
        try:
            top = extract_archive(archive, tmpdir / "unpacked")
        except (zipfile.BadZipFile, ValueError) as exc:
            print(f"解压失败：{exc}", file=sys.stderr)
            return 1

        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(top), str(dest))
        if keep_classes is not None:
            (dest / "classes.txt").write_bytes(keep_classes)

    images = sum(1 for _ in (dest / "images").rglob("*") if _.is_file())
    labels = sum(1 for _ in (dest / "labels").rglob("*.txt"))
    print(f"完成：{dest}")
    print(f"  图片 {images} 张，标注 {labels} 个")
    print("  下一步：python run.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())