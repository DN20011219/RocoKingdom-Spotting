"""从项目站点拉取运行时模型到 models/best.pt。

模型与数据集不随仓库分发，下载地址与校验值见 README「数据集与预训练模型」。
重新训练并替换站点上的 best.pt 后，DEFAULT_SHA256 需要同步更新。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://dataset.dingning.tech/best.pt"
DEFAULT_SHA256 = "0a2762845a0f660caaf1236abfec21d12976d418726bf33cb8c36275a17dba68"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = PROJECT_ROOT / "models" / "best.pt"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, expected_sha256: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "RocoKingdom-Spotting-fetch-model"}
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=30) as resp, tmp.open("wb") as out:
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

        actual = sha256_of(tmp)
        if actual != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"SHA-256 校验失败\n  期望 {expected_sha256}\n  实际 {actual}\n"
                "站点上的模型可能已更新，请核对 README 里的校验值"
            )
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 YOLO 模型到 models/best.pt")
    parser.add_argument("--url", default=DEFAULT_URL, help="模型下载地址")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="保存路径")
    parser.add_argument("--force", action="store_true", help="已存在且校验通过时也重新下载")
    args = parser.parse_args()

    if args.dest.exists() and not args.force:
        if sha256_of(args.dest) == DEFAULT_SHA256:
            print(f"已存在且校验通过，跳过下载：{args.dest}")
            return 0
        print(f"本地文件校验不符，重新下载：{args.dest}")

    print(f"下载 {args.url}")
    try:
        download(args.url, args.dest, DEFAULT_SHA256)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"下载失败：{exc}", file=sys.stderr)
        return 1

    print(f"完成：{args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())