"""从 Gitee 发行版拉取运行时模型到 models/best.pt。

模型与数据集不随仓库分发：
  https://gitee.com/dinning/RocoKingdom-Spotting-dataset/releases
GitHub Releases 保留同样的模型作为海外镜像，Gitee 失败时自动回退。
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

GITEE_BASE = "https://gitee.com/dinning/RocoKingdom-Spotting-dataset/releases/download/assets-v1"
GITHUB_BASE = "https://github.com/DN20011219/RocoKingdom-Spotting/releases/latest/download"

# 按顺序尝试的下载源：国内走 Gitee，海外或 Gitee 故障时回退 GitHub
MIRRORS = (
    (GITEE_BASE + "/best.pt", GITEE_BASE + "/best.pt.sha256"),
    (GITHUB_BASE + "/best.pt", GITHUB_BASE + "/best.pt.sha256"),
)

# 已知版本的校验值，两个源都取不到 sha256 文件时兜底
FALLBACK_SHA256 = "0a2762845a0f660caaf1236abfec21d12976d418726bf33cb8c36275a17dba68"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = PROJECT_ROOT / "models" / "best.pt"


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


def expected_sha256(sha256_urls) -> str:
    for url in sha256_urls:
        try:
            with open_with_retry(url, timeout=10) as resp:
                text = resp.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError):
            continue
        match = re.search(r"\b[0-9a-f]{64}\b", text)
        if match:
            return match.group(0)
    return FALLBACK_SHA256


def download(url: str, dest: Path, expected: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "RocoKingdom-Spotting-fetch-model"}
    )
    started = time.monotonic()
    try:
        with open_with_retry(request, timeout=30) as resp, tmp.open("wb") as out:
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
        if actual != expected:
            raise ValueError(
                f"SHA-256 校验失败\n  期望 {expected}\n  实际 {actual}\n"
                "可能是发版更新了模型但校验文件没同步"
            )
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 YOLO 模型到 models/best.pt")
    parser.add_argument("--url", help="指定模型地址，跳过镜像顺序")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="保存路径")
    parser.add_argument("--force", action="store_true", help="已存在且校验通过时也重新下载")
    args = parser.parse_args()

    sources = [args.url] if args.url else [url for url, _ in MIRRORS]
    sha256_urls = [sha for _, sha in MIRRORS]
    expected = expected_sha256(sha256_urls)

    if args.dest.exists() and not args.force:
        if sha256_of(args.dest) == expected:
            print(f"已存在且校验通过，跳过下载：{args.dest}")
            return 0
        print(f"本地文件校验不符，重新下载：{args.dest}")

    for model_url in sources:
        print(f"下载 {model_url}")
        try:
            download(model_url, args.dest, expected)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(f"  失败：{exc}", file=sys.stderr)
            continue
        print(f"完成：{args.dest}")
        return 0

    print("所有下载源都失败了，可手动下载后放进 models/", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())