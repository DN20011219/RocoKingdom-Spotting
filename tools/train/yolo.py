#!/usr/bin/env python3
"""YOLO 模型训练工具 — 基于 ultralytics，支持自定义数据集训练。

用法:
    cd RocoKingdom-Spotting
    python tools/train/yolo.py --data datasets/pets.yaml --model yolo11n.pt
    python tools/train/yolo.py --data datasets/pets.yaml --epochs 200 --batch 32
    python tools/train/yolo.py --resume  # 恢复中断的训练
"""

import sys
from pathlib import Path

# 路径自举：tools/train/ -> tools/ -> 项目根目录
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# PowerShell 对 \r 回车覆盖支持不佳，导致 tqdm 进度条不断追加新行。
# 检测到 PowerShell 时自动切换到 cmd.exe 重新执行。
if sys.platform == "win32" and "POWERSHELL" in __import__("os").environ and not __import__("os").environ.get("_SPOTTING_CMD_SHELL"):
    import subprocess
    import shutil
    import os
    python_exe = shutil.which("python") or sys.executable
    cmd = [python_exe] + sys.argv
    env = os.environ.copy()
    env["_SPOTTING_CMD_SHELL"] = "1"  # 标记已切换，防止递归
    sys.exit(subprocess.call(cmd, shell=True, env=env))  # shell=True 在 Windows 上使用 cmd.exe

import argparse
import time


def _resolve_device(device: str) -> str:
    """解析 device 参数。'auto' 时自动检测 GPU 可用性。"""
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="YOLO 模型训练")
    parser.add_argument("--data", type=str, required=False, default=None,
                        help="数据集配置文件路径 (YOLO 格式 yaml)")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="基础模型/预训练权重 (默认: yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数 (默认: 100)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="训练图片尺寸 (默认: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="batch size (默认: 16)")
    parser.add_argument("--device", default="auto",
                        help="设备: auto / cuda:0 / cpu (默认: auto)")
    parser.add_argument("--name", type=str, default="train_run",
                        help="实验名称 (默认: train_run)")
    parser.add_argument("--resume", action="store_true",
                        help="恢复中断的训练")
    parser.add_argument("--workers", type=int, default=8,
                        help="数据加载线程数 (默认: 8)")
    parser.add_argument("--patience", type=int, default=50,
                        help="早停耐心值 (默认: 50)")
    args = parser.parse_args()

    # 检查 ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] 未安装 ultralytics，请运行: pip install ultralytics")
        sys.exit(1)

    # 解析设备
    resolved_device = _resolve_device(args.device)

    # 恢复训练
    if args.resume:
        print("[模式] 恢复训练")
        # 查找最新的训练结果
        runs_dir = Path(__file__).resolve().parent.parent.parent / "runs" / "detect"
        if not runs_dir.exists():
            print("[错误] 未找到训练结果目录，无法恢复")
            sys.exit(1)

        # 查找最新的实验
        experiments = sorted(runs_dir.glob(f"{args.name}*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not experiments:
            print(f"[错误] 未找到实验 '{args.name}' 的训练结果")
            sys.exit(1)

        latest_exp = experiments[0]
        last_model = latest_exp / "weights" / "last.pt"
        if not last_model.exists():
            print(f"[错误] 未找到 last.pt: {last_model}")
            sys.exit(1)

        print(f"    恢复自: {last_model}")
        model = YOLO(str(last_model))
        results = model.train(
            resume=True,
            device=resolved_device,
        )
        print(f"\n[完成] 训练已恢复")
        return

    # 正常训练
    if not args.data:
        print("[错误] 必须指定 --data 数据集配置文件")
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent.parent.parent / data_path

    if not data_path.exists():
        print(f"[错误] 数据集配置文件不存在: {data_path}")
        sys.exit(1)

    # 解析模型路径
    model_path = Path(args.model)
    if not model_path.is_absolute():
        # 先检查是否在 models/ 下
        project_models = Path(__file__).resolve().parent.parent.parent / "models" / model_path
        if project_models.exists():
            model_path = project_models
        # 否则使用 ultralytics 默认（会自动下载）

    print("=" * 60)
    print("YOLO 模型训练")
    print("=" * 60)
    print(f"    数据集: {data_path}")
    print(f"    模型: {model_path}")
    print(f"    设备: {resolved_device}")
    print(f"    轮数: {args.epochs}")
    print(f"    图片尺寸: {args.imgsz}")
    print(f"    Batch size: {args.batch}")
    print(f"    实验名称: {args.name}")
    print(f"    早停耐心: {args.patience}")
    print(f"    线程数: {args.workers}")
    print("=" * 60)

    # 加载模型
    print("\n[1] 加载模型...")
    model = YOLO(str(model_path))

    # 开始训练
    print("[2] 开始训练...")
    start_time = time.time()

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=resolved_device,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        exist_ok=True,
        verbose=True,
    )

    elapsed = time.time() - start_time
    print(f"\n[3] 训练完成!")
    print(f"    总耗时: {elapsed / 60:.1f} 分钟")

    # 输出结果
    if results and hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\n[指标]")
        for key, value in metrics.items():
            print(f"    {key}: {value:.4f}")

    # 输出模型保存路径
    save_dir = Path(__file__).resolve().parent.parent.parent / "runs" / "detect" / args.name
    best_model = save_dir / "weights" / "best.pt"
    last_model = save_dir / "weights" / "last.pt"

    print(f"\n[模型]")
    if best_model.exists():
        print(f"    最佳模型: {best_model}")
    if last_model.exists():
        print(f"    最新模型: {last_model}")

    print(f"\n[提示] 可将最佳模型复制到 models/ 目录供检测使用:")
    print(f"    cp {best_model} models/your_model_name.pt")


if __name__ == "__main__":
    main()
