#!/usr/bin/env python3
"""YOLO 模型训练工具 — 基于 ultralytics，支持自定义数据集训练。

用法:
    cd RocoKingdom-Spotting
    python tools/train/yolo.py                                  # 使用默认数据集
    python tools/train/yolo.py --data datasets/other.yaml       # 自定义数据集
    python tools/train/yolo.py --epochs 200 --batch 32          # 自定义参数
    python tools/train/yolo.py --resume                         # 恢复中断的训练
"""

import sys
from pathlib import Path

# 路径自举：tools/train/ -> tools/ -> 项目根目录
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# PowerShell 对 \r 回车覆盖支持不佳，导致 tqdm 进度条不断追加新行。
# 检测到 PowerShell 时自动切换到 cmd.exe 重新执行。
if sys.platform == "win32" and "PSMODULEPATH" in __import__("os").environ and not __import__("os").environ.get("_SPOTTING_CMD_SHELL"):
    import subprocess
    import shutil
    import os
    python_exe = shutil.which("python") or sys.executable
    cmd = [python_exe] + sys.argv
    env = os.environ.copy()
    env["_SPOTTING_CMD_SHELL"] = "1"  # 标记已切换，防止递归
    sys.exit(subprocess.call(cmd, shell=True, env=env))  # shell=True 在 Windows 上使用 cmd.exe

import argparse
import shutil
import time
from datetime import datetime


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


def _runtime_model_paths() -> list[Path]:
    """从 config.json 解析运行时 YOLO 检测器声明的 model_path，返回绝对路径列表。

    运行时 (run.py -> pipeline.py) 加载的是 config.json 里声明的权重路径，
    训练产物必须同步到那里才会真正生效。解析失败时返回空列表，不抛异常。
    """
    root = Path(__file__).resolve().parent.parent.parent
    try:
        from config import load_config
        detectors = load_config().detectors
    except Exception as e:
        print(f"[同步] 读取 config.json 失败，跳过运行时模型同步: {e}")
        return []

    paths: list[Path] = []
    for name, det in detectors.items():
        if det.get("type") != "yolo":
            continue
        model_path = det.get("params", {}).get("model_path")
        if not model_path:
            print(f"[同步] 检测器 '{name}' 未声明 model_path，跳过")
            continue
        target = Path(model_path)
        if not target.is_absolute():
            target = root / target
        if target not in paths:
            paths.append(target)

    if not paths:
        print("[同步] config.json 中没有 YOLO 检测器声明 model_path，跳过运行时模型同步")
    return paths


def main():
    parser = argparse.ArgumentParser(description="YOLO 模型训练")
    parser.add_argument("--data", type=str, default="datasets/yolo_dataset/data.yaml",
                        help="数据集配置文件路径 (默认: datasets/yolo_dataset/data.yaml)")
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

    # 自动复制最佳模型到 models/ 并生成说明文档
    if best_model.exists():
        models_dir = Path(__file__).resolve().parent.parent.parent / "models"
        models_dir.mkdir(exist_ok=True)

        # 从 data.yaml 读取类别名
        class_names: list[str] = []
        try:
            import yaml
            with open(data_path, "r", encoding="utf-8") as f:
                data_cfg = yaml.safe_load(f)
            if data_cfg and "names" in data_cfg:
                class_names = list(data_cfg["names"])
        except Exception:
            pass

        # 以训练完成时间为后缀命名
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{args.name}_{ts}"
        dest_model = models_dir / f"{model_name}.pt"
        shutil.copy2(str(best_model), str(dest_model))
        print(f"\n[导出] 最佳模型已复制到 models/:")
        print(f"    {dest_model}")

        # 生成模型说明文档
        doc_path = models_dir / f"{model_name}.txt"
        lines = [
            f"模型: {model_name}.pt",
            f"训练完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"训练耗时: {elapsed / 60:.1f} 分钟",
            f"基础模型: {args.model}",
            f"训练轮数: {args.epochs}",
            f"图片尺寸: {args.imgsz}",
            f"Batch size: {args.batch}",
            f"设备: {resolved_device}",
            f"数据集: {data_path}",
            f"",
            f"可识别类别 ({len(class_names)} 个):",
        ]
        for i, name in enumerate(class_names):
            lines.append(f"  [{i}] {name}")
        if not class_names:
            lines.append("  (未能从 data.yaml 读取类别信息)")

        # 补充关键指标
        if results and hasattr(results, 'results_dict'):
            metrics = results.results_dict
            lines.append("")
            lines.append("关键指标:")
            for k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"):
                if k in metrics:
                    lines.append(f"  {k}: {metrics[k]:.4f}")

        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[导出] 模型说明已保存到:")
        print(f"    {doc_path}")

        # 同步到运行时实际加载的路径，否则 run.py 仍然用着旧权重
        runtime_paths = _runtime_model_paths()
        for target in runtime_paths:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = Path(str(target) + ".bak")
                shutil.copy2(str(target), str(backup))
                print(f"[同步] 旧模型已备份到: {backup}")
            shutil.copy2(str(best_model), str(target))
            print(f"[同步] 已更新运行时模型: {target}")
    else:
        print(f"\n[提示] 可将最佳模型复制到 models/ 目录供检测使用:")
        print(f"    cp {best_model} models/your_model_name.pt")


if __name__ == "__main__":
    main()
