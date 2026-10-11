from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RADH-FuseNet 两阶段训练、受试者独立评估和推理")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="演示")
    demo.add_argument("--output", default="runs/demo")
    demo.add_argument("--device", default="cpu")
    # Packaged copies mirror root configs/ so installed wheels retain the demo.
    demo.add_argument("--config", default=str(Path(__file__).resolve().parent/"configs"/"smoke.json"))
    demo.add_argument("--robustness", action="store_true", help="增加冻结模型的扰动测试")
    synth = sub.add_parser("make-demo-data", help="生成显式标记的生理样合成 NPZ")
    synth.add_argument("--output", default="data/demo.npz")
    synth.add_argument("--subjects", type=int, default=6)
    synth.add_argument("--windows-per-subject", type=int, default=5)
    synth.add_argument("--samples", type=int, default=128)
    synth.add_argument("--horizon", type=int, default=4)
    synth.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train", help="仅用训练受试者拟合归一化；验证集选最优检查点")
    train.add_argument("--data", required=True)
    train.add_argument("--config", default="configs/paper.json")
    train.add_argument("--output", required=True)
    train.add_argument("--split-json", help="明确指定 train/val/test 受试者 ID")
    train.add_argument("--device", default="cpu")
    evaluate = sub.add_parser("evaluate", help="按检查点保存的受试者分区评估，不重新拟合预处理")
    predict = sub.add_parser("predict", help="输出分类、可靠性、不确定性、连续预测及预测方差 CSV")
    for command in (evaluate, predict):
        command.add_argument("--data", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--batch-size", type=int, default=16)
        command.add_argument("--threads", type=int, default=2, help="CPU 计算线程数")
    evaluate.add_argument("--partition", choices=("train", "val", "test"), default="test")
    evaluate.add_argument("--robustness", action="store_true")
    predict.add_argument("--partition", choices=("all", "train", "val", "test"), default="all")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "make-demo-data":
        from .data import make_demo_data
        print(make_demo_data(args.output, args.subjects, args.windows_per_subject, args.samples, args.horizon, seed=args.seed))
    elif args.command == "train":
        from .train import train_model
        print(json.dumps(train_model(args.data, args.config, args.output, args.device, args.split_json), ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        import torch
        from .evaluate import evaluate_checkpoint
        torch.set_num_threads(args.threads)
        print(json.dumps(evaluate_checkpoint(args.data, args.checkpoint, args.output, args.device, args.batch_size, args.partition, args.robustness), ensure_ascii=False, indent=2))
    elif args.command == "predict":
        import torch
        from .predict import predict_file
        torch.set_num_threads(args.threads)
        print(predict_file(args.data, args.checkpoint, args.output, args.device, args.batch_size, args.partition))
    elif args.command == "demo":
        from .data import make_demo_data
        from .evaluate import evaluate_checkpoint
        from .predict import predict_file
        from .train import read_config, train_model
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        config = read_config(args.config)
        data_path = make_demo_data(output/"synthetic.npz", samples=config.get("demo", {}).get("samples", 128),
                                  subjects=config.get("demo", {}).get("subjects", 6),
                                  windows_per_subject=config.get("demo", {}).get("windows_per_subject", 5),
                                  horizon=config["model"].get("horizon", 4),
                                  num_classes=config["model"].get("num_classes", 5),
                                  seed=config["training"].get("seed", 42),
                                  sample_hz=config["model"].get("sampling_rate", 32))
        result = train_model(data_path, args.config, output, args.device, synthetic_demo=True)
        evaluate_checkpoint(data_path, result["checkpoint"], output/"metrics.json", args.device,
                            batch_size=config["training"].get("batch_size", 2), robustness=args.robustness)
        predict_file(data_path, result["checkpoint"], output/"predictions.csv", args.device,
                     batch_size=config["training"].get("batch_size", 2), partition="test")
        print(f"测试完成 产物：{output.resolve()}")


if __name__ == "__main__":
    main()
