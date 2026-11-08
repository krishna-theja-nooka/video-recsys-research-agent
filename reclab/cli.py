import argparse
import json

from reclab.data import download, prepare


def main():
    parser = argparse.ArgumentParser(description="Bounded recommendation research lab")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--data", default="data/ml-100k")
    fetch.add_argument("--accept-license", action="store_true")
    prep = sub.add_parser("prepare")
    prep.add_argument("--data", default="data/ml-100k")
    prep.add_argument("--engine", choices=["python", "spark"], default="python")
    exp = sub.add_parser("experiment")
    exp.add_argument("--data", default="data/ml-100k")
    exp.add_argument("--output", default="runs/research")
    exp.add_argument("--epochs", type=int, default=8)
    exp.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    tune = sub.add_parser("tune")
    tune.add_argument("--data", default="data/ml-100k")
    tune.add_argument("--output", default="runs/tuning")
    tune.add_argument("--trials", type=int, default=4)
    args = parser.parse_args()
    if args.command == "download":
        print(download(args.data, args.accept_license))
    elif args.command == "prepare":
        print(json.dumps(prepare(args.data, args.engine), indent=2))
    elif args.command == "experiment":
        from reclab.experiment import run
        from reclab.train import Config

        run(args.data, args.output, Config(epochs=args.epochs), tuple(args.seeds))
    elif args.command == "tune":
        from reclab.experiment import tune

        print(json.dumps(tune(args.data, args.output, args.trials), indent=2))


if __name__ == "__main__":
    main()
