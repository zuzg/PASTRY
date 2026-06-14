from src.config import parse_args
from src.experiment import Experiment


def main() -> None:
    cfg = parse_args()
    experiment = Experiment(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    main()
