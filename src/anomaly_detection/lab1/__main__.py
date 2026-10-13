import argparse

from anomaly_detection.lab1 import ex1, ex2, ex3, ex4

TASKS = {
    1: ex1.run,
    2: ex2.run,
    3: ex3.run,
    4: ex4.run,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Laborator 1 - Anomaly Detection")
    parser.add_argument(
        "--ex",
        type=int,
        choices=TASKS.keys(),
    )
    args = parser.parse_args()

    if args.ex:
        TASKS[args.ex]()
    else:
        for run_fn in TASKS.values():
            run_fn()


if __name__ == "__main__":
    main()
