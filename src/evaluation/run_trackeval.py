import argparse
import sys
import numpy as np

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

sys.path.insert(0, "external/TrackEval")

import trackeval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker-name", required=True)
    parser.add_argument("--seqmap-file", required=True)
    args = parser.parse_args()

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = False
    eval_config["NUM_PARALLEL_CORES"] = 1
    eval_config["PRINT_CONFIG"] = False
    eval_config["BREAK_ON_ERROR"] = True

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = "data/trackeval/gt"
    dataset_config["TRACKERS_FOLDER"] = "data/trackeval/trackers"
    dataset_config["OUTPUT_FOLDER"] = "outputs/trackeval"
    dataset_config["TRACKERS_TO_EVAL"] = [args.tracker_name]
    dataset_config["BENCHMARK"] = "DLCV"
    dataset_config["SPLIT_TO_EVAL"] = "train"
    dataset_config["SEQMAP_FOLDER"] = "data/trackeval/seqmaps"
    dataset_config["SEQMAP_FILE"] = f"data/trackeval/seqmaps/{args.seqmap_file}"
    dataset_config["DO_PREPROC"] = False
    dataset_config["PRINT_CONFIG"] = False

    metrics_config = {
        "METRICS": ["HOTA", "CLEAR", "Identity"],
        "THRESHOLD": 0.5,
        "PRINT_CONFIG": False,
    }

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

    metrics_list = [
        trackeval.metrics.HOTA(metrics_config),
        trackeval.metrics.CLEAR(metrics_config),
        trackeval.metrics.Identity(metrics_config),
    ]

    evaluator.evaluate(dataset_list, metrics_list)


if __name__ == "__main__":
    main()
