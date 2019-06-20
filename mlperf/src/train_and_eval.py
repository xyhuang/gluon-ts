import argparse
import csv
import json
import sys
import time

import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas
from gluonts.evaluation.backtest import backtest_metrics, make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer

def main():
  parser = argparse.ArgumentParser(description="Run DeepAR model")
  parser.add_argument("--result-file", type=str, help="result file")
  parser.add_argument("--log-file", type=str, help="log file")
  parser.add_argument("--device", type=str, help="device")
  parser.add_argument("--num-epochs-per-iteration", type=int, help="number of epochs per iteration")
  parser.add_argument("--num-iterations", type=int, help="number of iterations")
  args = parser.parse_args()

  log_file = open(args.log_file, "a")
  sys.stdout = sys.stderr = log_file

  run_start = time.time()

  dataset = get_dataset("electricity", regenerate=False)

  estimator = DeepAREstimator(
    prediction_length=dataset.metadata.prediction_length,
    context_length=168,
    freq=dataset.metadata.time_granularity,
    num_layers=3,
    num_cells=40,
    cell_type="lstm",
    num_eval_samples=100,
    dropout_rate=0.1,
    embedding_dimension=20,
    trainer=Trainer(ctx=args.device, epochs=args.num_epochs_per_iteration, learning_rate=1E-3, hybridize=True, num_batches_per_epoch=64,),
  )

  train_time = 0
  eval_time = 0

  nd = []
  rmse = []
  for epoch in range(args.num_iterations):
    train_start = time.time()
    predictor = estimator.train(dataset.train)
    train_stop = time.time()
    train_time += train_stop - train_start
    eval_start = time.time()
    agg_metrics, item_metrics = backtest_metrics(
        train_dataset=dataset.train,
        test_dataset=dataset.test,
        forecaster=predictor,
    )
    eval_stop = time.time()
    eval_time += eval_stop - eval_start

    for m, v in agg_metrics.items():
      if m == "ND":
        nd.append(v)
        print(f"{m}: {v}")
      if m == "NRMSE":
        rmse.append(v)
        print(f"{m}: {v}")

  run_stop = time.time()
  print(f"Train time: {train_time}")
  print(f"Evaluation time: {eval_time}")
  print(f"Run time: {run_stop - run_start}")

  with open(args.result_file, "w", newline='') as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(nd)
    writer.writerow(rmse)
  
  log_file.close()


if __name__ == "__main__":
  main()
