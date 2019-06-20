from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas

def main():
  dataset = get_dataset("electricity", regenerate=False)

  train_series = to_pandas(next(iter(dataset.train)))
  test_series = to_pandas(next(iter(dataset.test)))

  print(f"Length of training data: {len(dataset.train)}")
  print(f"Length of testing data: {len(dataset.test)}")

  print(f"Length of training series: {len(train_series)}")
  print(f"length of test series: {len(test_series)}")
  print(f"Length of forecasting window in test dataset: {len(test_series) - len(train_series)}")
  print(f"Recommended prediction horizon: {dataset.metadata.prediction_length}")
  print(f"Frequency of the time series: {dataset.metadata.time_granularity}")

if __name__ == "__main__":
  main()
