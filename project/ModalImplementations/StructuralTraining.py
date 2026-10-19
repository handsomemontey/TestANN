import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    data = pd.read_csv(Path(__file__).resolve().parent.parent / "06-study_hours_grades.csv")
    print(data.head())


if __name__ == "__main__":
    main()
