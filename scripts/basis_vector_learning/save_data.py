from typing import List
import os
import numpy as np
import pandas as pd

def save_to_csv(data) -> None:
    pd.DataFrame(data).to_csv("data.csv")