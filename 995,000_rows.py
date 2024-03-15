import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

chunksize = 10000
chunks = []
for chunk in pd.read_csv('995,000_rows.csv', chunksize=chunksize):
    chunks.append(chunk)

df=pd.concat(chunks)