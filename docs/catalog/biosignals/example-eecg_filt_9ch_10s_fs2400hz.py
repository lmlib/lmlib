"""
EECG_FILT_9CH_10S_FS2400HZ.csv
==============================

- Esophageal ECG Signal (filtered)
- 9 channel
- 10 seconds
- Sampling rate: 2400Hz

"""

from lmlib.utils import load_lib_csv_mc 
import matplotlib.pyplot as plt

y = load_lib_csv_mc('EECG_FILT_9CH_10S_FS2400HZ.csv')

plt.figure(figsize=(12, 6)) 
for m in range(9): 
    plt.plot(y[:, m] + 8-m, label=f'ch{m}') 
plt.legend() 
plt.xlabel('k') 
plt.show()