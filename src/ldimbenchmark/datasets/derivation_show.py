# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

time_series = pd.Series(np.sin(pd.Series(range(1, 100)) / 10))
time_series


time_series.plot()


# time_series_noise = time_series + np.random.normal(0, 0.1, time_series.shape)
# time_series_noise.plot()

# time_series_resolution = time_series.resample("1D").mean()

# time_series_resolution.plot()

time_series_resample_sensitivity = np.divmod(time_series, 0.2)[0] * 0.2
# sensitivity = 0.2

# for index, datapoint in enumerate(time_series):
#     divisor, rest = divmod(datapoint, sensitivity)
#     time_series_resample_sensitivity.append(divisor * sensitivity + sensitivity)


test = pd.Series(time_series_resample_sensitivity)
test.plot()
print(time_series_resample_sensitivity)
print(time_series)
