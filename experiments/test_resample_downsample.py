# %%
import numpy as np
import pandas as pd


input = pd.read_csv(
    "test_data/datasets/battledim/pressures/n1.csv",
    index_col="Timestamp",
    parse_dates=True,
)
input = pd.read_csv(
    "test_data/datasets/graz-ragnitz/pressures/HG3420.csv",
    index_col="Timestamp",
    parse_dates=True,
)

input


# %%
print(len(input))

split_date = "2019-01-01"


# def downsample(input, value):
#     dataframe = input.reset_index()
#     last = dataframe.iloc[-1]
#     print(last)
#     dataframe = dataframe.groupby(
#         (dataframe["Timestamp"] - dataframe["Timestamp"][0]).dt.total_seconds()
#         // (value),
#         group_keys=True,
#     ).first()
#     dataframe.iloc[-1] = last
#     dataframe = dataframe.set_index("Timestamp")
#     return dataframe


# dataframe = pd.concat(
#     [
#         downsample(input.iloc[0 : int(len(input) / 2)], 60 * 60),
#         downsample(input.iloc[int(len(input) / 2) : -1], 60 * 60),
#     ],
#     axis=0,
# )
# dataframe

value = 60 * 60  # downsampling seconds = 10 minutes
dataframe = input.reset_index()
dataframe = dataframe.groupby(
    (dataframe["Timestamp"] - dataframe["Timestamp"][0]).dt.total_seconds() // (value),
    group_keys=True,
).first()
dataframe = dataframe.set_index("Timestamp")
dataframe


# %%
print(len(dataframe))


should_have_value_count = len(input)
should_have_value_count
resample_frequency = "1T"
# sensor_data = dataframe.iloc[0 : int(len(dataframe) / 2)]
sensor_data = dataframe
# sensor_data = input

resampled = sensor_data.resample(resample_frequency).mean()
resampled

missing_values_count = should_have_value_count - len(resampled)
if missing_values_count > 0:
    freq_end = pd.Timedelta(resample_frequency) * missing_values_count

freq_end


missing_dates = pd.DataFrame(
    index=pd.date_range(
        resampled.iloc[-1].name,
        resampled.iloc[-1].name + freq_end,
        freq=resample_frequency,
    )
)
missing_dates[0] = np.NaN
missing_dates.columns = resampled.columns
missing_dates
# complete_resample = pd.concat([resampled, missing_dates], axis=0)
complete_resample = resampled.combine_first(missing_dates)
# # resampled.join(missing_dates, how="inner")
complete_resample

# len(complete_resample)
# resampled


# print(len(resampled))
