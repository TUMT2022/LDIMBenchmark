from ldimbenchmark.methods.utils.cusum import cusum, cusum_old
import pandas as pd
import numpy as np


def test_cusum(snapshot, benchmark):
    n = 40
    np.random.seed(1332452)
    data = pd.DataFrame(
        {
            "test": range(n),  # np.random.randint(2, size=10)
            "test2": range(0, n * 12, 12),
            "test3": np.random.randint(40, size=n),
            "null": np.zeros(n),
        },  # np.random.randint(2, size=10)
        index=pd.date_range("2015-07-03", periods=n, freq="D"),
    )
    # leak_det, df_cs = benchmark(cusum, data)
    leak_det, df_cs = benchmark(cusum, data)
    snapshot.assert_match(data)
    snapshot.assert_match(df_cs.astype(float).round(3).to_csv())
    snapshot.assert_match(leak_det)


def test_cusum_old(snapshot, benchmark):
    n = 40
    np.random.seed(1332452)
    data = pd.DataFrame(
        {
            "test": range(n),  # np.random.randint(2, size=10)
            "test2": range(0, n * 12, 12),
            "test3": np.random.randint(40, size=n),
            "null": np.zeros(n),
        },  # np.random.randint(2, size=10)
        index=pd.date_range("2015-07-03", periods=n, freq="D"),
    )
    leak_det, df_cs = benchmark(cusum_old, data)
    snapshot.assert_match(data)
    snapshot.assert_match(df_cs.to_csv())
    snapshot.assert_match(leak_det)
