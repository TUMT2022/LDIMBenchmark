# %%
# https://github.com/numpy/numpy/issues/23244
import multiprocessing
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
from threadpoolctl import threadpool_limits


def task(num):
    T = 10000
    N = 100

    K0 = np.zeros((N, N))
    K1 = np.zeros((N, N))
    Kd = np.zeros((N, N))

    np.fill_diagonal(K0, 0)
    np.fill_diagonal(K1, 1)
    np.fill_diagonal(Kd, 0)

    test1 = np.multiply.outer(K0, np.ones(T))
    testA = np.multiply.outer(K1, np.ones(T))
    # with threadpool_limits(limits=1, user_api="blas"):
    #     testB = np.tensordot(K1, np.ones(T), axes=((), ()))
    # # test1 = np.multiply.outer(K0, np.ones(T))


if __name__ == "__main__":
    T = 1000
    N = 100

    K0 = np.zeros((N, N))
    K1 = np.zeros((N, N))
    Kd = np.zeros((N, N))

    np.fill_diagonal(K0, 0)
    np.fill_diagonal(K1, 1)
    np.fill_diagonal(Kd, 0)

    test1 = np.multiply.outer(K0, np.ones(T))
    testA = np.multiply.outer(K1, np.ones(T))

    print(cpu_count())

    with multiprocessing.get_context("spawn").Pool(
        # with Pool(
        processes=cpu_count()
        - 1,
    ) as p:
        jobs = [p.apply_async(func=task, args=[num]) for num in range(cpu_count() + 2)]

        for job in tqdm(jobs):
            job.get()
