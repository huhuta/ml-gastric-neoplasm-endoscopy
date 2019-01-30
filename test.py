from multiprocessing.dummy import Pool as ThreadPool


def test_fn(a, b):
    print(a, b)


pool = ThreadPool(4)
a = [1, 2, 3]
b = [4, 5, 6]
results = pool.starmap(test_fn, zip(a, b))
