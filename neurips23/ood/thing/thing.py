import numpy as np
import expann_py

from base import BaseOODANN
from benchmark.datasets import DATASETS

print(expann_py)

class ExpAnnWrapper(BaseOODANN):
    def __init__(self, metric, index_params):
        self.engine = expann_py.expANN()
        self.name = self.engine.name()
        self.res = None

    def fit(self, dataset):
        ds = DATASETS[dataset]()
        for vectors in ds.get_dataset_iterator():
            for vector in vectors:
                v = expann_py.Vec(vector.tolist())
                self.engine.store_vector(v)
        self.engine.build()

    def query(self, X, k):
        query_vectors = []
        for query_vector in X:
            v = expann_py.Vec(query_vector.tolist())
            query_vectors.append(v)
        result_indices = self.engine.query_k_batch(query_vectors, k)

        max_len = len(max(result_indices, key=len))
        for result_index in result_indices:
            while len(result_index) < max_len:
                result_index.append(0)
        self.res = np.array(result_indices)

    def set_query_arguments(self, query_args):
        pass

    def load_index(self, dataset):
        return False

    def get_results(self):
        if self.res is None:
            raise ValueError("Run a query before getting results")
        return self.res

    def __str__(self):
        return self.name
