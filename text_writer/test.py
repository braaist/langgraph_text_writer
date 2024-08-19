class CachedFunction:
    """
    Wrapper around a given callable which caches its results.
    We would like to "cache" multiple callables at the same time.
    """
    _cache = dict()

    def __init__(self, function, warm_cache=dict()):
        self.function = function
        self._cache.update(**warm_cache)

    def calculate_function_value(self, *args, **kwargs):
        '''return the function value for the given arguments & cache it'''
        all_args = tuple(args, 'kwargs': kwargs)
        if all_args not in self._cache:
            self._cache[all_args] = self.function(*args, **kwargs)
        return self._cache[all_args]

    def save_to_file(self, file_name):
        import pickle
        f = open(file_name, 'wb')
        pickle.dump(self, f)

    def __repr__(self):
        return "CachedFunction"

    def __hash__(self):
        import random
        return random.randint(0, 100000)

    def __eq__(self, other):
        return self.function == other.function


add1 = lambda x: x + 1
cached_add1 = CachedFunction(add1)
# invoke add1, store the result (3) in cache & return it
print(cached_add1.calculate_function_value(2))
# return cached value (3)
print(cached_add1.calculate_function_value(2))
