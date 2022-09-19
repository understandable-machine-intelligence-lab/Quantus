from quantus.helpers.utils import expand_indices
import numpy as np

a = np.arange(0, 1000).reshape((10, 10, 10))
#sl = (slice(None), slice(5, 7), slice(5, 10))
#sl = [slice(5, 10), np.array([3, 4])]
sl = [slice(0, 2), np.array([0, 1]), np.array([0, 2])]
#sl = [1, 4, 6, 87, 9, 0, 105]

print(a[expand_indices(a, sl, [0, 1, 2])].shape)
print(a[expand_indices(a, sl, [0, 1, 2])])