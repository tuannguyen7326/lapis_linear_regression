import numpy as np

def makeXbar(data:np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            np.ones((data.shape[0], 1)),
            data
        ),
        axis=1
    )