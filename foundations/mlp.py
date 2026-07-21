import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)

         # Ensure input is float64
        h = np.asarray(x, dtype=np.float64)

        # Forward pass through each layer
        for i, (W, b) in enumerate(zip(weights, biases)):
            W = np.asarray(W, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)

            # Linear transformation
            h = np.dot(h, W) + b

            # Apply ReLU to hidden layers only
            if i < len(weights) - 1:
                h = np.maximum(0, h)

        # Round the final output
        return np.round(h, 5)