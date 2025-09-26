import numpy as np
def Neumann(phi):
    nrow , ncol = phi.shape
    nrow -=1
    ncol -=1
    g = phi
    (g[0,0],g[0,ncol],g[nrow,0],g[nrow,ncol]) = (g[2,2],g[2,ncol-3],g[nrow-3,2],g[nrow-3,ncol-3])
    (g[0,1:-1],g[nrow,1:-1]) = (g[2,1:-1],g[nrow-3,1:-1])
    (g[1:-1,1],g[1:-1,ncol]) = (g[1:-1,2],g[1:-1,ncol-3])
    
    return g

matrix = np.arange(100, dtype=np.float64).reshape(10, 10)

print("## Matrix Before Neumann:\n", matrix)

# In the original C++ code, `neumann(matrix)` modifies the matrix in-place.
# The Python/OpenCV equivalent `copyMakeBorder` returns a new matrix with the border.
# If you need to simulate in-place modification for a different neumann logic,
# you would manipulate the matrix directly.

# For this example, we'll demonstrate the common `BORDER_REPLICATE` method.
matrix_after_neumann = Neumann(matrix)

print("\n## Matrix After Neumann (BORDER_REPLICATE):\n", matrix_after_neumann)