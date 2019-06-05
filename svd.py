from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
import scipy.io
import numpy as np
import time

if __name__ == "__main__":
    sc = SparkContext(appName="PythonSVDExample")

    M = scipy.io.mmread("someMatrix.mtx")
    rows = M.row.tolist()
    cols = M.col.tolist()
    data = M.data.tolist()
    dataLength = len(data)
    numRows = M.shape[0]
    numCols = M.shape[1]

    matrix = np.zeros((numRows,numCols))
    for i in range(0,dataLength):
        matrix[rows[i],cols[i]] = data[i]
    rows = sc.parallelize(matrix.tolist())

    mat = RowMatrix(rows)

    # Compute the top 5 singular values and corresponding singular vectors.
    start = time.time()
    svd = mat.computeSVD(5, computeU=True)
    end = time.time()
    U = svd.U       # The U factor is a RowMatrix.
    s = svd.s       # The singular values are stored in a local dense vector.
    V = svd.V       # The V factor is a local dense matrix.

    t = end-start

    #collected = U.rows.collect()
    #print("U factor is:")
    #for vector in collected:
    #    print(vector)
    print("Singular values are: %s" % s)
    #print("V factor is:\n%s" % V)
    print("Time for SVD: %s seconds" % t)
    sc.stop()
