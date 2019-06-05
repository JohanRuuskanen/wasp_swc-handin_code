from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
import scipy.io
import numpy as np
import time

if __name__ == "__main__":
    sc = SparkContext(appName="PythonSVDExample")

    M = scipy.io.mmread('someMatrix.mtx')
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
    qr = mat.tallSkinnyQR()
    end = time.time()

    t = end-start

    print("Time for QR decomposition: %s seconds" % t)
    sc.stop()
