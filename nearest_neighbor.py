import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    self.Xtr = X
    self.ytr = y

  def predict_euclidean(self, X):
    return self.predict(X, NearestNeighbor.euclidean)

  def predict_manhattan(self, X):
    return self.predict(X, NearestNeighbor.manhattan)

  def predict(self, X, dist):
    num_test = X.shape[0]
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    for i in xrange(num_test):
      print "comparing ", i
      # Xtr is a matrix of 5,000 pictures, each picture represented by 3072 numbers
      # X[i, :] is the ith picture in the test set
      # np.abs(Xtr - X[i, :]) essentially means compare every picture
      # in the training set to the ith test picture by subtraction.
      # Let's say Xtr is
      # [ [1,2,3],
      #   [4,5,1],
      #   [0,1,2] ]
      # and X[1, :] is [1,2,3]

      # result = np.abs(Xtr - X[i, :]) is
      # [ [0,0,0],
      #   [3,3,|-2|] ]
      # 
      # np.sum(result, axis = 1) says add all the colums in a single row together
      # essentially, we're looking for the picture whose distance from the test picture
      # is smallest.
      # distances = np.sum(np.abs(self.Xtr - X[i, :]), axis = 1)
      distances = dist(self.Xtr, X[i, :])
      min_index = np.argmin(distances)
      Ypred[i] = self.ytr[min_index]

    return Ypred

  @staticmethod
  def manhattan(Xtr, Xtest_instance):
    return np.sum(np.abs(Xtr - Xtest_instance), axis = 1)

  @staticmethod
  def euclidean(Xtr, Xtest_instance):
    return np.sqrt(np.sum(np.square(Xtr - Xtest_instance), axis = 1))