import numpy as np

class LapisLinearRegression:
    coef_:np.ndarray = None

    def __init__(self) -> None:
        pass

    
    def fit(self, XBar:np.ndarray, y:np.ndarray) -> None:
        aDagger = np.linalg.pinv(np.dot(XBar.T, XBar))
        b = np.dot(XBar.T, y)
        self.coef_ = np.dot(aDagger, b)
    

    def predict(self, x) -> np.ndarray:
        return np.dot(x, self.coef_)

### TEST

# import preProcessData
# X = np.array([147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]).reshape((-1,1))
# y = np.array([49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]).reshape((-1, 1))

# X_bar = preProcessData.makeXbar(X)

# linearRegr = LapisLinearRegression()
# linearRegr.fit(X_bar, y)
# print('w=', linearRegr.coef_)

# xTest = np.array([
#     [1.0, X[0,0]],
#     [1.0, X[-1,0]]
# ])
# yTest = linearRegr.predict(xTest)

# minX = np.min(X) - 10
# maxX = np.max(X) + 10
# minY = np.min(y) - 5
# maxY = np.max(y) + 5

# print(minX, maxX, minY, maxY)

# from matplotlib import pyplot as plt
# plt.plot(X, y, 'ro')
# plt.plot(xTest, yTest)
# plt.axis((minX, maxX, minY, maxY))
# plt.show()