import numpy as np
import matplotlib.pyplot as plt


class RecommendModel:
    def __init__(self, rated_matrix, nf=200, alpha=40, _lambda=40):
        R = np.array(rated_matrix)

        self.nf = nf       # 차원 갯수
        self.alpha = alpha     # cui = 1 + alpha * rui
        self._lambda = _lambda  # 규제항

        self.nu = R.shape[0]
        self.ni = R.shape[1]

        self.X = np.random.rand(self.nu, self.nf) * 0.01
        self.Y = np.random.rand(self.ni, self.nf) * 0.01

        self.P = np.copy(R)
        self.P[self.P > 0] = 1
        self.C = 1 + self.alpha * R

    def loss(self):
        predict_error = np.sum(self.C * np.square(self.P - self.predict()))
        regularization = self._lambda * (np.sum(np.square(self.X)) + np.sum(np.square(self.Y)))
        total_loss = predict_error + regularization
        return predict_error, regularization, total_loss

    def optimize_user(self):
        yT = np.transpose(self.Y)
        yTy = np.matmul(yT, self.Y)
        for u in range(self.nu):
            Cu = np.diag(self.C[u])
            yTCuY = yTy + np.matmul(np.matmul(yT, Cu - np.eye(Cu.shape[0])), self.Y)
            lI = self._lambda * np.eye(self.nf)
            yTCuPu = np.matmul(np.matmul(yT, Cu), self.P[u])
            Xu = np.matmul(np.linalg.inv(yTCuY + lI), yTCuPu)
            self.X[u] = Xu

    def optimize_item(self):
        xT = np.transpose(self.X)
        xTx = np.matmul(xT, self.X)
        for i in range(self.ni):
            Ci = np.diag(self.C[:, i])
            xTCiX = xTx + np.matmul(np.matmul(xT, Ci - np.eye(Ci.shape[0])), self.X)
            lI = self._lambda * np.eye(self.nf)
            xTCiPi = np.matmul(np.matmul(xT, Ci), self.P[:, i])
            Yi = np.matmul(np.linalg.inv(xTCiX + lI), xTCiPi)
            self.Y[i] = Yi

    def train_one_step(self):
        self.optimize_user()
        self.optimize_item()

    def train(self, epoch=15):
        predict_errors, regularization_list, total_losses = [], [], []
        for i in range(epoch):
            self.train_one_step()
            predict_error, regularization, total_loss = self.loss()

            predict_errors.append(predict_error)
            regularization_list.append(regularization)
            total_losses.append(total_loss)

            # print('------------------------step %d----------------------' % (i+1))
            # print('predict error: %f' % predict_error)
            # print('regularization: %f' % regularization)
            # print('total loss: %f' % total_loss)
        return predict_errors, regularization_list, total_losses

    def predict(self, user_n=-1):
        result = np.matmul(self.X, np.transpose(self.Y))
        if user_n == -1:
            return result
        else:
            return result[user_n]
