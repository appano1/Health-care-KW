import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, rated_matrix, nf=200, alpha=40):
        self.nu = rated_matrix.shape[0]
        self.ni = rated_matrix.shape[1]
        self.nf = nf
        self.X = tf.cast(np.random.rand(self.nu, nf) * 0.01, dtype=tf.float32)
        self.Y = tf.cast(np.random.rand(self.ni, nf) * 0.01, dtype=tf.float32)
        self.P = np.copy(rated_matrix)
        self.P[self.P > 0] = 1
        self.C = 1 + alpha * rated_matrix
        self.lambda_var = 40

    def loss_function(self, r_lambda=40):
        self.lambda_var = r_lambda
        xTy = tf.matmul(tf.transpose(self.X), self.Y)
        predicted_error = tf.square(self.P - xTy)
        confidence_error = tf.reduce_sum(tf.matmul(self.C, predicted_error))
        regularization = r_lambda * tf.reduce_sum(tf.square(self.X) + tf.square(tf.square(self.Y)))
        total_loss = confidence_error + regularization
        return tf.reduce_sum(predicted_error), confidence_error, regularization, total_loss

    def optimize_user(self):
        yT = tf.transpose(self.Y)
        yTy = tf.matmul(yT, self.Y)
        for u in range(self.nu):
            Cu = tf.linalg.diag(self.C[u])
            yTCuY = yTy + tf.matmul(tf.matmul(yT, Cu - tf.eye(Cu.shape[1])), self.Y)
            lI = self.lambda_var * tf.eye(self.nf)
            print(self.P[u])
            yTCuPu = tf.matmul(tf.matmul(yT, Cu), self.P[u])
            self.X[u] = tf.matmul(tf.linalg.inv(yTCuY + lI), yTCuPu)

    def optimize_item(self):
        xT = tf.transpose(self.X)
        xTx = tf.matmul(xT, self.X)
        for i in range(self.ni):
            Ci = tf.linalg.diag(self.C[:, i])
            xTCiX = xTx + tf.matmul(tf.matmul(xT, Ci - tf.eye(Ci.shape[1])), self.X)
            lI = self.lambda_var * tf.eye(self.nf)
            xTCiPi = tf.matmul(tf.matmul(xT, Ci), self.P[:, i])
            self.Y[i] = tf.matmul(tf.linalg.inv(xTCiX + lI), xTCiPi)

    # @tf.function
    def train_one_step(self):
        self.optimize_user()
        self.optimize_item()

    def train(self, epoch=15):
        predict_errors, confidence_errors, regularization_list, total_losses = [], [], [], []
        for i in range(epoch):
            self.train_one_step()
            predict_error, confidence_error, regularization, total_loss = self.loss_function()

            predict_errors.append(predict_error)
            confidence_errors.append(confidence_error)
            regularization_list.append(regularization)
            total_losses.append(total_loss)

            print('------------------------step %d----------------------' % i)
            print('predict error: %f' % predict_error)
            print('confidence error: %f' % confidence_error)
            print('regularization: %f' % regularization)
            print('total loss: %f' % total_loss)
        return predict_errors, confidence_errors, regularization_list, total_losses

    def predict(self):
        return tf.tensordot(self.X, self.Y, axes=1)


u1 = np.array([0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0])
u2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
u3 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0])
u4 = np.array([0, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0])
u5 = np.array([0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0])
u6 = np.array([0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0])
u7 = np.array([0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5])
u8 = np.array([0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4])
u9 = np.array([0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0])
u10 = np.array([0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0])

R = np.array([u1, u2, u3, u4, u5, u6, u7, u8, u9, u10], dtype=np.float32)

model = Model(R)
model.train()
print(model.predict())
