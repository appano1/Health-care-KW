import tensorflow as tf


class RecommendModel:
    def __init__(self, rated_matrix, nf=200, alpha=40):
        self.nu = rated_matrix.shape[0]
        self.ni = rated_matrix.shape[1]
        self.nf = nf

        self.X = tf.Variable(tf.random.normal(shape=(self.nu, nf), dtype=tf.float32))
        self.X.assign(tf.math.multiply(self.X, 0.01))
        self.Y = tf.Variable(tf.random.normal(shape=(self.ni, nf), dtype=tf.float32))
        self.Y.assign(tf.math.multiply(self.Y, 0.01))

        self.C = 1 + alpha * rated_matrix
        self.lambda_var = 40
        self.P = tf.Variable(tf.identity(rated_matrix))
        for i in range(self.nu):
            self.P[i].assign(tf.map_fn(lambda x: 1 if x > 0 else x, self.P[i]))

    def loss(self, r_lambda=40):
        self.lambda_var = r_lambda
        predict = self.predict()
        predicted_error = tf.reduce_sum(self.C * tf.square(self.P - predict))
        regularization = r_lambda * (tf.reduce_sum(tf.square(self.X)) + tf.reduce_sum(tf.square(self.Y)))
        total_loss = predicted_error + regularization
        return predicted_error, regularization, total_loss

    def optimize_user(self):
        yT = tf.transpose(self.Y)
        yTy = tf.matmul(yT, self.Y)
        for u in range(self.nu):
            Cu = tf.linalg.diag(self.C[u])
            yTCuY = yTy + tf.matmul(tf.matmul(yT, Cu - tf.eye(Cu.shape[1])), self.Y)
            lI = self.lambda_var * tf.eye(self.nf)
            yTCuPu = tf.matmul(tf.matmul(yT, Cu), tf.expand_dims(self.P[u], axis=1))
            Xu = tf.squeeze(tf.matmul(tf.linalg.inv(yTCuY + lI), yTCuPu), axis=1)
            self.X[u].assign(Xu)

    def optimize_item(self):
        xT = tf.transpose(self.X)
        xTx = tf.matmul(xT, self.X)
        for i in range(self.ni):
            Ci = tf.linalg.diag(self.C[:, i])
            xTCiX = xTx + tf.matmul(tf.matmul(xT, Ci - tf.eye(Ci.shape[1])), self.X)
            lI = self.lambda_var * tf.eye(self.nf)
            xTCiPi = tf.matmul(tf.matmul(xT, Ci), tf.expand_dims(self.P[:, i], axis=1))
            Yi = tf.squeeze(tf.matmul(tf.linalg.inv(xTCiX + lI), xTCiPi), axis=1)
            self.Y[i].assign(Yi)

    @tf.function
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

            tf.print('------------------------step %d----------------------' % i)
            tf.print('predict error: %f' % predict_error)
            tf.print('regularization: %f' % regularization)
            tf.print('total loss: %f' % total_loss)
        return predict_errors, regularization_list, total_losses

    def predict(self):
        return tf.matmul(self.X, tf.transpose(self.Y))
