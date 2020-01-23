import numpy as np
import matplotlib.pyplot as plt

r_lambda = 40  # 규제항
nf = 200       # 차원 갯수
alpha = 40     # cui = 1 + alpha * rui

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

R = np.array([u1, u2, u3, u4, u5, u6, u7, u8, u9, u10])
print(R.shape)

nu = R.shape[0]
ni = R.shape[1]

X = np.random.rand(nu, nf) * 0.01
Y = np.random.rand(ni, nf) * 0.01
print(X)

P = np.copy(R)
P[P > 0] = 1
print(P)

C = 1 + alpha * R
print(C)


def loss_function(C, P, xTy, X, Y, r_lambda):
    predict_error = np.square(P - xTy)
    confidence_error = np.sum(C * predict_error)
    regularization = r_lambda * (np.sum(np.square(X)) + np.sum(np.square(Y)))
    total_loss = confidence_error + regularization
    return np.sum(predict_error), confidence_error, regularization, total_loss


def optimize_user(X, Y, C, P, nu, nf, r_lambda):
    yT = np.transpose(Y)
    for u in range(nu):
        Cu = np.diag(C[u])
        yT_Cu_y = np.matmul(np.matmul(yT, Cu), Y)
        lI = np.dot(r_lambda, np.identity(nf))
        yT_Cu_pu = np.matmul(np.matmul(yT, Cu), P[u])
        X[u] = np.linalg.solve(yT_Cu_y + lI, yT_Cu_pu)


def optimize_item(X, Y, C, P, ni, nf, r_lambda):
    xT = np.transpose(X)
    for i in range(ni):
        Ci = np.diag(C[:, i])
        xT_Ci_x = np.matmul(np.matmul(xT, Ci), X)
        lI = np.dot(r_lambda, np.identity(nf))
        xT_Ci_pi = np.matmul(np.matmul(xT, Ci), P[:, i])
        Y[i] = np.linalg.solve(xT_Ci_x + lI, xT_Ci_pi)


predict_errors = []
confidence_errors = []
regularization_list = []
total_losses = []

for i in range(50):
    if i != 0:
        optimize_user(X, Y, C, P, nu, nf, r_lambda)
        optimize_item(X, Y, C, P, ni, nf, r_lambda)
    predict = np.matmul(X, np.transpose(Y))
    predict_error, confidence_error, regularization, total_loss = \
        loss_function(C, P, predict, X, Y, r_lambda)

    predict_errors.append(predict_error)
    confidence_errors.append(confidence_error)
    regularization_list.append(regularization)
    total_losses.append(total_loss)

    print('------------------------step %d----------------------' % i)
    print('predict error: %f' % predict_error)
    print('confidence error: %f' % confidence_error)
    print('regularization: %f' % regularization)
    print('total loss: %f' % total_loss)

predict = np.matmul(X, np.transpose(Y))
print('final predict')
print([predict])

plt.plot([i for i in range(50)], predict_errors)
plt.plot([i for i in range(50)], confidence_errors)
plt.plot([i for i in range(50)], regularization_list)
plt.plot([i for i in range(50)], total_losses)
plt.legend(['predict error', 'confidence error', 'regularization', 'loss'])
plt.show()
