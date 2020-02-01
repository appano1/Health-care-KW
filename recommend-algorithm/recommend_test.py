import numpy as np
from datetime import datetime
import recommendation_model_np as rm
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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

model = rm.RecommendModel(R)

t1 = datetime.now()
predicts, regularization, losses = model.train()
t2 = datetime.now()
print(model.predict())
print('Duration : ', t2 - t1)

plt.plot([i for i in range(15)], predicts)
plt.xlabel('Prediction')
plt.show()
plt.plot([i for i in range(15)], regularization)
plt.xlabel('Regularization')
plt.show()
plt.plot([i for i in range(15)], losses)
plt.xlabel('Loss')
plt.show()

'''
Using np => Duration :  0:00:00.235455
Using tf => Duration :  0:00:07.405049
'''
