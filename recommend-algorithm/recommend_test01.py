import recommendation_model_np as RM
import numpy as np
import sys

R = sys.argv[1].split(',')
R = np.array(R, dtype=np.float32).reshape([10, 11])

model = RM.RecommendModel(R)
model.train()
result = model.predict(int(sys.argv[2]))
for num in result:
    print(num)
