from flyai.dataset import Dataset
from model import Model

data = Dataset()
model = Model(data)

# 调用 predict 方法
p = model.predict(image_path='image/15782960406893788.jpg')
print(p)

# 调用 predict_all 方法