# Test riêng model V2
from src.model import ModelLoader
loader = ModelLoader()
model_v2 = loader.get_model("V2")
print(model_v2.summary())