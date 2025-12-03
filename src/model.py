import tensorflow as tf

class ModelLoader:
    def __init__(self):
        self._model_paths = {
            "V1": "src/models/baseline.keras",
            "V2": "src/models/train001.keras",
            "V3": "src/models/train002.keras",
            "V4": "src/models/train003.keras"
        }
        self._cache = {}
        # Tên lớp tiếng Việt + tiếng Anh
        self._class_list = [
            'Ớt chuông (Bellpepper)',
            'Cà rốt (Carrot)',
            'Dưa leo (Cucumber)',
            'Hành tây (Onion)',
            'Khoai tây (Potato)',
            'Cà chua (Tomato)'
        ]

    def get_model(self, version: str):
        if version not in self._cache:
            self._cache[version] = tf.keras.models.load_model(self._model_paths[version], compile=False)
        return self._cache[version]

    def get_class_list(self):
        return self._class_list

    def get_all_versions(self):
        return list(self._model_paths.keys())


class Predictor:
    def __init__(self, class_list):
        self.classes = class_list

    def predict(self, model, X):
        """
        Trả về: (tên lớp dự đoán, xác suất dự đoán)
        X: np.ndarray shape (1,224,224,3)
        """
        try:
            X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
            probs = model(X_tensor, training=False).numpy()[0]
            idx = probs.argmax()
            return self.classes[idx], float(probs[idx])
        except Exception as e:
            raise RuntimeError(f"Lỗi khi dự đoán: {str(e)}")
