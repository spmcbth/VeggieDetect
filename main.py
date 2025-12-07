import streamlit as st
from PIL import Image
from src.preprocess import preprocess
from src.model import ModelLoader, Predictor

# ===================== Khởi tạo =====================
model_loader = ModelLoader()
class_list = model_loader.get_class_list()
predictor = Predictor(class_list)

# ===================== Streamlit Config =====================
st.set_page_config(
    page_title="VeggieDetect",
    layout="wide",
    page_icon="🥬"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# ===================== Header =====================
st.markdown("""
<div class="main-header">
    <h1>🥬 VeggieDetect</h1>
    <p>Nhận dạng rau củ thông minh với Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# ===================== Session State =====================
if 'result' not in st.session_state:
    st.session_state['result'] = None
if 'predicted' not in st.session_state:
    st.session_state['predicted'] = False
if 'image' not in st.session_state:
    st.session_state['image'] = None
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = None

# ===================== Layout 2 cột =====================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-header">📤 Tải ảnh lên</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Chọn ảnh rau củ của bạn", type=["png","jpg","jpeg"], help="Hỗ trợ định dạng: PNG, JPG, JPEG"
    )
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.session_state['image'] = image
            st.image(image, use_container_width=True)
            st.session_state['result'] = None
            st.session_state['predicted'] = False
        except Exception as e:
            st.error(f"Lỗi khi đọc ảnh: {str(e)}")
    else:
        st.info("Vui lòng tải ảnh lên để bắt đầu nhận dạng")

with col2:
    st.markdown('<div class="section-header">🔎 Chọn mô hình</div>', unsafe_allow_html=True)

    selected_model = st.selectbox(
        "Phiên bản model",
        options=model_loader.get_all_versions(),
        index=0
    )
    st.session_state['selected_model'] = selected_model

    if st.button("Nhận dạng"):
        if st.session_state['image'] is None:
            st.warning("Vui lòng tải ảnh trước.")
        else:
            with st.spinner("Đang xử lý..."):
                try:
                    X = preprocess(st.session_state['image'])
                    model = model_loader.get_model(selected_model)
                    pred_class, pred_prob = predictor.predict(model, X)

                    st.session_state['result'] = (pred_class, pred_prob)
                    st.session_state['predicted'] = True

                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")

    # ===================== Kết quả =====================
    st.markdown('<div class="section-header">Kết quả</div>', unsafe_allow_html=True)

    if st.session_state['predicted']:
        pred_class, pred_prob = st.session_state['result']
        st.markdown(
            f"""
            <div class="result-box result-success">
                <strong>{pred_class}</strong><br>
                Accuracy: {pred_prob*100:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-box">Chưa có kết quả</div>',
            unsafe_allow_html=True
        )

# ===================== Footer =====================
st.markdown("""
<div class="footer">
    <p>✨ CT282 – Deep Learning | Nhóm thực hiện: Mạch Gia Hân, Trần Trương Ngọc Uyển, Trần Tiểu Mẫn ✨</p>
</div>
""", unsafe_allow_html=True)
