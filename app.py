import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import os
import random  # <-- THÊM VÀO ĐỂ GIẢ LẬP KẾT QUẢ

# --- Cấu hình trang (TỐI ƯU) ---
st.set_page_config(
    page_title="Chẩn đoán bệnh da (TTCS)",
    page_icon="🩺",
    layout="centered"
)

# --- Hằng số (CLASS_NAMES) ---
CLASS_NAMES_ENG = [
    'Actinic keratoses (akiec)', 
    'Basal cell carcinoma (bcc)', 
    'Benign keratosis-like lesions (bkl)', 
    'Dermatofibroma (df)', 
    'Melanoma (mel)', 
    'Melanocytic nevi (nv)', 
    'Vascular lesions (vasc)'
]
CLASS_NAMES_VI = [
    'Dày sừng quang hóa', 
    'Ung thư biểu mô tế bào đáy', 
    'Tổn thương giống dày sừng lành tính', 
    'U sợi bì', 
    'U hắc tố', 
    'Nốt ruồi tế bào hắc tố', 
    'Tổn thương mạch máu'
]

# --- Tải mô hình AI (LOGIC DEMO MỚI) ---
@st.cache_resource
def load_app_model():
    """
    Tải mô hình Keras .h5
    Nếu không tìm thấy file, sẽ chạy ở chế độ DEMO.
    """
    MODEL_FILE_PATH = 'model_vFinal.h5'
    
    # Kiểm tra xem file model có tồn tại không
    if os.path.exists(MODEL_FILE_PATH):
        # Nếu TỒN TẠI: Tải mô hình thật
        try:
            model = load_model(MODEL_FILE_PATH) 
            st.sidebar.success("Đã tải mô hình AI thật thành công!")
            return model
        except Exception as e:
            st.error(f"Lỗi khi tải mô hình '{MODEL_FILE_PATH}': {e}")
            return None # Trạng thái lỗi
    else:
        # Nếu KHÔNG TỒN TẠI: Chạy chế độ DEMO
        st.sidebar.warning(
            "Không tìm thấy file 'model_vFinal.h5'.\n\n"
            "Ứng dụng đang chạy ở **CHẾ ĐỘ DEMO**.\n\n"
            "*(Kết quả chỉ là ngẫu nhiên)*"
        )
        return "DEMO_MODE" # Trạng thái Demo

# --- Hàm tiền xử lý ảnh (Không thay đổi) ---
def preprocess_image(image_pil):
    """
    Tiền xử lý ảnh người dùng tải lên
    """
    # VIỆC CỦA AN: (Hỏi Đăng khi có mô hình)
    target_size = (224, 224) 
    image_resized = image_pil.resize(target_size)
    image_array = np.array(image_resized)
    image_normalized = image_array / 255.0 # <-- Giả định
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded

# --- Hàm dự đoán (LOGIC DEMO MỚI) ---
def predict(model, processed_image):
    """
    Dự đoán từ ảnh đã tiền xử lý.
    Xử lý cả 2 trường hợp: Mô hình thật và Chế độ Demo.
    """
    if model == "DEMO_MODE":
        # --- LOGIC GIẢ LẬP (DEMO) ---
        time.sleep(2) # Giả lập thời gian AI xử lý
        
        # Chọn ngẫu nhiên 1 bệnh
        predicted_index = random.randint(0, len(CLASS_NAMES_VI) - 1)
        predicted_class_name_vi = CLASS_NAMES_VI[predicted_index]
        
        # Tạo ngẫu nhiên độ tự tin
        confidence_score = random.uniform(0.75, 0.98)
        
        # Tạo ngẫu nhiên mảng xác suất
        all_scores = np.random.rand(len(CLASS_NAMES_VI))
        all_scores[predicted_index] = confidence_score
        all_scores = all_scores / np.sum(all_scores) # Chuẩn hóa
        
        return predicted_class_name_vi, confidence_score, all_scores
        
    else:
        # --- LOGIC AI THẬT ---
        prediction_scores = model.predict(processed_image)
        predicted_index = np.argmax(prediction_scores[0])
        predicted_class_name_vi = CLASS_NAMES_VI[predicted_index]
        confidence_score = prediction_scores[0][predicted_index]
        
        return predicted_class_name_vi, confidence_score, prediction_scores[0]

# --- GIAO DIỆN APP (Build lại) ---

# --- 1. Thanh bên (SIDEBAR) ---
st.sidebar.title("Giới thiệu dự án 🏥")
st.sidebar.info(
    """
    **Môn học:** Thực tập chuyên ngành (TTCS)\n
    **Mục tiêu:** PoC phân loại 7 bệnh da liễu.\n
    **Dataset:** HAM10000 (Kaggle)\n
    **Model:** Transfer Learning
    """
)
st.sidebar.header("Thành viên nhóm")
st.sidebar.write("- Tùng (Data Lead)")
st.sidebar.write("- Đăng (Model Lead)")
st.sidebar.write("- An (App Lead & PM)")

st.sidebar.header("⚠️ Cảnh báo Y tế")
st.sidebar.warning(
    """
    Kết quả từ AI chỉ mang tính tham khảo, là sản phẩm PoC 
    và **không** thay thế cho chẩn đoán y tế chuyên nghiệp.
    """
)

# --- 2. Giao diện chính (MAIN PAGE) ---
st.title("🩺 Trình chẩn đoán Bệnh Da liễu")
st.write("Tải ảnh lên để AI phân tích. (Đã tối ưu cho điện thoại 📱)")

# Tải mô hình (Thật hoặc Demo)
model = load_app_model()

# --- Tạo Tabs ---
tab1, tab2 = st.tabs(["Chẩn đoán Trực tiếp", "Phân tích & Thống kê"])

# --- Tab 1: Chẩn đoán (Bố cục 1 cột) ---
with tab1:
    if model is None:
        # Trường hợp này chỉ xảy ra nếu file .h5 CÓ TỒN TẠI nhưng bị lỗi
        st.error("Mô hình AI bị lỗi, không thể tải. Vui lòng liên hệ Model Lead (Đăng).")
    else:
        # --- Thẻ 1: Tải ảnh lên ---
        with st.container(border=True):
            st.header("Bước 1: Tải ảnh lên")
            uploaded_file = st.file_uploader(
                "Chọn một tệp ảnh da liễu", 
                type=["jpg", "jpeg", "png"], 
                key="uploader",
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                image_pil = Image.open(uploaded_file)
                st.image(image_pil, caption="Ảnh đã tải lên.", use_column_width=True)
        
        # --- Thẻ 2: Kết quả (Chỉ hiện khi đã tải ảnh) ---
        if uploaded_file is not None:
            with st.container(border=True):
                st.header("Bước 2: Xem kết quả")
                
                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                with col_btn2:
                    start_button = st.button(
                        "Bắt đầu chẩn đoán", 
                        use_container_width=True, 
                        type="primary"
                    )

                if start_button:
                    with st.spinner('Đang phân tích, vui lòng chờ...'):
                        
                        processed_image = preprocess_image(image_pil)
                        # Hàm predict sẽ tự biết dùng model thật hay demo
                        predicted_class, confidence, all_scores = predict(model, processed_image)
                        
                        st.success("Đã hoàn tất chẩn đoán!")
                        
                        st.metric(
                            label="Kết quả dự đoán", 
                            value=predicted_class, 
                            delta=f"Độ tự tin: {confidence * 100:.2f}%"
                        )
                        
                        st.subheader("Phân bố xác suất dự đoán")
                        chart_data = pd.DataFrame({
                            "Bệnh (VI)": CLASS_NAMES_VI,
                            "Xác suất": all_scores
                        })
                        st.bar_chart(chart_data.set_index("Bệnh (VI)"))

# --- Tab 2: Phân tích & Thống kê (An toàn) ---
with tab2:
    st.header("Phân tích dữ liệu & Hiệu suất mô hình")
    st.write("Khu vực của Data Lead (Tùng) và Model Lead (Đăng).")
    
    with st.container(border=True):
        st.subheader("1. Phân bố dữ liệu huấn luyện (Tùng)")
        st.write("Biểu đồ này cho thấy sự mất cân bằng của 7 lớp bệnh trong dataset HAM10000.")
        
        # Dữ liệu giả lập (Lấy số liệu thật từ Tùng)
        chart_data_distribution = pd.DataFrame({
            "Loại bệnh": CLASS_NAMES_VI,
            # Giả lập số liệu cho giống HAM10000
            "Số lượng ảnh": [327, 514, 1099, 115, 1113, 6705, 142]
        })
        st.bar_chart(chart_data_distribution.set_index("Loại bệnh"))
        st.caption("PLACEHOLDER: Dữ liệu thống kê này là giả lập (nhưng dựa trên số liệu thật của HAM10000).")

    with st.container(border=True):
        st.subheader("2. Ma trận nhầm lẫn (Confusion Matrix) (Đăng)")
        st.write("Ma trận nhầm lẫn trên tập Test, cho thấy mô hình hay nhầm lẫn giữa các bệnh nào.")
        
        # Logic an toàn, không crash app
        if os.path.exists('confusion_matrix.png'):
            st.image('confusion_matrix.png', caption="Ma trận nhầm lẫn (từ Model Lead)")
        else:
            st.error("Chưa tìm thấy file 'confusion_matrix.png'. (An hãy lấy file này từ Đăng).")

