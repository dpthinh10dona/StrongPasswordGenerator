# 🛡️ Hệ Thống Nâng Cấp Mật Khẩu Bằng AI (AI-Powered Password Enhancer)

Một hệ thống thông minh kết hợp Deep Learning (Transformer) và Machine Learning (Random Forest) để đánh giá và tự động nâng cấp độ an toàn của mật khẩu.

## 📖 Giới thiệu dự án
Thay vì tạo ra các chuỗi ký tự ngẫu nhiên khó nhớ, hệ thống này cho phép người dùng nhập vào một "mật khẩu yếu" (có ý nghĩa và dễ nhớ đối với họ). Trí tuệ nhân tạo sau đó sẽ phân tích và tự động bổ sung các ký tự cần thiết để biến nó thành một mật khẩu đạt chuẩn bảo mật cao nhất, nhưng vẫn giữ lại "gốc" của người dùng.

Dự án được phát triển nhằm mục đích ứng dụng các mô hình xử lý ngôn ngữ tự nhiên (NLP) vào việc sinh chuỗi ký tự ở cấp độ Character-level.

## ✨ Tính năng nổi bật
* **Đánh giá tự động (Scoring):** Phân loại độ mạnh của mật khẩu (Yếu, Vừa, Mạnh) bằng mô hình Random Forest Classifier dựa trên các đặc trưng kỹ thuật được trích xuất.
* **Nâng cấp thông minh (Enhancement):** Sử dụng kiến trúc Custom Mini-Transformer để phân đoán và sinh thêm các ký tự bảo mật (Chữ hoa, Số, Ký tự đặc biệt) nối tiếp vào mật khẩu gốc.
* **Kiểm duyệt nghiêm ngặt (Strict Validation):** Bộ lọc "Luật thép" đảm bảo mật khẩu đầu ra luôn đáp ứng 100% các tiêu chí an toàn bắt buộc trước khi trả kết quả về cho người dùng.

## 🛠️ Công nghệ sử dụng
* **Deep Learning:** PyTorch (Xây dựng mô hình ngôn ngữ Transformer từ đầu)
* **Machine Learning:** Scikit-learn (Mô hình phân lớp Random Forest)
* **Xử lý dữ liệu:** Pandas, NumPy
* **Môi trường:** Python 3, Google Colab (Tối ưu hóa huấn luyện trên GPU)
## 🚀 Hướng dẫn cài đặt & Chạy dự án

### 1. Cài đặt thư viện
Yêu cầu Python >= 3.8. Cài đặt các gói phụ thuộc bằng lệnh:
```
pip install torch pandas scikit-learn tqdm
```
### 2. Cấu trúc mã nguồn

* `StrongPasswordGenerator.ipynb`: File huấn luyện và thực hiện thí nghiệm trên Google Colab.
* `passwords_dataset.csv`: Dữ liệu huấn luyện (tải từ Kaggle).
* `README.md`: Hướng dẫn dự án.

### 3. Ví dụ chạy thực tế

Hệ thống sẽ yêu cầu bạn nhập tiêu chí và mật khẩu gốc:

```python
# Ví dụ cấu hình
criteria = {"LOWER": 1, "UPPER": 1, "NUM": 1, "SPEC": 1}
weak_pw = "toranub123"

# Kết quả từ AI
Mật khẩu nâng cấp: toranub123!A
Điểm độ mạnh: 99% (Mạnh)



```

---

## 📊 Kết Quả Huấn Luyện

Mô hình đạt được mức Loss thấp (~0.01) sau 5 epochs huấn luyện trên tập dữ liệu mật khẩu lớn, cho thấy khả năng sinh chuỗi ký tự rất gần với hành vi đặt mật khẩu an toàn của con người.

---

## 👨‍💻 Tác Giả

Dự án được thực hiện bởi Nhóm **TrustMiiBro**.

* Github: https://github.com/dpthinh10dona
* Dự án: StrongPasswordGenerator

---

> **Lưu ý:** Dự án này phục vụ mục đích học tập và nghiên cứu của cá nhân trong lĩnh vực an ninh mạng.
