# BÁO CÁO THỰC HÀNH LAB MLOPS
**Khóa học:** AI In Action - VinUni
**Học viên:** Ha Hung Phuoc - 2A202600367
**Đề tài:** Triển khai CI/CD Pipeline 

---

## 1. Lựa chọn Siêu tham số (Hyperparameters)

Dựa trên kết quả thực nghiệm tại Bước 1 với MLflow, tôi đã chọn bộ siêu tham số sau cho mô hình Random Forest:

| Tham số | Giá trị |
| :--- | :--- |
| `n_estimators` | 500 |
| `max_depth` | None (Unlimited) |
| `min_samples_split` | 2 |

**Lý do lựa chọn:**
- **Hiệu năng:** Bộ tham số này đạt Accuracy là **0.746** và F1-score là **0.745** trên tập evaluation, vượt ngưỡng yêu cầu của lab (0.70).
- **Độ ổn định:** Việc tăng `n_estimators` lên 500 giúp mô hình ổn định hơn và giảm thiểu phương sai (variance) so với các giá trị thấp hơn (như 100).
- **Tính tổng quát:** Mặc dù để `max_depth` tự do, nhưng kết quả trên tập Eval vẫn duy trì ở mức cao, chứng tỏ mô hình không bị overfitting quá mức trên tập dữ liệu này.

---

## 2. Khó khăn gặp phải và Cách giải quyết

Trong quá trình thực hiện, tôi đã gặp một số vấn đề kỹ thuật và đã xử lý như sau:

### 2.1. Lỗi kết nối SSH (Connection Timed Out)
- **Khó khăn:** Không thể SSH vào VM bằng IP được cung cấp ban đầu.
- **Giải quyết:** Kiểm tra lại trạng thái VM trên AWS Console. Phát hiện IP đã thay đổi sau khi khởi động lại. Tôi đã cập nhật lại `VM_HOST` trong GitHub Secrets và kiểm tra Security Group để đảm bảo cổng 22 đã được mở cho IP của mình.

### 2.2. Lỗi môi trường trên Amazon Linux (pip3 command not found)
- **Khó khăn:** Lab hướng dẫn sử dụng lệnh `apt` của Ubuntu, nhưng VM thực tế chạy Amazon Linux. Lệnh `pip3` chưa có sẵn.
- **Giải quyết:** Sử dụng trình quản lý gói `yum` để cài đặt thủ công: `sudo yum install -y python3-pip`. Sau đó cài đặt các thư viện cần thiết như FastAPI, Boto3, v.v.

### 2.3. Lỗi định dạng JSON trên PowerShell
- **Khó khăn:** Khi dùng lệnh `curl` trên Windows PowerShell để test endpoint `/predict`, dữ liệu JSON bị báo lỗi "Expecting property name enclosed in double quotes" do PowerShell tự động loại bỏ dấu ngoặc kép khi gọi file thực thi bên ngoài.
- **Giải quyết:** Chuyển sang sử dụng `curl.exe` (binary gốc của Windows) và áp dụng kỹ thuật escape dấu ngoặc kép bằng backtick (<code>\`"</code>) hoặc lưu JSON vào biến trước khi gửi.

### 2.4. Lỗi "Model not found" khi gọi API
- **Khó khăn:** API trả về lỗi 500 do không tìm thấy file `model.pkl` trong thư mục `~/models/`.
- **Giải quyết:** Để kiểm tra nhanh tính đúng đắn của code, tôi đã sử dụng lệnh `scp` để đẩy trực tiếp file model từ máy cá nhân lên VM thay vì chờ pipeline đồng bộ qua S3:
  `scp -i key.pem models/model.pkl ec2-user@IP:~/models/model.pkl`

---

## 3. Kết luận
Toàn bộ pipeline đã hoạt động chính xác. Endpoint `/health` trả về trạng thái "ok" và endpoint `/predict` trả về kết quả dự đoán đúng định dạng JSON yêu cầu. Hệ thống đã sẵn sàng cho việc huấn luyện và triển khai liên tục.
