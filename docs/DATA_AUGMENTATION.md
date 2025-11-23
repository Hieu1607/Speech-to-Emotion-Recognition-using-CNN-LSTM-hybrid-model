# 🎵 Data Augmentation trong Speech Emotion Recognition

Tăng cường dữ liệu âm thanh (Data Augmentation) là một kỹ thuật cực kỳ quan trọng trong bài toán Nhận diện Cảm xúc qua Lời nói (SER) để cải thiện hiệu suất và khả năng tổng quát hóa của mô hình.

SER thường đối mặt với vấn đề dữ liệu ít và mất cân bằng, đặc biệt là đối với các cảm xúc ít phổ biến hơn. Tăng cường dữ liệu giúp tạo ra các biến thể mới của dữ liệu gốc, mô phỏng các điều kiện môi trường thực tế và làm cho mô hình mạnh mẽ hơn (robust).

Dưới đây là các phương pháp tăng cường dữ liệu âm thanh phổ biến và hiệu quả nhất trong SER:

## 🚀 Các Phương pháp Tăng cường Dữ liệu Âm thanh

Các phương pháp này có thể được áp dụng trực tiếp lên tín hiệu âm thanh thô hoặc áp dụng lên biểu đồ Spectrogram (biểu diễn tần số-thời gian).

### 1. Tăng cường dựa trên Tín hiệu Âm thanh (Time Domain)

Các phương pháp này thay đổi trực tiếp tín hiệu dạng sóng (waveform) của file âm thanh:

| Phương pháp | Mô tả | Ứng dụng trong SER |
|-------------|-------|-------------------|
| **Thay đổi Tốc độ Phát (Time Stretching)** | Thay đổi tốc độ phát âm thanh (ví dụ: làm chậm 10%, làm nhanh 5%) mà không thay đổi cao độ (pitch). | Mô phỏng tốc độ nói khác nhau của người nói trong các trạng thái cảm xúc khác nhau (ví dụ: nói nhanh khi giận dữ/sợ hãi, nói chậm khi buồn). |
| **Thêm Nhiễu (Adding Noise)** | Thêm nhiễu ngẫu nhiên hoặc nhiễu nền thực tế (ví dụ: tiếng ồn trắng, tiếng ồn từ môi trường công cộng) vào tín hiệu gốc. | Tăng cường tính mạnh mẽ (robustness) của mô hình với môi trường thực tế có tạp âm. |
| **Thay đổi Biên độ (Changing Amplitude)** | Nhân toàn bộ tín hiệu với một hệ số ngẫu nhiên (ví dụ: từ 0.8 đến 1.2) để làm âm lượng to hơn hoặc nhỏ hơn. | Mô phỏng việc thu âm ở các khoảng cách khác nhau hoặc mức âm lượng khác nhau của người nói. |
| **Thay đổi Cao độ (Pitch Shifting)** | Tăng hoặc giảm cao độ (pitch) của âm thanh mà không thay đổi tốc độ. | Mô phỏng sự khác biệt giữa giọng nói nam/nữ, hoặc các trạng thái kích hoạt cảm xúc khác nhau (cao độ tăng khi giận dữ, giảm khi buồn). |

### 2. Tăng cường dựa trên Phổ (Frequency/Spectral Domain)

Các phương pháp này hoạt động trên biểu đồ Mel Spectrogram hoặc Log-Mel Spectrogram, thường được xử lý như một hình ảnh 2D:

| Phương pháp | Mô tả | Ứng dụng trong SER |
|-------------|-------|-------------------|
| **SpecAugment (Phương pháp phổ biến nhất)** | Gồm hai kỹ thuật chính được áp dụng ngẫu nhiên lên Spectrogram: | Ngăn chặn mô hình chỉ học các đặc trưng cục bộ quá cụ thể, buộc mô hình phải học các đặc trưng tổng quát hơn của cảm xúc. |
| a. **Masking Tần số (Frequency Masking)** | Che phủ (zero-out) một dải tần số liên tục. | Mô phỏng việc mất thông tin tần số hoặc biến thể âm sắc. |
| b. **Masking Thời gian (Time Masking)** | Che phủ (zero-out) một đoạn thời gian liên tục. | Mô phỏng việc mất thông tin tạm thời hoặc các quãng nghỉ không quan trọng. |
| **Sự thay đổi về tần số/thời gian ngẫu nhiên (Random Time/Frequency Shifts)** | Dịch chuyển ngẫu nhiên toàn bộ biểu đồ Spectrogram theo chiều tần số hoặc chiều thời gian. | Tương tự như thay đổi tốc độ/cao độ, giúp mô hình nhận biết cảm xúc độc lập với vị trí chính xác của feature trong phổ. |

---

## 💡 Lời khuyên Khi Áp dụng Tăng cường Dữ liệu

- **Kết hợp Đa dạng:** Bạn nên kết hợp nhiều phương pháp tăng cường khác nhau (ví dụ: thay đổi tốc độ + thêm nhiễu + SpecAugment) để tạo ra tập dữ liệu đa dạng nhất.

- **Giữ nguyên Nhãn (Label Preservation):** Điều quan trọng nhất là sau khi tăng cường, nhãn cảm xúc của file âm thanh mới phải không thay đổi. Ví dụ, việc thêm nhiễu không được làm thay đổi cảm xúc từ "vui" thành "giận dữ".

- **Tỷ lệ Tăng cường:** Không nên tạo quá nhiều mẫu tăng cường từ một mẫu gốc (ví dụ: chỉ nên tạo 3-5 biến thể mới cho mỗi mẫu gốc) để tránh việc mô hình bị học thuộc lòng các đặc điểm riêng của mẫu gốc.

- **Sử dụng Thư viện:** Các thư viện như `librosa` và `torchaudio` trong Python cung cấp các hàm dễ sử dụng để thực hiện hầu hết các kỹ thuật tăng cường này.
