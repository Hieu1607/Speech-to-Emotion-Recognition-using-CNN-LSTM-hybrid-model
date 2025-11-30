# 📝 Các Feature Chính trong Nhận diện Cảm xúc qua Lời nói (SER)

Trong bài toán nhận diện cảm xúc qua lời nói (Speech Emotion Recognition - SER), các đặc trưng (feature) được trích xuất từ tín hiệu âm thanh thô được chia thành 4 nhóm chính:

## 1. Nhóm Feature Tần số và Biên độ (Feature Cảm xúc Cơ bản - LLDs)

Đây là các đặc trưng vật lý cơ bản nhất của giọng nói:

- **Tần số cơ bản** ($\mathbf{F_0}$ / Pitch): Cao độ của giọng nói.
- **Năng lượng / Biên độ** (Energy / Amplitude): Độ lớn của giọng nói.
- **Thời gian** (Temporal Features): Tốc độ nói, thời gian dừng/nghỉ.

### Zero Crossing Rate (ZCR)
**Zero Crossing Rate** là tần suất mà tín hiệu âm thanh thay đổi dấu (từ dương sang âm hoặc ngược lại) trong một khoảng thời gian nhất định. 

- **Ý nghĩa**: ZCR cao thường biểu thị âm thanh có nhiều tần số cao (như âm /s/, /f/), trong khi ZCR thấp biểu thị âm thanh có nhiều tần số thấp (như nguyên âm).
- **Ứng dụng trong SER**: Các cảm xúc khác nhau có thể tạo ra các đặc điểm ZCR khác nhau. Ví dụ, giận dữ có thể tạo ra nhiều âm sắc cao hơn, dẫn đến ZCR cao hơn.

### Root Mean Square Energy (RMSE)
**Root Mean Square Energy** đo lường năng lượng trung bình của tín hiệu âm thanh, phản ánh độ "mạnh" hoặc "to" của giọng nói.

- **Công thức**: $RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}$ 
- **Ý nghĩa**: RMSE cao có nghĩa là giọng nói to và mạnh mẽ, RMSE thấp có nghĩa là giọng nói nhỏ và yếu.
- **Ứng dụng trong SER**: Các cảm xúc như giận dữ, vui vẻ thường có RMSE cao hơn so với buồn bã, sợ hãi.

## 2. Nhóm Feature Phổ (Spectral Features)

Các đặc trưng mô tả sự phân bố năng lượng của tín hiệu theo tần số, liên quan đến âm sắc (timbre) và màu sắc giọng nói.

## 3. Nhóm Feature Chất lượng Giọng nói (Voice Quality Features)

Các đặc trưng liên quan đến độ thô ráp, rè, hoặc rung của giọng nói (ví dụ: Jitter, Shimmer, Harmonic-to-Noise Ratio).

## 4. Nhóm Feature Cao cấp (High-Level Statistical Functionals)

Các chỉ số thống kê (trung bình, phương sai, cực đại, cực tiểu, phân vị...) được tính toán trên các feature của nhóm 1 và 2 trong một khoảng thời gian dài (toàn bộ phát ngôn).

---

## 🌟 Nhóm Feature Phổ (Spectral Features) và Tính Ứng dụng Rộng rãi

Nhóm Feature Phổ được sử dụng rộng rãi và được coi là hiệu quả nhất trong SER vì chúng đại diện cho các đặc điểm vật lý ổn định của bộ máy phát âm, ít bị ảnh hưởng bởi nội dung ngôn ngữ cụ thể.

---

## 🎤 Mel-Frequency Cepstral Coefficients (MFCCs)

MFCCs là feature phổ biến nhất, được sử dụng trong cả SER và nhận dạng giọng nói (ASR).

| Khái niệm | Giải thích về ứng dụng rộng rãi |
|-----------|--------------------------------|
| **Mô phỏng tai người** | MFCCs được tính toán bằng cách áp dụng thang đo Mel (một thang đo phi tuyến tính), mô phỏng cách tai người xử lý âm thanh (nhạy cảm hơn với các tần số thấp). |
| **Đại diện Âm sắc** | MFCCs cô đọng thông tin về hình dạng phổ của âm thanh (phong bao phổ), phản ánh hình dạng của ống thanh. Hình dạng ống thanh thay đổi tinh tế theo cảm xúc, khiến MFCCs trở thành chỉ báo mạnh mẽ cho sự khác biệt cảm xúc. |
| **Hiệu quả về mặt tính toán** | Là một tập hợp các hệ số nhỏ gọn (thường là 12-13 hệ số) nhưng chứa đựng hầu hết thông tin cần thiết về âm thanh. |

---

## 📈 Các Feature Phổ Khác

Các đặc trưng phổ khác như **Spectral Centroid** (Trọng tâm phổ) và **Spectral Roll-off** cũng rất quan trọng vì chúng là chỉ số trực tiếp của sự phân bố năng lượng. Sự dịch chuyển năng lượng sang tần số cao hoặc thấp là một phản ứng vật lý với cảm xúc (ví dụ: giọng nói sáng hơn, cao hơn khi vui vẻ/giận dữ).

---

## 🖼️ Nhóm Feature Dựa trên Hình ảnh (Spectrogram-based) và Sự Phát triển của Deep Learning

Trong những năm gần đây, sự kết hợp của SER với các mô hình Deep Learning (đặc biệt là CNN và Attention) đã thúc đẩy việc sử dụng các biểu diễn tín hiệu lời nói dưới dạng hình ảnh.

### Mel Spectrogram

**Mô tả:** Là biểu đồ 2D trực quan hóa cường độ (năng lượng) của các tần số theo thời gian. Trục Y là tần số (thường trên thang Mel), trục X là thời gian, và màu sắc/độ sáng là năng lượng.

**Tại sao được sử dụng rộng rãi:**

- **Bảo toàn Thông tin:** Spectrogram giữ lại cả thông tin về tần số và thời gian của tín hiệu lời nói, điều mà các vector feature cố định (MFCCs) không thể làm được một cách trọn vẹn.

- **Sức mạnh của CNN:** Khi biểu diễn dưới dạng hình ảnh, các mô hình Mạng nơ-ron tích chập (CNN) có thể được áp dụng. CNN rất giỏi trong việc tìm kiếm các mẫu hình không gian cục bộ (Local Spatial Patterns) — trong ngữ cảnh này, chúng tìm kiếm các cấu trúc cảm xúc tinh tế về cách tần số thay đổi qua các khung thời gian gần nhau (chính là sự thay đổi của cao độ, âm sắc, và năng lượng).

- **Tích hợp đa chiều:** Spectrogram là cách hiệu quả để mô hình xử lý một cách tự nhiên đồng thời thông tin tần số và thời gian, cho phép mạng học các mối quan hệ phức tạp giữa chúng để phân loại cảm xúc.