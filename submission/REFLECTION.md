# Reflection — Lab 21

**1. Điều gì làm bạn ngạc nhiên nhất?**

Điều làm tôi ngạc nhiên nhất là fine-tune đạt target 0.970, cao hơn baseline prompt tốt 0.765, nhưng vẫn trượt regression gate vì điểm regression giảm 0.0689, vượt xa mức cho phép 0.020. Trước đó tôi dễ nghĩ điểm tác vụ chính tăng mạnh đồng nghĩa mô hình đã tốt hơn toàn diện. Kết quả này cho thấy cần xem cả năng lực mô hình làm mất đi, không chỉ năng lực nó học thêm. Tôi cũng bất ngờ khi `attn_only` với rank được cân bằng tham số đạt cùng target 0.970 như `correct`; trong thí nghiệm này, gắn adapter vào nhiều lớp hơn không tạo ra mức tăng target quan sát được.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Phần tốn công tính toán nhất là NB4: phải huấn luyện ba cấu hình đối chứng, mỗi cấu hình cùng ngân sách 30 bước. Theo `runs.csv`, ba run này mất khoảng 265–470 giây mỗi run; sau đó NB5 còn phải đánh giá các adapter. Tôi đã dự đoán phần train sẽ tốn thời gian, nhưng ban đầu chưa tính đủ thời gian cho việc chạy trọn bộ đánh giá và tải kết quả về khỏi runtime Colab. Việc kiểm tra môi trường cũng quan trọng: máy cá nhân có RTX 3050, nhưng PyTorch trong `.venv` là bản CPU nên không dùng được GPU từ môi trường đó; chạy pipeline đầy đủ trên Colab T4 phù hợp hơn.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng nghĩ chỉ cần train loss giảm và điểm phân loại tăng thì fine-tune đã thành công. Bốn run cho thấy train loss không đủ để xếp hạng chất lượng: `attn_only` có train loss thấp hơn `correct` nhưng chỉ hoà về target, còn `wrong_lr` có loss 1.5702 và target/format bằng 0 trong lần chạy này. Tôi cũng không còn mặc định rằng QLoRA luôn là lựa chọn tốt nhất khi thiếu VRAM. QLoRA giảm peak VRAM từ 8.78 GB xuống 3.86 GB, nhưng target thấp hơn `correct` 0.03 và latency cao hơn; đó là đánh đổi cần đo, không thể kết luận chỉ từ lượng bộ nhớ tiết kiệm.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI assistant để hiểu thứ tự chạy NB1–NB5, cách chọn `EVAL_LIMIT`, nơi lưu artefact Colab, và để hỗ trợ hoàn thiện `REPORT.md` cùng reflection này từ log thực tế. AI assistant ban đầu kết luận máy tôi không có CUDA vì PyTorch trong `.venv` báo `cuda_available=False`. Kết luận đó chưa đủ căn cứ: sau khi tôi nói máy có GPU, kiểm tra `nvidia-smi` xác nhận RTX 3050 và vấn đề thực sự là PyTorch CPU-only. Tôi đã yêu cầu đối chiếu lại bằng thông tin driver và sửa hướng dẫn sang dùng Colab T4. Bài học của tôi là phải xác minh môi trường và con số từ log/artefact trước khi tin kết luận do AI đưa ra.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Tôi sẽ bắt đầu bằng việc làm rõ yêu cầu đầu ra và tiêu chí thành công với khách hàng, đồng thời kiểm tra quyền sử dụng, dữ liệu nhạy cảm và cách loại bỏ thông tin định danh. Sau đó tôi sẽ tạo một tập train và tập đánh giá đại diện, giữ tập đánh giá riêng và đóng băng nó trước khi huấn luyện. Tôi sẽ đo baseline bằng prompt tốt trước, rồi mới quyết định fine-tune có cần thiết không. Nếu fine-tune, tôi sẽ theo dõi cả chất lượng tác vụ, định dạng, độ trễ và các năng lực tổng quát cần bảo toàn; không triển khai chỉ vì train loss giảm hoặc một chỉ số target tăng.
