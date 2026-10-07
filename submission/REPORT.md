# Lab 21 — Báo cáo đánh giá fine-tuning

**Họ tên:** Nguyễn Danh Gia Minh
**MSSV:** 2A202602441
**Ngày chạy:** 2026-10-07
**Tier:** T4
**Base model:** `unsloth/Qwen3.5-4B`
**GPU thực tế:** NVIDIA T4 trên Google Colab

## 1. Thiết lập

| Mục | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt, đầu ra JSON triage |
| Train / validation | 225 / 25, split seed 42 |
| Tập đánh giá | 50 target và 15 regression |
| `max_length` | 1024 theo tier; p95 đo được là 98 token, p99 là 100, gợi ý làm tròn là 256 |
| `MASK_MODE` | `assistant-only` |
| Epochs / optimizer steps | 2 / 30 |

Tokenizer giữ lại khối `<think>` trong chat template. Mask proof có 39/94 token được giám sát (0.4149); câu trả lời nằm trong loss và phần câu hỏi không nằm trong loss. Tập dữ liệu huấn luyện mặc định có câu trả lời JSON thuần, không có reasoning trace.

## 2. Bằng chứng mask — NB1

`results/mask_proof.json` xác nhận:

| Kiểm tra | Kết quả |
|---|---:|
| Tỷ lệ token được giám sát | 0.4149 |
| Câu trả lời nằm trong loss | Có |
| Câu hỏi không nằm trong loss | Có |

Phần được tính loss bắt đầu ở cuối lượt assistant (`</think>` đóng rỗng trên mẫu này), bao gồm JSON trả lời và token kết thúc lượt. Phần user chứa câu hỏi không được tính loss. `results/token_stats.json` ghi p95 = 98 và `suggested_max_length` = 256; pipeline vẫn dùng 1024 theo tier T4, nên có thể xem xét giảm xuống 256 trong một lần chạy mới, sau khi xác minh không bị cắt mẫu.

## 3. Baselines và kết quả — NB2, NB5

Baseline được đo và đóng băng trước khi huấn luyện. `EVAL_LIMIT` không được bật; đánh giá dùng đủ 50 mẫu target.

| Run | Target | Regression | Format | Latency (ms/mẫu) |
|---|---:|---:|---:|---:|
| (a) Base + naive prompt | 0.000 | 0.7911 | 0.000 | 3331.6 |
| (b) Base + optimized prompt | 0.765 | 0.7911 | 1.000 | 999.8 |
| (c) LoRA fine-tune + naive prompt | 0.970 | 0.7222 | 1.000 | 1389.4 |

Baseline (b) cao hơn rõ rệt baseline (a) về target và format, nên đây là mốc so sánh có ý nghĩa. Fine-tune cải thiện target so với (b) **0.205**, nhưng regression giảm **0.0689** (xấp xỉ 0.069), vượt mức suy giảm cho phép 0.020. Format vẫn đạt 1.000; latency của fine-tune cao hơn baseline (b) khoảng 389.6 ms/mẫu. `valid_trace_rate` là 0.0; vì corpus huấn luyện không chứa reasoning trace, con số này không được diễn giải như bằng chứng về hiện tượng sụp đổ reasoning.

## 4. Giải phẫu các cấu hình — NB4

Cả bốn run dùng 30 optimizer steps. `attn_only` được tăng rank để khớp số tham số huấn luyện với `correct` (sai lệch dưới 5%).

| Run | Placement | Rank | Trainable params | Learning rate | Train loss | Target | Format | Latency (ms) | Peak VRAM (GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6293* | 0.970 | 1.000 | 1389.4 | 8.78 |
| `attn_only` | attention q/v | 283 | 32,456,704 | 1e-4 | 0.5377 | 0.970 | 1.000 | 893.0 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | 0.000 | 0.000 | 5243.2 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 1.000 | 1761.6 | 3.86 |

\* `runs.csv` có hai dòng `correct`: loss 0.6264 (401.3 giây) và 0.6293 (418.0 giây). Bảng dùng dòng ghi sau cùng; cần giữ cả file CSV để thể hiện lịch sử này. QLoRA dùng ít hơn khoảng 4.92 GB VRAM (xấp xỉ 56%) so với `correct`.

`attn_only` hoà `correct` ở target (0.970), dù có train loss thấp hơn; kết quả này không cho thấy lợi thế target từ việc gắn adapter vào mọi lớp linear trong thí nghiệm này. `wrong_lr` đạt target và format 0.000, đồng thời có train loss cao nhất (1.5702), phù hợp với việc LR 1e-5 quá thấp cho cấu hình LoRA này trong ngân sách 30 step. `qlora` giảm peak VRAM từ 8.78 xuống 3.86 GB (tiết kiệm khoảng 56%), nhưng target thấp hơn `correct` 0.030 và latency cao hơn (1761.6 so với 1389.4 ms/mẫu). Đây là một đánh đổi đo được; không thể kết luận chất lượng chỉ từ train loss.

## 5. Phán quyết — NB5

**Regression gate: FAILED**
**Target Δ:** +0.205
**Regression Δ:** -0.0689
**Ngưỡng suy giảm regression:** tối đa 0.020
**Valid trace rate:** 0.0

Fine-tune đã học tốt hơn tác vụ ticket: target tăng từ 0.765 lên 0.970 và format giữ ở 1.000. Tuy nhiên, mô hình cũng làm giảm điểm regression từ 0.7911 xuống 0.7222. Mức giảm 0.0689 lớn hơn khoảng ba lần ngưỡng 0.020, vì vậy không thể xem đây là một mô hình đạt yêu cầu chỉ dựa trên điểm target. Đây là đánh đổi giữa chuyên môn hóa và khả năng tổng quát, không phải lỗi thực thi pipeline: gate đã phát hiện đúng sự suy giảm và giữ nguyên phán quyết thay vì nới ngưỡng sau khi nhìn thấy kết quả. Cần thử replay một phần nhỏ dữ liệu regression/kiến thức tổng quát trong huấn luyện, rồi đánh giá lại trên cùng các tập đã đóng băng. Không nên triển khai adapter hiện tại nguyên trạng cho hệ thống cần duy trì cả hai năng lực.

## 6. Phân tích định tính và giới hạn artefact

`results/qualitative.json` lưu 50 ticket cùng điểm và phần đầu dự đoán fine-tune; các mẫu có điểm 1.0 là đúng cả bốn trường, còn điểm 0.75 là đúng ba trong bốn trường. Tệp này không lưu dự đoán baseline (b) theo từng ticket và cắt ngắn chuỗi dự đoán, nên không thể xác định từ artefact hiện tại chính xác từng ticket mà fine-tune thắng/thua baseline (b), hoặc chỉ ra trường sai ở các mẫu 0.75. Nếu rubric yêu cầu ví dụ thắng/thua theo cặp, cần chạy thêm một lượt sinh baseline (b) trên cùng 50 ticket và lưu dự đoán đầy đủ để đối chiếu; không nên suy diễn các trường hợp đó từ điểm tổng.

## 7. Kết luận và điều học được

Chưa nên triển khai bản fine-tune này nguyên trạng. Kết quả target 0.970 là một cải thiện đáng kể so với baseline (b) 0.765, đồng thời đầu ra vẫn đạt format JSON 1.000. Nhưng mục tiêu của thí nghiệm không chỉ là tối ưu phân loại ticket: regression giảm 0.0689, trong khi lab chỉ cho phép giảm tối đa 0.020. Do đó adapter chưa đáp ứng cổng hồi quy dù đã thành công trên tác vụ đích. Đánh giá công bằng cũng cho thấy prompt tốt là một baseline mạnh: baseline (b) đạt 0.765, cao hơn nhiều so với baseline (a) 0.000, vì thế kết luận được đưa ra so với mốc đã đóng băng chứ không phải prompt ngây thơ. QLoRA giảm VRAM khoảng 56%, nhưng hiệu quả chất lượng cần đọc từ autopsy trước khi quyết định đánh đổi; train loss thấp hơn tự nó không chứng minh chất lượng tốt hơn. Bước tiếp theo hợp lý là thử thêm replay data ở tỷ lệ nhỏ, giữ nguyên eval và ngưỡng, rồi chạy lại cả đánh giá target lẫn regression. Chỉ triển khai nếu lần chạy mới cải thiện hoặc giữ target mà đồng thời đưa regression vào giới hạn cho phép.

Ba điều rút ra từ lần chạy này:

1. Phải so fine-tune với baseline đã được prompt tốt và đóng băng trước huấn luyện.
2. Điểm target cao không đủ nếu mô hình làm suy giảm năng lực tổng quát vượt ngưỡng.
3. Mask được xác minh trước train; thay đổi dữ liệu replay cần được đánh giá lại bằng cùng phép đo, không bằng train loss.

**Nếu có thêm hai giờ:** thử replay 1–5% dữ liệu tổng quát, giữ nguyên tập eval và so sánh lại bốn nhóm điểm; sau đó hoàn thiện bảng ví dụ định tính từ các artefact dự đoán theo từng mẫu.
