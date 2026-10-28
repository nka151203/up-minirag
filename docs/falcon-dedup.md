# Gộp
- Điều kiện gộp: chỉ xét những thực thể cùng loại mới được gộp, ví dụ 1 thực thể person chỉ được gộp với 1 thực thể person khác chứ không được merge với 1 thực thể event.
# Gộp hiển nhiên
- Những cặp entity nào có tên chuẩn hóa (bỏ dấu, viết thường hết, bỏ kí tự đặc biệt) thì mặc định gộp.
- Những Những cặp entity nào có tên chuẩn hóa thuộc tập alternative_names (cũng chuẩn hóa các phần tử) thì cũng gộp
- Gộp xong nhớ phải extend phần alternative_names (Union Find)
# Không gộp hiển nhiên
Có sự xung đột thời gian của các loại thực thể
+ 2 Person có sự lệch nhau về birth_date và death_date (giả sử 2 thực thể không thể gộp nếu 1 người chết trước khi 1 người được sinh ra)
+ Các loại thực thể bạn hãy tự suy luận rule-base dựa trên MiniRAG/minirag/schema.py
# Falcon
- Lấy đồ thị là kết quả của Gộp hiển nhiên và Không gộp hiển nhiên (không xét 2 thực thể rõ ràng khác nhau).
Bước 1 — Multi-View Coarse Blocking
Cần thiết
Đây là bước tạo ứng viên hợp lý, nhưng bản thân blocking dựa trên embedding và topology không mới. Novelty nên nằm ở khả năng tìm được bí danh xa nhau về ngữ nghĩa mà vẫn kiểm soát được số ứng viên và duy trì recall.
Bước 2 — Counterfactual Trajectory Synthesis
Tiềm năng novelty cao
Đây là phần giàu tiềm năng nhất, nhưng phải định nghĩa được phép gộp giả định tạo ra mâu thuẫn mới nào. Nếu chỉ nối toàn bộ sự kiện rồi hỏi SLM timeline có hợp lý hay không, reviewer có thể xem đó là kiểm tra nhất quán bằng ngữ cảnh mở rộng, chưa phải counterfactual reasoning thực sự.
Bước 3 — SLM Falsification
Tiềm năng cao nếu có kiểm chứng
Ma trận thuận và phản chứng là hướng tốt, nhưng cần chứng minh SLM tìm được phản chứng thật, không chỉ phát hiện những điều nghe có vẻ bất hợp lý. Các loại mâu thuẫn phải có định nghĩa kiểm chứng được, kèm thời gian, nguồn và độ tin cậy.
Bước 4 — Signed Graph Partitioning
Không đủ mới nếu đứng riêng
Tối ưu phân hoạch trên đồ thị cạnh dương và âm đã có tiền lệ rõ ràng. Phần cần mới là cách xây dựng trọng số phản chứng có điều kiện theo phép gộp, và cách đưa những mâu thuẫn chỉ xuất hiện ở cấp cụm hoặc timeline vào mục tiêu tối ưu.

