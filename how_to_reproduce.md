# Hướng dẫn reproduce MiniRAG

Tài liệu này mô tả cách chạy lại pipeline của **MiniRAG** trong repo này khi dùng **một endpoint OpenAI-compatible duy nhất** cho phần sinh văn bản: chuẩn bị môi trường → index dataset LiHua-World → hỏi đáp (QA) → đánh giá.

Toàn bộ lệnh dưới đây được chạy **từ thư mục gốc của repo**:

```bash
cd /home/nka15/projects/tkg/MiniRAG
```

> Cấu hình cần điền chỉ gồm 3 thông tin: **base URL**, **API key**, **tên model**. Embedding mặc định vẫn chạy local bằng `all-MiniLM-L6-v2` (giống paper); có thể chuyển embedding sang chính endpoint đó nếu muốn.

---

## 1. Dataset

### 1.1. Đường dẫn

Dataset duy nhất đi kèm repo là **LiHua-World**:

| Thành phần | Đường dẫn | Vai trò |
|---|---|---|
| Gốc dataset | `dataset/LiHua-World/` | Thư mục chứa toàn bộ dataset |
| Dữ liệu chat gốc (nén) | `dataset/LiHua-World/data/LiHuaWorld.zip` | File nén ~548 KB chứa 442 file `.txt` |
| Dữ liệu chat gốc (sau giải nén) | `dataset/LiHua-World/data/LiHua-World/{week1…week52, bakery_deliver}/*.txt` | Knowledge base đưa vào `rag.insert()` |
| Câu hỏi + đáp án | `dataset/LiHua-World/qa/query_set.csv` | 637 câu hỏi kèm Gold Answer, Evidence, Type |
| Câu hỏi dạng JSON | `dataset/LiHua-World/qa/query_set.json` | Bản JSON của `query_set.csv` (637 object) |
| Mô tả dataset | `dataset/LiHua-World/README.md` | Mô tả gốc của tác giả |
| Notebook đánh giá | `dataset/LiHua-World/evaluation.ipynb` | Chấm điểm đáp án bằng LLM judge |

### 1.2. Đặc điểm dataset

LiHua-World là dataset tiếng Anh mô phỏng **một năm chat record của user ảo tên LiHua**, thiết kế cho kịch bản RAG on-device.

| Đặc điểm | Giá trị thực tế trong repo |
|---|---|
| Ngôn ngữ | Tiếng Anh |
| Khoảng thời gian | 2026-01-05 → 2026-12-31 |
| Số file chat | **442 file `.txt`** sau khi giải nén |
| Tổ chức thư mục | 52 thư mục tuần (`week1`…`week52`) + 1 thư mục `bakery_deliver/` |
| Dung lượng | ~978 KB (chưa nén), 548 KB (zip) |
| Số câu hỏi | **637** |
| Phân loại câu hỏi (cột `Type`) | **506 `Single`**, **66 `Multi`**, **65 `Null`** |
| Câu hỏi `Null` | Gold Answer = `"Insufficient information"` (câu không trả lời được từ dữ liệu) |
| Evidence | 563 câu dạng timestamp `YYYYMMDD_HH:MM`, 65 câu `N/A`, một vài giá trị không đồng nhất |

**Định dạng file chat** (ví dụ `week38/20260923_1600.txt`):

```text
Time: 20260923_16:00
WolfgangSchulz: Hey! Wanna hit the grocery store after work today? ...
LiHua: Sounds great! I could use some snacks for the weekend. ...
WolfgangSchulz: How about around 5:30? ...
```

- Dòng đầu là timestamp `Time: YYYYMMDD_HH:MM`.
- Các dòng sau có dạng `Sender: nội dung`, có thể có nhiều người tham gia và emoji.
- Tên file = `YYYYMMDD_HHMM.txt` (bỏ dấu `:` so với timestamp trong nội dung).
- Thư mục `bakery_deliver/` chứa 17 file đặt tên theo ngày `YYYYMMDD.txt`, không theo tuần.

**Định dạng file QA** (`qa/query_set.csv`):

| Cột | Ý nghĩa | Ví dụ |
|---|---|---|
| `Question` | Câu hỏi | `What time does Li Hua check in with Adam about moving in?` |
| `Gold Answer` | Đáp án chuẩn | `5:30 PM` |
| `Evidence` | Các timestamp chứng cứ, nối bằng `<and>` | `20260121_10:00<and>20260701_10:00` |
| `Type` | Loại câu hỏi (`Single` / `Multi` / `Null`) | `Multi` |

> Lưu ý: README gốc mô tả 3 loại là Single-hop / Multi-hop / Summary, nhưng CSV thực tế dùng nhãn `Single` / `Multi` / `Null`. Evidence dạng `20260121_10:00` tương ứng file `20260121_1000.txt`.

### 1.3. Điều kiện tiên quyết về dataset

Script index tìm file `.txt` bằng `os.walk()` trên `--datapath`, nên **phải giải nén zip trước khi index**. Nếu để nguyên `.zip`, chương trình sẽ không tìm thấy dữ liệu và index 0 file.

---

## 2. Cấu hình endpoint OpenAI-compatible

Repo đã được rút gọn để **chỉ dùng endpoint OpenAI-compatible cho phần LLM**. Các provider khác (`ollama`, `lollms`, `azure_openai`, `bedrock`, `jina`, `lmdeploy`, `nvidia_openai`, `siliconcloud`, `zhipu`) đã bị xóa khỏi `minirag/llm/`.

### 2.1. Ba thông tin bắt buộc

| Biến | Ý nghĩa | Ví dụ |
|---|---|---|
| `OPENAI_API_BASE` | Base URL của endpoint | `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | API key | `sk-...` |
| `LLM_MODEL` | Tên model do endpoint cung cấp | `gpt-4o-mini`, `deepseek-chat`, `qwen2.5-7b-instruct` |

Có thể truyền qua file `.env`, biến môi trường, hoặc tham số CLI:

| CLI | Biến môi trường / `.env` | Mặc định |
|---|---|---|
| `--base-url` | `OPENAI_API_BASE` | — (bắt buộc) |
| `--api-key` | `OPENAI_API_KEY` | — (bắt buộc) |
| `--model` | `LLM_MODEL` | — (bắt buộc) |
| `--embedding-backend` | `EMBEDDING_BACKEND` | `hf` |
| `--embedding-model` | `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` (hf) / `text-embedding-3-small` (openai) |
| `--embedding-dim` | `EMBEDDING_DIM` | `384` (hf) / `1536` (openai) |

Thứ tự ưu tiên: **tham số CLI > biến môi trường / `.env`**.

### 2.2. File `.env`

```bash
cp .env.example .env
```

Mở `.env` và điền:

```dotenv
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=sk-your-key
LLM_MODEL=gpt-4o-mini
```

> File `.env` đã nằm trong `.gitignore`; không commit key.

### 2.3. Embedding

| Backend | Cách chạy | Yêu cầu |
|---|---|---|
| `hf` (mặc định) | `all-MiniLM-L6-v2` chạy local, đúng như paper | `torch` + `transformers`; chạy CPU được, không cần GPU |
| `openai` | Gọi `/embeddings` của chính endpoint trên | Endpoint phải hỗ trợ embeddings; phải đặt đúng `--embedding-dim` |

Ví dụ dùng embedding qua endpoint (bỏ comment trong `.env`):

```dotenv
EMBEDDING_BACKEND=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
```

> **Quan trọng:** cấu hình embedding phải giống nhau giữa bước index và bước QA. Nếu đổi backend/model/dim, phải xóa `./LiHua-World` và index lại từ đầu.

---

## 3. Yêu cầu môi trường

| Thành phần | Yêu cầu / ghi chú |
|---|---|
| Python | `>= 3.9` (repo khai báo trong `setup.py`; máy hiện tại có Python 3.12) |
| Python packages | `pip install -e .`; thêm `torch transformers` nếu dùng embedding `hf` (mặc định) |
| Endpoint | Một endpoint OpenAI-compatible cho chat completions; thêm `/embeddings` nếu dùng `--embedding-backend openai` |
| GPU | **Không bắt buộc**. LLM chạy qua API; embedding `hf` chạy CPU. |
| Đĩa cứng | ~100 MB cho embedding model HF (nếu dùng), cộng thư mục index `./LiHua-World` |

> `openai`, `transformers`, `torch` được `pipmaster` tự cài khi import nếu thiếu, nhưng nên cài chủ động để kiểm soát phiên bản.

---

## 4. Chuẩn bị môi trường

```bash
cd /home/nka15/projects/tkg/MiniRAG
```

### 4.1. Tạo virtualenv và cài package

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install torch transformers
```

- `pip install -e .`: cài MiniRAG ở chế độ editable theo `setup.py` + `requirements.txt`.
- `pip install torch transformers`: chỉ cần cho embedding `hf` (mặc định). Nếu dùng `--embedding-backend openai`, có thể bỏ qua.

### 4.2. Cấu hình endpoint

```bash
cp .env.example .env
# sửa .env: OPENAI_API_BASE, OPENAI_API_KEY, LLM_MODEL
```

### 4.3. Giải nén dataset

```bash
unzip -o dataset/LiHua-World/data/LiHuaWorld.zip -d dataset/LiHua-World/data/
find dataset/LiHua-World/data -name '*.txt' | wc -l
```

- Lệnh 1 giải nén chat record vào `dataset/LiHua-World/data/LiHua-World/`.
- Lệnh 2 kiểm tra: kết quả đúng phải là **442**.

### 4.4. Tạo thư mục output (không bắt buộc)

```bash
mkdir -p logs
```

- `Step_1_QA.py` đã tự tạo thư mục cha của `--outputpath`; lệnh này chỉ để chắc chắn.

---

## 5. Bước 0 — Index dataset

### Lệnh

Nếu đã điền `.env`:

```bash
python3 reproduce/Step_0_index.py \
  --workingdir ./LiHua-World \
  --datapath ./dataset/LiHua-World/data/
```

Hoặc truyền trực tiếp qua CLI:

```bash
python3 reproduce/Step_0_index.py \
  --base-url https://api.openai.com/v1 \
  --api-key sk-... \
  --model gpt-4o-mini \
  --workingdir ./LiHua-World \
  --datapath ./dataset/LiHua-World/data/
```

### Vai trò

Đọc tất cả file `.txt` dưới `--datapath`, gọi `rag.insert()` cho từng file để xây **heterogeneous graph** (chunk + entity) và các vector store. Kết quả được lưu vào `--workingdir`.

### Ý nghĩa các tham số

| Tham số | Mặc định | Vai trò |
|---|---|---|
| `--base-url` | `$OPENAI_API_BASE` | Base URL endpoint OpenAI-compatible |
| `--api-key` | `$OPENAI_API_KEY` | API key |
| `--model` | `$LLM_MODEL` | Model dùng để trích xuất entity/relationship khi index |
| `--embedding-backend` | `hf` | `hf` = local, `openai` = dùng endpoint |
| `--embedding-model` | phụ thuộc backend | Model embedding |
| `--embedding-dim` | phụ thuộc backend | Số chiều embedding |
| `--workingdir` | `./LiHua-World` | Thư mục lưu graph + KV + vector DB + cache. Phải dùng lại đúng thư mục này ở bước QA |
| `--datapath` | `./dataset/LiHua-World/data/` | Thư mục chứa file `.txt` (tìm đệ quy) |
| `--querypath` | — | Không dùng trong bước index |
| `--outputpath` | — | Không dùng trong bước index |

### Hành vi cần biết

- **Model index chính là model bạn cấu hình** (không còn hardcode `gpt-4o-mini` như code gốc). Vì vậy kết quả có thể khác bảng trong paper nếu dùng model khác.
- Embedding mặc định `sentence-transformers/all-MiniLM-L6-v2` (384 chiều), chạy local trên CPU.
- Chunking: `chunk_token_size=1200`, `chunk_overlap_token_size=100`.
- Script gọi `insert()` **442 lần** (mỗi file một lần) → số lượng request lớn, thời gian chạy tùy tốc độ endpoint.
- Có cơ chế dedup theo hash nội dung và `llm_response_cache`, nên chạy lại sẽ bỏ qua document đã xử lý; tuy nhiên mỗi lần `insert()` vẫn quét lại các document `PROCESSED` để trích xuất entity bổ sung. Muốn index sạch hoàn toàn thì xóa `./LiHua-World`.
- Sau khi xong, trong `./LiHua-World` sẽ có các file dạng:
  - `graph_chunk_entity_relation.graphml`
  - `kv_store_full_docs.json`, `kv_store_text_chunks.json`, `kv_store_llm_response_cache.json`, `kv_store_doc_status.json`
  - `vdb_entities.json`, `vdb_entities_name.json`, `vdb_relationships.json`, `vdb_chunks.json`
  - `minirag.log`

### 5.1. Thống kê knowledge graph sau khi index

```bash
python3 reproduce/Step_0_A_graph_statistic.py --workingdir ./LiHua-World
```

- Vai trò: đọc `graph_chunk_entity_relation.graphml` và in số **node**, số **cạnh**, density, số connected component, số node cô lập, phân bố `entity_type` và các node có degree cao nhất.
- Có thể trỏ trực tiếp file graph bằng `--graph-file /đường/dẫn/graph_chunk_entity_relation.graphml`.
- Script chỉ đọc và in thông tin, không sửa graph.

---

## 6. Bước 1 — Hỏi đáp trên tập câu hỏi

### Lệnh

```bash
python3 reproduce/Step_1_QA.py \
  --workingdir ./LiHua-World \
  --querypath ./dataset/LiHua-World/qa/query_set.csv \
  --outputpath ./logs/default_output.csv
```

Nếu chưa điền `.env`, thêm `--base-url`, `--api-key`, `--model` giống bước index.

### Vai trò

Load lại index từ `--workingdir`, đọc các câu hỏi trong `--querypath`, gọi `rag.query(question, param=QueryParam(mode="mini"))` cho từng câu, rồi ghi kết quả ra CSV.

### Ý nghĩa các tham số

| Tham số | Mặc định | Vai trò |
|---|---|---|
| `--workingdir` | `./LiHua-World` | Phải trùng với thư mục đã index ở Bước 0 |
| `--querypath` | `./dataset/LiHua-World/qa/query_set.csv` | File CSV câu hỏi (đọc cột `Question`, `Gold Answer`) |
| `--outputpath` | `./logs/Default_output.csv` | File CSV kết quả |
| `--base-url`, `--api-key`, `--model` | từ `.env` | Endpoint/model dùng để trả lời |
| `--embedding-*` | từ `.env` / mặc định | Phải giống bước index |
| `--datapath` | — | Không dùng trong bước QA |

### Hành vi cần biết

- Toàn bộ LLM (trích keyword + sinh câu trả lời) đi qua endpoint OpenAI-compatible; không tải model local, không cần CUDA.
- Chế độ truy vấn là `mini` — MiniRAG dùng keyword/type extraction + graph + chunk retrieval.
- Output có đúng 3 cột: `Question`, `Gold Answer`, `minirag`.
- Script **có thể chạy tiếp (resume)**: nếu file output đã tồn tại, nó đếm số dòng đã có và bắt đầu từ câu tiếp theo. Có thể dừng giữa chừng rồi chạy lại cùng lệnh.
- Khi hoàn tất, file có **638 dòng** (1 header + 637 câu hỏi).
- Mỗi câu có thể phát sinh nhiều request (trích keyword + sinh câu trả lời), tùy tốc độ endpoint.

---

## 7. Bước 2 — Đánh giá kết quả

### 7.1. Chạy notebook

```bash
jupyter notebook dataset/LiHua-World/evaluation.ipynb
```

### Vai trò

Notebook dùng một **LLM judge** (`gpt-4o` hoặc `deepseek-chat`) để chấm từng đáp án theo thang:

- `1` = đúng
- `0` = không liên quan
- `-1` = sai

Sau đó tính tỉ lệ trên toàn bộ 637 câu. Hai chỉ số tương ứng bảng trong README:

- **acc** = `count(Score == 1) / N`
- **err** = `count(Score == -1) / N`

### 7.2. Chuẩn bị input cho notebook

Notebook đọc một file CSV (biến `ANA_FILE_PATH`, mặc định `./mthp_output.csv`) và yêu cầu **đủ 5 cột**:

```text
Question, Gold Answer, naive, lightrag, minirag
```

File do `Step_1_QA.py` tạo chỉ có `Question, Gold Answer, minirag`. Nếu chỉ muốn chấm MiniRAG, có thể tạo file tương thích bằng đoạn sau (điền rỗng 2 cột baseline, chỉ quan tâm `Score3`):

```bash
python3 - <<'PY'
import csv

src = "logs/default_output.csv"
dst = "logs/eval_input.csv"

with open(src, encoding="utf-8") as f, open(dst, "w", newline="", encoding="utf-8") as g:
    reader = csv.DictReader(f)
    writer = csv.DictWriter(
        g, fieldnames=["Question", "Gold Answer", "naive", "lightrag", "minirag"]
    )
    writer.writeheader()
    for row in reader:
        writer.writerow(
            {
                "Question": row["Question"],
                "Gold Answer": row["Gold Answer"],
                "naive": "",
                "lightrag": "",
                "minirag": row["minirag"],
            }
        )
print("Wrote", dst)
PY
```

Sau đó sửa `ANA_FILE_PATH` trong notebook thành `./logs/eval_input.csv`.

### 7.3. Các chỗ cần sửa trong notebook trước khi chạy

| Vị trí | Cần làm |
|---|---|
| Cell đầu | Thay `AAA` bằng API key thật, hoặc bỏ dòng `os.environ["OPENAI_API_KEY"] = AAA` và export biến môi trường từ shell |
| `ANA_FILE_PATH` | Trỏ tới CSV đầu vào đã chuẩn bị ở 7.2 |
| Cell DeepSeek | Thay `My_deepseek_key` bằng key thật |
| Cell OpenAI | Cần `OPENAI_API_KEY` hợp lệ; model dùng là `gpt-4o` |
| Phần đọc CSV | Notebook đọc các cột `naive`, `lightrag`, `minirag`; thiếu cột sẽ lỗi `KeyError` |

> Lưu ý: repo chỉ cung cấp code của **MiniRAG**, không có code chạy baseline NaiveRAG/LightRAG. Muốn tái lập đầy đủ bảng trong README (gồm cả baseline) cần chạy thêm các phương pháp đó hoặc chuẩn bị sẵn cột `naive`/`lightrag`.

---

## 8. (Tuỳ chọn) Chạy nhanh end-to-end và các chế độ khác

### 8.1. Smoke test bằng `main.py`

```bash
python3 main.py --workingdir ./LiHua-World --datapath ./dataset/LiHua-World/data/
```

- Vai trò: index toàn bộ dữ liệu rồi chạy **một câu hỏi toy** và in câu trả lời ra màn hình. Phù hợp để kiểm tra nhanh endpoint/môi trường.
- Lưu ý: `main.py` cũng gọi `rag.insert()` cho toàn bộ 442 file, nên nếu đã index rồi thì dùng `--workingdir` riêng hoặc bỏ qua bước này.

### 8.2. Chế độ truy vấn khác

MiniRAG hỗ trợ 3 mode qua `QueryParam(mode=...)`:

| Mode | Vai trò |
|---|---|
| `mini` | Mặc định, dùng trong paper: keyword/type-aware + graph + chunk |
| `light` | Truy vấn graph kiểu LightRAG |
| `naive` | Chỉ vector search trên chunk, không dùng graph |

Có thể đổi mode trong `Step_1_QA.py` tại dòng `rag.query(..., param=QueryParam(mode="mini"))`.

### 8.3. API server (tuỳ chọn)

```bash
pip install -e ".[api]"
minirag-server \
  --llm-binding openai \
  --llm-binding-host https://api.openai.com/v1 \
  --llm-binding-api-key sk-... \
  --llm-model gpt-4o-mini \
  --embedding-binding openai \
  --embedding-binding-host https://api.openai.com/v1 \
  --embedding-binding-api-key sk-... \
  --embedding-model text-embedding-3-small \
  --embedding-dim 1536
```

- Vai trò: chạy FastAPI server của MiniRAG.
- Sau khi dọn provider, server chỉ chấp nhận binding `openai` cho cả LLM và embedding.
- Nhớ đặt `--embedding-dim` đúng với model (ví dụ `text-embedding-3-small` = 1536).

### 8.4. Docker (tuỳ chọn)

```bash
docker compose up --build
```

- Vai trò: build và chạy MiniRAG dưới dạng container.
- Cảnh báo: `Dockerfile` có `COPY .env .` nên cần tạo file `.env` trước khi build. Port mặc định của server là `9721` (khớp với `docker-compose.yml`).

---

## 9. Kiểm tra kết quả và dọn dẹp

```bash
# Kiểm tra số file dữ liệu
find dataset/LiHua-World/data -name '*.txt' | wc -l        # mong đợi: 442

# Kiểm tra index đã tạo
ls -lh LiHua-World/

# Xem 3 dòng đầu kết quả QA
head -3 logs/default_output.csv

# Đếm số dòng kết quả (header + 637 câu)
wc -l logs/default_output.csv                               # mong đợi: 638

# Chạy lại index từ đầu
rm -rf ./LiHua-World

# Chạy lại QA từ đầu
rm -f logs/default_output.csv
```

---

## 10. Lỗi thường gặp

| Lỗi / triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `Thiếu cấu hình: --base-url hoặc OPENAI_API_BASE, ...` | Chưa điền đủ 3 thông tin endpoint | Điền `.env` hoặc truyền `--base-url`, `--api-key`, `--model` |
| Index 0 file, chạy rất nhanh rồi kết thúc | Chưa giải nén `LiHuaWorld.zip` | Chạy lại lệnh ở mục 4.3, kiểm tra đủ 442 `.txt` |
| `401 Unauthorized` / `403 Forbidden` | Sai hoặc thiếu API key | Kiểm tra `OPENAI_API_KEY` / `--api-key` |
| `404 model not found` | Sai tên model | Kiểm tra `LLM_MODEL` / `--model` đúng như endpoint cung cấp |
| `Connection error` / timeout | Sai base URL hoặc endpoint không truy cập được | Kiểm tra `OPENAI_API_BASE`, thường phải có `/v1` ở cuối |
| `KeyError`/lỗi shape khi query vector | `--embedding-dim` không khớp model embedding | Đặt đúng dim (hf: 384, text-embedding-3-small: 1536) và index lại |
| Đổi embedding nhưng kết quả lỗi | Index cũ tạo bằng embedding khác | `rm -rf ./LiHua-World` rồi index lại |
| `torch` / `transformers` không có | Đang dùng embedding `hf` nhưng chưa cài | `pip install torch transformers`, hoặc chuyển `--embedding-backend openai` |
| Embedding API lỗi `404 /embeddings` | Endpoint không hỗ trợ embeddings | Dùng `--embedding-backend hf` (mặc định) |
| Notebook lỗi `KeyError: 'naive'` | CSV đầu vào thiếu cột baseline | Dùng đoạn chuẩn bị ở mục 7.2 hoặc sửa notebook chỉ chấm `minirag` |
| Notebook lỗi `KeyError: 'Score1'` | Judge trả JSON thiếu key | Kiểm tra prompt/kết quả judge, thêm xử lý fallback |
| Chạy `python ...` báo command not found | Máy chỉ có `python3` | Dùng `python3` thay cho `python` |

---

## 11. Bảng tóm tắt câu lệnh

| # | Câu lệnh | Vai trò |
|---|---|---|
| 1 | `cd /home/nka15/projects/tkg/MiniRAG` | Đứng ở thư mục gốc repo (mọi đường dẫn mặc định là tương đối) |
| 2 | `python3 -m venv .venv && source .venv/bin/activate` | Tạo và kích hoạt môi trường ảo |
| 3 | `pip install -e . && pip install torch transformers` | Cài MiniRAG và dependency cho embedding local |
| 4 | `cp .env.example .env` rồi sửa 3 biến | Cấu hình endpoint OpenAI-compatible |
| 5 | `unzip -o dataset/LiHua-World/data/LiHuaWorld.zip -d dataset/LiHua-World/data/` | Giải nén 442 file chat |
| 6 | `find dataset/LiHua-World/data -name '*.txt' \| wc -l` | Xác nhận dataset đã sẵn sàng (442) |
| 7 | `python3 reproduce/Step_0_index.py --workingdir ./LiHua-World` | Xây graph + vector index từ toàn bộ chat record |
| 8 | `python3 reproduce/Step_0_A_graph_statistic.py --workingdir ./LiHua-World` | In số node/cạnh và thống kê knowledge graph vừa tạo |
| 9 | `python3 reproduce/Step_1_QA.py --workingdir ./LiHua-World --outputpath ./logs/default_output.csv` | Trả lời 637 câu hỏi qua endpoint, ghi CSV |
| 10 | `jupyter notebook dataset/LiHua-World/evaluation.ipynb` | Chấm điểm bằng LLM judge, tính acc/err |
| 11 | `python3 main.py --model <model>` | (Tuỳ chọn) smoke test end-to-end với một câu hỏi toy |
| 12 | `rm -rf ./LiHua-World logs/default_output.csv` | (Tuỳ chọn) xoá index/kết quả để chạy lại từ đầu |

---

## 12. Tóm tắt luồng reproduce

```text
Dataset (dataset/LiHua-World/)
        │
        ├── data/LiHuaWorld.zip ──unzip──> 442 file .txt
        │
        └── qa/query_set.csv ──637 câu hỏi──┐
                                            │
Step_0_index.py                             │
  LLM: OpenAI-compatible endpoint           │
  Embedding: HF local (mặc định)            │
        │                                   │
        ▼                                   │
./LiHua-World/  (graph + KV + vector DB)    │
        │                                   │
        └──────────► Step_1_QA.py ◄─────────┘
                     LLM: cùng endpoint
                     Embedding: cùng cấu hình
                            │
                            ▼
                 logs/default_output.csv
                            │
                            ▼
              evaluation.ipynb (LLM judge)
                            │
                            ▼
                       acc / err
```
