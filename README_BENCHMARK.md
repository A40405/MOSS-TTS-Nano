# Hướng Dẫn Benchmark MOSS-TTS-Nano

Tài liệu này viết theo hướng triển khai thực tế, bám sát `README.md` gốc của repo và bổ sung phần benchmark.

Benchmark này mặc định chỉ chấm tiếng Anh.

Ngoài chế độ chạy theo một demo, benchmark còn có sẵn preset tiếng Anh:

- `english_news`: một prompt tiếng Anh kiểu bản tin
- `english_mix`: bộ prompt tiếng Anh ngắn, trung bình, dài để gần thực tế sản phẩm hơn

## 1. Mục Tiêu Benchmark

Script benchmark đo các chỉ số sau:

- `RTF = total generation time / generated audio duration`
- latency `p50 / p95 / p99`
- streaming first chunk audio time `p50 / p95 / p99`
- `CCU` tối đa sao cho:
  - `p95 first chunk < 200 ms`
  - error rate `< 1%`

## 2. Đọc Kỹ Setup Từ README Gốc

Phần setup chính thức trong `README.md` của repo là:

1. Tạo môi trường Python sạch
2. `pip install -r requirements.txt` hoặc `pip install -r requirements-gpu.txt`
3. `pip install -e .`
4. Nếu `WeTextProcessing` hoặc `pynini` lỗi thì xử lý theo nhánh riêng

Điểm quan trọng:

- `README.md` gốc khuyến nghị Python `3.12`
- repo có pin `torch==2.7.0`, `torchaudio==2.7.0`, `transformers==4.57.1`
- `WeTextProcessing` là dependency thật, không phải thứ có thể “bỏ qua” nếu bạn muốn đúng setup gốc
- `onnxruntime` là dependency thật của repo
- benchmark này chỉ dùng bộ demo tiếng Anh trong `assets/demo.jsonl`
- file `requirements-gpu.txt` trong repo hiện đã được chỉnh sang nhánh CUDA `cu126` để phù hợp hơn với máy đang có CUDA 12.x

## 3. Cấu Hình Máy Đã Phát Hiện

Mình đã dò được:

- OS: Windows 10 Home Single Language 25H2, build 26200
- CPU: Intel Core i5-10300H
- CPU cores / threads: 4 / 8
- RAM: 16 GB
- GPU: NVIDIA GeForce GTX 1650
- VRAM: 4 GB
- NVIDIA driver: 555.97
- CUDA theo `nvidia-smi`: 12.5
- Python ở base shell: 3.13.11


## 4. Hai Luồng Cài Đặt Khuyến Nghị

### 4.1 Tạo env sạch

```powershell
conda create -n moss-tts-nano python=3.12 -y
conda activate moss-tts-nano
```

### 4.2 Luồng Cài Thường

Luồng này phù hợp nếu bạn muốn:

- bám đúng setup gốc của repo
- ưu tiên chạy CPU / ONNX CPU trước
- xử lý ổn định phần `WeTextProcessing` trên Windows trước khi bật GPU

Lệnh cài:

```powershell
pip install -r requirements.txt
python -m pip install -e .
```

Nếu `WeTextProcessing` hoặc `pynini` lỗi, làm tiếp đúng theo fallback của README:

```powershell
conda install -c conda-forge pynini=2.1.6.post1 -y
python -m pip install git+https://github.com/WhizZest/WeTextProcessing.git
python -m pip install -r requirements.txt
python -m pip install -e .
```

Nếu không dùng conda, README gốc nói bạn phải tự có wheel `pynini` đúng Python/platform rồi mới cài `WeTextProcessing`.

Sau khi cài xong luồng thường, bạn có thể smoke test CPU / ONNX CPU trước:

```powershell
python infer_onnx.py --execution-provider cpu --disable-wetext-processing --realtime-streaming-decode 1 --prompt-audio-path assets/audio/zh_1.wav --text "Welcome to the ONNX Runtime CPU demo." --output-audio-path generated_audio\smoke_onnx_cpu.wav
```

### 4.3 Luồng Cài GPU

Luồng này phù hợp nếu bạn muốn:

- ưu tiên benchmark ONNX CUDA trên máy có NVIDIA GPU
- đồng bộ `torch`, `torchaudio`, `onnxruntime-gpu` theo cùng một nhánh CUDA
- chạy server benchmark gần với thực tế sản phẩm hơn

Repo hiện đã có sẵn [requirements-gpu.txt](./requirements-gpu.txt) cho nhánh này, và file đó đang pin:

- `torch==2.7.0+cu126`
- `torchaudio==2.7.0+cu126`
- `onnxruntime-gpu>=1.20.0`

Lệnh cài luồng GPU trên env sạch:

```powershell
conda install -c conda-forge pynini=2.1.6.post1 -y
python -m pip install git+https://github.com/WhizZest/WeTextProcessing.git
python -m pip install -r requirements-gpu.txt
python -m pip install -e .
```

Lý do có thể cài gọn như trên:

- `requirements-gpu.txt` đã chứa toàn bộ base dependencies cần thiết
- file này đã pin luôn `torch`, `torchaudio`, `onnxruntime-gpu` cho nhánh GPU
- `WeTextProcessing` đã nằm sẵn trong `requirements-gpu.txt`


**Ghi Chú Cho Luồng GPU**

Check version với luồng GPU:

```powershell
python -c "import torch, onnxruntime as ort; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(ort.get_available_providers())"
```

Kỳ vọng sau khi cài xong:

- `torch.cuda.is_available()` trả về `True`
- `torch.__version__` có hậu tố `+cu126`
- `onnxruntime` thấy `CUDAExecutionProvider`
- repo hiện tự cấu hình cache Hugging Face ngắn hơn tại `.\.hf-cache` trên Windows để giảm rủi ro lỗi đường dẫn quá dài

## 5. Cách Thực Tế Trên Máy Tôi

Mình không cài tắt theo kiểu bỏ qua lỗi. Quy trình thực tế mình vừa chạy là:

```powershell
conda create -n moss-tts-nano python=3.12 -y
conda activate moss-tts-nano
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Do `pip install -r requirements.txt` bị fail ở bước build `pynini` trên Windows, mình chuyển sang đúng nhánh fallback của README:

```powershell
conda install -n moss-tts-nano -c conda-forge pynini=2.1.6.post1 -y
python -m pip install git+https://github.com/WhizZest/WeTextProcessing.git
python -m pip install -r requirements.txt
python -m pip install -e .
```

Sau khi base env đã sạch, mình mới chuyển sang GPU/ONNX:

```powershell
python -m pip uninstall -y onnxruntime
python -m pip install -r requirements-gpu.txt
python -m pip install -e .
```

Kết quả cuối cùng của env:

- `Python 3.12.13`
- `torch 2.7.0+cu126`
- `torchaudio 2.7.0+cu126`
- `onnxruntime-gpu 1.26.0`
- `torch.cuda.is_available() = True`
- `onnxruntime` thấy `CUDAExecutionProvider`

Lưu ý rất quan trọng:

- Không cài trực tiếp từ `requirements-gpu.txt` ngay từ đầu, vì README gốc của repo đi theo `requirements.txt` trước rồi mới chuyển ONNX sang GPU
- lý do phải cài theo 2 pha là để xử lý đúng lỗi `pynini/WeTextProcessing` trên Windows trước, sau đó mới khóa stack GPU cho ONNX

<!-- ## 5. Setup Khuyến Nghị Trên Máy Hiện Tại

Máy hiện tại đã có env `llama_gpu` khá sát GPU/CUDA, nên nếu bạn muốn chạy ngay không rebuild từ đầu thì:

```powershell
conda activate llama_gpu
python -m pip install -U pip
python -m pip install -r requirements-gpu.txt
python -m pip install -e .
python -m pip install onnxruntime-gpu
```

Nếu `WeTextProcessing` chưa có, cài thêm:

```powershell
python -m pip install git+https://github.com/WhizZest/WeTextProcessing.git
```

Lưu ý:

- Cài `requirements-gpu.txt` trước
- Cài `-e .` sau
- Đừng bỏ qua `WeTextProcessing` nếu bạn muốn bám đúng README gốc
- Nếu `pynini` báo lỗi, cài `pynini` trước rồi mới cài `WeTextProcessing` -->

## 6. Các File Chính

- `infer.py`: PyTorch inference
- `infer_onnx.py`: ONNX inference
- `app.py`: server PyTorch
- `app_onnx.py`: server ONNX
- `benchmark_moss_tts_nano.py`: benchmark script
- `README.md`: tài liệu gốc phải bám theo

## 7. Smoke Test

### 7.1 PyTorch GPU

```powershell
python infer.py --enable-wetext-processing 0 --device cuda --dtype float16 --max-new-frames 32 --prompt-audio-path assets/audio/zh_1.wav --text "Welcome to the local smoke test." --output-audio-path generated_audio\smoke_pytorch.wav
```

Ghi chú:
- `--enable-wetext-processing 0` chỉ là để smoke test không bị chặn nếu môi trường thiếu `WeTextProcessing`

### 7.2 ONNX CUDA

```powershell
python infer_onnx.py --execution-provider cuda --disable-wetext-processing --realtime-streaming-decode 1 --prompt-audio-path assets/audio/zh_1.wav --text "Welcome to the ONNX Runtime CUDA demo." --output-audio-path generated_audio\smoke_onnx.wav
```

### 7.3 ONNX CPU

```powershell
python infer_onnx.py --execution-provider cpu --disable-wetext-processing --realtime-streaming-decode 1 --prompt-audio-path assets/audio/zh_1.wav --text "Welcome to the ONNX Runtime CPU demo." --output-audio-path generated_audio\smoke_onnx_cpu.wav
```

## 8. Start Server

### 8.1 ONNX CUDA server

```powershell
python app_onnx.py --execution-provider cuda
```

### 8.2 ONNX CPU server

```powershell
python app_onnx.py --execution-provider cpu
```

Server mặc định:

```text
http://127.0.0.1:18083
```

## 9. Chạy Benchmark

Sau khi server đã sẵn sàng:

```powershell
conda activate moss-tts-nano
python benchmark_moss_tts_nano.py --language en --mode both --server-base-url http://127.0.0.1:18083
```

Nếu muốn dùng preset sản phẩm:

```powershell
python benchmark_moss_tts_nano.py --language en --preset english_news --mode both --server-base-url http://127.0.0.1:18083
```

```powershell
python benchmark_moss_tts_nano.py --language en --preset english_mix --mode both --server-base-url http://127.0.0.1:18083
```

Chỉ streaming:

```powershell
python benchmark_moss_tts_nano.py --language en --mode streaming --server-base-url http://127.0.0.1:18083
```

Chỉ non-streaming:

```powershell
python benchmark_moss_tts_nano.py --language en --mode nonstreaming --server-base-url http://127.0.0.1:18083
```

Sweep concurrency:

```text
1, 2, 4, 8, 16, 32
```

Mặc định, mỗi mức concurrency chạy `requests_per_level * concurrency` request. Với `requests_per_level=4`, số request là:

- 1 -> 4
- 2 -> 8
- 4 -> 16
- 8 -> 32
- 16 -> 64
- 32 -> 128

Nếu muốn đổi số request:

```powershell
python benchmark_moss_tts_nano.py --language en --requests-per-level 2 --mode both
```

Lưu ý:

- khi dùng `--preset english_news` hoặc `--preset english_mix`, benchmark sẽ chạy lần lượt trên toàn bộ prompt tiếng Anh đã định nghĩa sẵn
- `english_news` nhẹ hơn, phù hợp khi bạn muốn đo nhanh
- `english_mix` nặng hơn, nhưng gần thực tế sản phẩm hơn vì có cả prompt ngắn, trung bình và dài

## 10. Output Sinh Ra

Benchmark sẽ ghi vào `benchmark_results/<timestamp>/`:

- raw JSONL
- summary CSV
- summary Markdown
- machine profile JSON

## 11. Cách Đọc Kết Quả

- `RTF < 1.0`: nhanh hơn thời gian thực
- latency càng thấp càng tốt
- first chunk càng thấp càng tốt cho streaming
- `CCU` càng cao càng tốt, miễn:
  - `p95 first chunk < 200 ms`
  - error rate `< 1%`

## 12. Các Lỗi Thực Tế Đã Gặp


### 12.1 Thiếu `pynini` / `WeTextProcessing`

Giải pháp:

- cài `pynini` trước
- cài `WeTextProcessing` sau
- rồi chạy lại `pip install -r requirements-gpu.txt`

### 12.2 Thiếu `onnxruntime`

Giải pháp:

- cài `onnxruntime-gpu` nếu chạy CUDA
- cài `onnxruntime` nếu chạy CPU

### 12.3 Lỗi `PyTorch 2.7.0+cu118 does not support CUDA 12.x`

Nếu bạn chạy `infer_onnx.py --execution-provider cuda` và thấy các lỗi như:

- `The installed PyTorch 2.7.0+cu118 does not support CUDA 12.x`
- `Failed to load cublasLt64_12.dll`
- `Failed to load cudart64_12.dll`

thì đây là lỗi lệch CUDA runtime giữa môi trường PyTorch và `onnxruntime-gpu`.

Cách xử lý chuẩn có 2 hướng:

1. Đồng bộ toàn bộ env sang CUDA 12.x
   - cài PyTorch CUDA 12.x theo trang chính thức của PyTorch
   - giữ `onnxruntime-gpu` bản CUDA 12.x
   - cách này phù hợp nếu bạn muốn CUDA ONNX chạy ổn định lâu dài
   - trên Windows, cần có đủ CUDA Toolkit và cuDNN DLL tương ứng trong `PATH`; chỉ `nvidia-smi` báo CUDA 12.x là chưa đủ

2. Giữ PyTorch CUDA 11.8 và đổi ONNX Runtime sang nhánh CUDA 11.8 phù hợp
   - chỉ nên dùng nếu bạn muốn giữ nguyên env hiện tại
   - cần đảm bảo đúng build `onnxruntime-gpu` cho CUDA 11.x và các DLL CUDA/cuDNN tương ứng trong `PATH`

Với máy này, nếu mục tiêu là benchmark sản phẩm, mình khuyên dùng một env sạch đã đồng bộ CUDA ngay từ đầu thay vì trộn `cu118` với `cu12`.

### 12.4 Thiếu file model ONNX sau khi download dở dang

Nếu bạn thấy lỗi như:

- `FileNotFoundError: ... codec_browser_onnx_meta.json`

thì model ONNX đã download không hoàn tất.

Cách xử lý:

- xóa thư mục `models/MOSS-Audio-Tokenizer-Nano-ONNX` bị download dở:

```powershell
Remove-Item -Recurse -Force models\MOSS-Audio-Tokenizer-Nano-ONNX
```

- chạy lại `infer_onnx.py`
- nếu mạng không ổn, hãy pre-download lại toàn bộ ONNX assets trước khi benchmark

### 12.5 Smoke test có thể fail ở bước download model

Đây là lỗi network/cache, không phải lỗi benchmark code. Nếu gặp `IncompleteRead` hoặc tải bị ngắt:

- chạy lại khi mạng ổn hơn
- hoặc pre-download model trước
- hoặc dùng local cache sẵn có

### 12.6 `app.py` là CPU-only

Nếu muốn benchmark GPU streaming, dùng:

- `app_onnx.py --execution-provider cuda`

### 12.7 Lỗi `WinError 206` do đường dẫn cache Hugging Face quá dài

Nếu bạn chạy `infer.py` và thấy lỗi như:

- `FileNotFoundError: [WinError 206] The filename or extension is too long`

thì nguyên nhân thường là cache của Hugging Face trên Windows bị lồng thư mục quá sâu, nhất là khi model dùng `trust_remote_code=True`.

Trạng thái hiện tại của repo:

- `infer.py` và runtime PyTorch đã tự cấu hình cache Hugging Face ngắn hơn vào thư mục `.\.hf-cache`
- vì vậy sau khi cập nhật code mới, bạn chỉ cần chạy lại lệnh inference là đủ

Nếu máy vẫn còn cache cũ và bạn muốn dọn sạch:

```powershell
Remove-Item -Recurse -Force .hf-cache
```

Sau đó chạy lại:

```powershell
python infer.py --enable-wetext-processing 0 --device cuda --dtype float16 --max-new-frames 32 --prompt-audio-path assets/audio/zh_1.wav --text "Welcome to the local smoke test." --output-audio-path generated_audio\smoke_pytorch.wav
```

## 13. Lệnh Khuyến Nghị Để Chạy Nhanh Trên Máy Này

```powershell
conda activate moss-tts-nano
python app_onnx.py --execution-provider cuda
```

Terminal khác:

```powershell
conda activate moss-tts-nano
python benchmark_moss_tts_nano.py --language en --preset english_mix --mode both --server-base-url http://127.0.0.1:18083
```

## 14. Ghi Chú Quan Trọng

- Nếu muốn đúng chuẩn README gốc, hãy tạo env mới Python 3.12 và cài đầy đủ theo thứ tự ở mục 4
- Nếu model chưa cache thì lần đầu sẽ tải
- Nếu tải model lỗi do mạng, benchmark script vẫn đúng, chỉ là phụ thuộc nguồn ngoài
- Nếu máy không có GPU/CUDA, đổi sang `app_onnx.py --execution-provider cpu`
