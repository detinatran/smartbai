# smartbai: Cognitive Twin cho nhà máy SME ở Dubai

**smartbai** ("smart factory in Dubai") xây một *cognitive digital twin* cho một dây chuyền sản xuất SME, theo hướng của dự án RDI của Dubai Future Foundation [*Cognitive Twins: Leveraging Artificial Intelligence and Omniverse-Powered Digital Twins for Transforming Dubai's Manufacturing Sector*](https://rdi.dubaifuture.ae/?p=356).

"Cognitive" ở đây có nghĩa cụ thể như sau. Twin không chỉ hiển thị dữ liệu. Một LLM agent **đọc trạng thái twin, chạy mô phỏng what-if trong twin để kiểm chứng, rồi mới đề xuất hành động**. Mọi khẳng định của agent phải dựa trên quan sát từ twin.

> Trạng thái: giai đoạn đầu. Isaac Sim đã chạy headless (vật lý + render). Scene dây chuyền, server tool và adapter LLM chưa làm.

## Kiến trúc

```
 Bối cảnh Dubai (khí hậu, giá điện, ca làm SME)       Cảm biến & máy của dây chuyền
                 \                                       /
                  v                                     v
          ┌──────── Omniverse Digital Twin (trạng thái, mô phỏng) ────────┐
          │              ^ truy vấn what-if   | kết quả                     │
          v              |                    v                            │
       Tầng AI cognitive: Dự đoán → Suy luận (LLM agent) → Quyết định ─────┘
                                 |                     |
                       Quản lý SME (hỏi đáp)    hành động trả về nhà máy
                                 └── Đánh giá: kịch bản, chi phí, OEE
```

| Tầng | Nội dung | Công cụ |
|---|---|---|
| 1. Twin vật lý/hình học | scene 3D: băng chuyền, các trạm, camera, cảm biến ảo | NVIDIA Isaac Sim 6.2 / Omniverse (OpenUSD) |
| 2. Dữ liệu | luồng cảm biến và sự kiện gắn vào từng trạm | mô phỏng + dữ liệu công khai (xem dưới) |
| 3. AI dự đoán | dự báo hỏng hóc, lỗi sản phẩm, OEE | PyTorch |
| 4. Cognitive | LLM agent gọi tool của twin, chạy what-if, nộp quyết định có bằng chứng | adapter LLM (xem [`docs/adapter_interface.md`](docs/adapter_interface.md)) |

## Câu hỏi nghiên cứu

Agent được kiểm chứng bằng twin (có `run_whatif`, mọi con số phải trỏ tới một quan sát của twin) có ra quyết định tốt hơn và bịa ít hơn không? So với:
- (a) agent chỉ đọc dữ liệu, không có what-if;
- (b) luật cố định.

Đo trên một bộ kịch bản chốt trước: máy hỏng, đơn gấp, nắng nóng cực đoan, lỗi chất lượng. Chỉ số: độ đúng, chi phí/sản lượng sau quyết định, tỷ lệ khẳng định không có bằng chứng.

## Dữ liệu (một nhà máy thống nhất, không ghép rời rạc)

| Thành phần | Nguồn | Ghi chú |
|---|---|---|
| Cấu trúc dây chuyền, lỗi sản phẩm | [Bosch Production Line Performance](https://www.kaggle.com/competitions/bosch-production-line-performance) (Kaggle) | cột theo line/station → layout scene; phải tự chấp nhận luật cuộc thi trên Kaggle |
| Sức khoẻ máy mỗi trạm | mô phỏng trong twin, hiệu chỉnh theo kiểu [AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (CC BY 4.0) | chưa làm |
| Khí hậu Dubai | thời tiết lịch sử theo giờ | nhiệt độ môi trường → tải làm mát, mòn, điện năng |
| Năng lượng / bối cảnh | [Dubai Pulse](https://www.dubaipulse.gov.ae) (DEWA consumption, DAFZA utilities), MoIAT open data | tham số giá điện, tiêu thụ |
| Camera QC (tuỳ chọn) | [VisA](https://github.com/amazon-science/spot-diff) (CC BY 4.0) | baseline trong `modules/qc_visa` |

Không có dữ liệu cấp máy của nhà máy Dubai nào công khai. Dữ liệu thật của SME đối tác sẽ thay vào khi nhóm dự án cung cấp.

## Cấu trúc repo

| Đường dẫn | Nội dung |
|---|---|
| [`PLAN.md`](PLAN.md) | kế hoạch và các bước |
| [`docs/adapter_interface.md`](docs/adapter_interface.md) | **giao diện giữa twin và adapter LLM** (danh sách tool, quy tắc bằng chứng, ràng buộc chạy) |
| [`twin/isaac_smoke.py`](twin/isaac_smoke.py) | test headless Isaac Sim: vật lý + render Replicator |
| [`modules/qc_visa/`](modules/qc_visa) | baseline phát hiện lỗi sản phẩm trên VisA (PatchCore, ResNet-50), protocol chốt trước; chưa chạy |

## Cài đặt Isaac Sim (Windows, đã kiểm tra)

```bash
python3.12 -m venv is && is\Scripts\activate
pip install --upgrade pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==6.2.0.0" --extra-index-url https://pypi.nvidia.com
set OMNI_KIT_ACCEPT_EULA=YES      # chỉ đặt sau khi đã đọc và đồng ý NVIDIA Omniverse EULA
python twin/isaac_smoke.py out/isaac_smoke
```

- Cần driver NVIDIA ≥ 580.88 và khoảng 20 GB ổ trống. Trên Windows nên bật long path.
- Kết quả đã kiểm tra trên RTX 4090: app khởi động sau khoảng 40 giây; khối hộp rơi và nằm yên ở z = 0.25; Replicator xuất ảnh RGB.
