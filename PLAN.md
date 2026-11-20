# smartbai: Cognitive Twin cho một nhà máy SME ở Dubai (dự án DFF "Cognitive Twins")

Ngày lập: 2026-11-20.

## Kiến trúc (vòng lặp kín)
```
 Dubai context (khí hậu, giá điện, ca làm SME)        Cảm biến & máy của dây chuyền
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

## Xương sống dữ liệu (một nhà máy thống nhất, không ghép rời rạc)
| Thành phần | Nguồn | Vai trò | Trạng thái |
|---|---|---|---|
| Cấu trúc dây chuyền: các trạm, luồng sản phẩm, lỗi chất lượng | Bosch Production Line Performance (Kaggle) | topology line/station → layout scene Omniverse; lỗi sản phẩm | phải tự đăng nhập Kaggle và chấp nhận luật cuộc thi |
| Sức khoẻ máy ở mỗi trạm | mô phỏng trong twin, hiệu chỉnh kiểu AI4I 2020 (UCI, CC BY 4.0) | dự báo hỏng hóc | chưa làm |
| Khí hậu Dubai | dữ liệu thời tiết lịch sử theo giờ (ví dụ Open-Meteo / Meteostat) | nhiệt độ môi trường → tải làm mát, mòn, điện năng | chưa làm |
| Bối cảnh năng lượng/kinh tế | Dubai Pulse (DEWA consumption, DAFZA utilities), MoIAT open data | tham số giá điện, tiêu thụ ngành | chưa làm |
| Camera QC trạm cuối (tuỳ chọn) | VisA (Amazon, CC BY 4.0) | phát hiện lỗi sản phẩm | baseline đã có code (`modules/qc_visa`), chưa chạy |

Đã loại: NASA C-MAPSS (động cơ máy bay, không phải máy nhà máy), Tennessee Eastman (quy trình hoá chất liên tục, không khớp dây chuyền rời rạc).

## Các bước
1. Cài Isaac Sim 6.2 (pip) trên máy GPU Windows; kiểm tra chạy headless. **Đã xong** (`twin/isaac_smoke.py`).
2. Dựng scene dây chuyền SME mẫu (đóng gói thực phẩm/đồ uống): băng chuyền, các trạm theo topology Bosch, camera, cảm biến ảo. Script hoá bằng Python để tái lập được.
3. Bộ mô phỏng sự kiện rời rạc của dây chuyền (sản lượng, hàng chờ, hỏng máy, nhiệt độ Dubai), đồng bộ với scene.
4. Tầng AI: dự báo hỏng hóc mỗi trạm, dự báo lỗi sản phẩm (Bosch), OEE.
5. Tầng cognitive: LLM agent gọi công cụ (giao diện: `docs/adapter_interface.md`) (đọc trạng thái twin, chạy what-if, đề xuất). So sánh: agent có what-if, agent chỉ đọc dữ liệu, và luật cố định.
6. Bộ kịch bản đánh giá (chốt trước khi chạy): máy hỏng, đơn gấp, nắng nóng cực đoan, lỗi chất lượng. Chỉ số: độ đúng câu trả lời, chi phí/sản lượng sau quyết định, tỷ lệ bịa.
