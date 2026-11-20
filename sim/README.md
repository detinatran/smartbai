# Bộ mô phỏng sự kiện rời rạc (twin v2)

[`line_sim.py`](line_sim.py) mô phỏng dây chuyền twin v2. Mô hình hoá theo phút, chỉ cần Python + numpy + pandas, một tuần mô phỏng mất khoảng 9 giây CPU.

## Cái gì lấy từ dữ liệu, cái gì là giả định

| Thành phần | Nguồn |
|---|---|
| Tuyến đi của linh kiện: xác suất vào chuyền, rẽ nhánh, ra chuyền | số luồng nối giữa các trạm của bộ Bosch (1.18 triệu sản phẩm) |
| Tải của từng trạm | tỷ lệ sản phẩm đi qua trạm (Bosch) |
| Lỗi sản phẩm | tỷ lệ lỗi của từng trạm (Bosch); phần vượt trung bình gán cho trạm đó; hiệu chỉnh để tỷ lệ lỗi chung bằng số liệu |
| Nhiệt độ ngoài trời | khí hậu Dubai theo giờ 2015–2025 (Open-Meteo) |
| Lưu lượng vào, mức tải danh định, MTBF/MTTR, độ nhạy nhiệt của tốc độ hỏng, mô hình cảm biến, công suất điện, chi phí | **giả định**, liệt kê trong `ASSUMED` và ghi kèm mọi lần chạy; sẽ thay bằng số của SME đối tác |

Mỗi lần hỏng máy đều có một **giai đoạn suy giảm** trước đó (mặc định 4 giờ danh định). Trong giai đoạn này nhiệt độ động cơ và độ rung trôi lên, và trạm tạo ra nhiều lỗi hơn. Nhờ vậy dự báo hỏng có tín hiệu để học. Tốc độ hỏng tăng theo nhiệt độ trong xưởng, mà nhiệt độ xưởng phụ thuộc nhiệt độ ngoài trời Dubai và mức làm mát.

## API (dùng cho server tool)

```python
from line_sim import LineSim, load_climate
sim = LineSim(cfg, load_climate(csv), start="2025-07-15 06:00", seed=0)
sim.run(minutes)                       # chạy twin tiến lên
sim.kpis()                             # sản lượng/giờ, tỷ lệ lỗi, lead time, WIP, availability, OEE, số lần hỏng, kWh, chi phí
sim.station_view("S32")                # trạng thái, tốc độ, hàng chờ, nhiệt động cơ, độ rung
sim.sensor_history("S32", "vibration", hours=6)
sim.apply([{"action": "set_speed", "station_id": "S35", "pct": 80}])          # hành động thật lên twin
sim.whatif([{"action": "schedule_maintenance", "station_id": "S2"}], hours=24, reps=8)  # chạy thử, twin không đổi
```

Hành động: `set_speed`, `schedule_maintenance`, `set_dispatch` (chia tải giữa các trạm song song), `set_hall_coupling` (làm mát thêm/bớt). `whatif` chạy nhiều lần lặp, dùng chung chuỗi ngẫu nhiên cho mọi phương án hỏi tại cùng thời điểm, và trả về trung bình ± độ lệch.

## Kiểm chứng ([`validate_sim.py`](validate_sim.py))

7 ngày × 3 seed, tháng 7 và tháng 1 năm 2025:

| | Mô phỏng | Bosch |
|---|---|---|
| Tương quan tỷ lệ qua từng trạm | 0.9999–1.0 (lệch lớn nhất 1.0–1.6 điểm %) | |
| Tương quan tỷ lệ lỗi theo trạm | 0.98–0.99 | |
| Tỷ lệ lỗi chung | 0.54–0.65% | 0.58% |
| Tỷ lệ lỗi tại S32 | 4.4–6.3% | 4.51% |
| Trung vị số trạm mỗi sản phẩm | 13 | 13 |

Hiệu ứng khí hậu (giả định độ nhạy nhiệt `temp_k = 0.05`): số lần hỏng tháng 7 cao hơn tháng 1 khoảng 20% (112–127 so với 91–105 lần/tuần). Điện HVAC khoảng 6,800 kWh/tuần vào tháng 7, gần 0 vào tháng 1.

Ví dụ what-if (twin lúc 2025-07-04 08:00, 5 trạm đang suy giảm, 5 lần lặp):

| Phương án | Sản lượng/giờ | Số lần hỏng | Điện HVAC (kWh) | Chi phí |
|---|---|---|---|---|
| Không làm gì (24 giờ) | 360 ± 3 | 20.4 ± 2.6 | 1,118 | 6,666 ± 570 |
| Bảo trì lần lượt 5 trạm suy giảm (24 giờ) | 354 ± 2 | 17.6 ± 2.7 | 1,118 | 6,232 ± 780 |
| Không làm gì (12 giờ) | 357 ± 8 | 12.2 ± 0.8 | 766 | 3,755 ± 320 |
| Làm mát thêm (12 giờ) | 349 ± 6 | 10.0 ± 0.7 | 930 | 3,667 ± 270 |
| S35 chậm còn 50% (12 giờ) | 303 ± 4 | 12.2 ± 1.3 | 766 | 3,600 ± 670 |
| S35 50% + dồn tải sang S36 (12 giờ) | 318 ± 4 | 11.0 ± 1.2 | 766 | 3,419 ± 370 |

Bảo trì giảm hỏng và chi phí nhưng mất một ít sản lượng vì phải dừng máy. Làm mát thêm giảm hỏng nhưng tăng điện HVAC khoảng 21%. Đây là những đánh đổi agent phải cân nhắc.

## Server tool ([`tool_server.py`](tool_server.py))

Đưa twin ra cho agent LLM qua HTTP theo [`../docs/adapter_interface.md`](../docs/adapter_interface.md). Chỉ dùng thư viện chuẩn:

```bash
python sim/tool_server.py TWIN_V2_CONFIG.json dubai_hourly_2015_2025.csv --port 8765 --start "2025-07-03 06:00"
curl localhost:8765/tools
curl -X POST localhost:8765/tool/get_station -d '{"station_id": "S32"}'
curl -X POST localhost:8765/tool/run_whatif -d '{"actions": [{"action": "schedule_maintenance", "station_id": "S2"}], "hours": 12}'
curl localhost:8765/admin/observations      # sổ quan sát để chấm bằng chứng
```

Mỗi kết quả tool có `observation_id`, `t_sim` và thời gian twin. Tool lỗi (trạm không tồn tại, thiếu tham số) trả về lỗi để model đọc như một quan sát. `/admin/reset` và `/admin/advance` dành cho phía dựng kịch bản đánh giá, agent không dùng.

`predict_failure` hiện là **baseline**: hồi quy logistic trên độ rung và nhiệt độ động cơ đã trừ phần do tải, chỉ dùng cảm biến quan sát được, không đọc trạng thái ẩn của mô phỏng. Trên một tuần tháng 7 (6,415 mẫu theo giờ, 458 lần hỏng trong 4 giờ tới), AUC = 0.785. Sẽ thay bằng mô hình học ở tầng 3.
