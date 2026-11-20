# Giao diện twin ↔ adapter LLM (bản nháp, có thể đổi)

Tài liệu này dành cho người viết adapter LLM. Mục tiêu là dùng lại một adapter agent nhiều lượt, kiểu adapter SOP-Bench: parse lệnh bằng AST, chỉ gọi tool trong whitelist, giao thức JSON, trace đầy đủ, chấm điểm có cờ bằng chứng. Thay vì tool mock đọc CSV, adapter gọi tool của twin.

## 1. Yêu cầu với adapter
- **Module Python** (`.py`) import được, không phải notebook:
  `run_case(task, tools, system_prompt, generate, out_dir, max_turns) -> result`.
- **`generate(messages)` tách riêng** để đổi backend: HF local (≤ 8B, có 4-bit) và API chuẩn OpenAI (vLLM / server ngoài).
- **Registry tool:** tên + JSON schema + hàm. Lệnh của model được parse (AST/JSON) rồi chỉ dispatch vào whitelist, không bao giờ `eval`.
- **Tool từ xa:** twin chạy ở process riêng. Tool là HTTP call có timeout. Lỗi tool được trả về cho model như một observation, không làm hỏng cả case.
- **Trace:** lưu từng lượt (prompt, output thô, lệnh đã parse, kết quả tool), ghim model revision / seed / hash prompt, xuất một thư mục hoặc ZIP mỗi lần chạy.
- **Chấm điểm:** F (định dạng), khớp hành động, C (ngữ nghĩa), R (lý do), và **cờ bằng chứng** (mục 3).

## 2. Tool phía twin (đã chạy được: [`sim/tool_server.py`](../sim/tool_server.py), `GET /tools` trả danh sách và tham số)

Mọi tool trả về JSON có `observation_id` (chuỗi duy nhất) và `t_sim` (thời điểm mô phỏng).

| Tool | Tham số | Trả về |
|---|---|---|
| `get_line_state` | không có | trạng thái mọi trạm, nhiệt độ ngoài trời và trong xưởng, KPI từ đầu phiên |
| `get_station` | `station_id` | trạng thái (up/down/maint), tốc độ, hàng chờ, nhiệt độ động cơ, độ rung, số lần hỏng |
| `get_sensor_history` | `station_id, signal (motor_temp/vibration/queue/status), hours ≤ 48` | chuỗi thời gian 10 phút/điểm |
| `predict_failure` | `station_id, horizon_h` | xác suất hỏng trong khoảng và bằng chứng cảm biến (baseline, AUC 0.785) |
| `run_whatif` | `actions` (danh sách hành động như các tool dưới, thêm `set_hall_coupling`), `hours ≤ 48`, `reps ≤ 10` | trung bình ± độ lệch KPI: sản lượng, tỷ lệ lỗi, OEE, số lần hỏng, điện, chi phí; twin thật không đổi |
| `schedule_maintenance` | `station_id, in_minutes` | dừng bảo trì 30 phút sau `in_minutes` |
| `set_speed` | `station_id, pct (30–150)` | xác nhận; chạy nhanh làm máy mòn nhanh hơn |
| `set_dispatch` | `weights` (vd. `{S35: 0.5, S36: 1.5}`) | chia tải giữa các trạm song song |
| `submit_decision` | `report` (JSON: hành động, lý do, các con số kèm `observation_id`) | kết thúc case |

## 3. Quy tắc bằng chứng (chỉ số chính)
- Mọi con số hay khẳng định về trạng thái trong `submit_decision.report` phải kèm `observation_id` của một tool đã gọi trong case.
- Adapter đánh dấu:
  - `unsupported`: không có `observation_id`, hoặc id không tồn tại;
  - `mismatch`: giá trị khác với observation;
  - `unverified_action`: đề xuất hành động mà không chạy `run_whatif` trước. Chỉ số này đo riêng cho điều kiện có what-if.
- So sánh ba điều kiện cùng model, cùng kịch bản, cùng ngân sách lượt:
  - (a) đủ tool, có `run_whatif`;
  - (b) không có `run_whatif`;
  - (c) luật cố định, không dùng LLM.

## 4. Ràng buộc chạy
- Windows 11, Python 3.12, RTX 4090 24 GB **dùng chung** (Isaac Sim + tiến trình khác). LLM local nên ≤ 8B ở 4-bit (khoảng 6–10 GB), hoặc gọi API.
- Mỗi kịch bản tối đa khoảng 12–20 lượt. Chạy hàng loạt, tiếp tục được khi bị ngắt (bỏ qua case đã có trace).

## 5. Việc tiếp theo phía twin
Dựng scene dây chuyền mẫu, server tool (HTTP) đúng bảng trên, và bộ kịch bản đánh giá chốt trước khi chạy.
