# Tích hợp LLM agent: đánh giá kết quả chạy thử

Ngày đánh giá: 24/11/2026. Mã lần chạy: `20261124T133343Z_26b0f2`.

Agent đề xuất phương án (proposer) dùng Qwen3.5 trong chế độ Certify đã hoàn thành **5/5 tình huống phát triển (DEV)**, với **18/18 phản hồi JSON hợp lệ**, không có lỗi gọi tool và không có lượt chạm giới hạn token đầu ra. Agent đã gửi sáu phương án hành động. Toàn bộ 16 nhận định về số liệu/trạng thái có dẫn chứng đã được đối chiếu với đúng trạm và trường dữ liệu trong observation được trích dẫn.

Kết quả xác nhận agent đã kết nối và hoạt động được với twin trên năm tình huống **chỉ thuộc nhóm suy giảm thiết bị (degradation)**. Chưa có đánh giá về lợi ích, độ an toàn của hành động, hiệu quả của bộ quyết định đã hiệu chuẩn (gate), hoặc độ ổn định trên các nhóm tình huống khác. Năm tình huống này được sử dụng trong quá trình phát triển sau khi sửa prompt và cấu hình chạy; chúng không phải tập kiểm thử độc lập và không được dùng để so sánh trực tiếp với kết quả đã công bố của nhóm hướng dẫn.

## Cấu hình đã chạy

- Model: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Phần cứng: hai GPU Tesla T4 trên Kaggle; lượng tử hóa NF4, tính toán FP16, tắt chế độ thinking, giải mã greedy, seed 42, batch size bằng 1.
- Giới hạn: 12 lượt; tối đa 1.024 token đầu ra mỗi lượt và 16.384 token đầu vào.
- Cấu hình bộ nhớ: xử lý đầu vào theo khối 256 token (chunked prefill), chỉ tính logits của token cuối và dừng sinh tại token kết thúc lượt hội thoại.
- Giao thức: đọc dữ liệu bằng tool và gọi `submit_decision` với các phương án (`candidates`), lý do (`rationale`) và nhận định có dẫn chứng (`claims`). Proposer không được thực thi hành động hoặc gọi `run_whatif`. Gửi thành công một phương án chưa đồng nghĩa phương án đã được thực thi hay chứng nhận an toàn.
- Model được tải một lần trong mỗi tiến trình. Phản hồi sai bị từ chối và trả lại lỗi để model xử lý; hệ thống không tự sửa output thành báo cáo thành công.

## Kết quả từng tình huống

| Tình huống DEV | Số lượt | Số phương án | Số nhận định có dẫn chứng | Thời gian suy luận (giây) |
|---|---:|---:|---:|---:|
| `dev_degradation_0` | 2 | 2 | 2 | 37,12 |
| `dev_degradation_1` | 8 | 1 | 7 | 158,62 |
| `dev_degradation_2` | 3 | 1 | 2 | 46,13 |
| `dev_degradation_3` | 2 | 1 | 3 | 34,12 |
| `dev_degradation_4` | 3 | 1 | 2 | 44,79 |

Tổng thời gian suy luận của model: **320,78 giây**. Số liệu này không bao gồm tải checkpoint, tách token, thực thi tool và các chi phí xử lý khác trên máy chủ.

`finished` có nghĩa server đã xác nhận nhận được báo cáo đúng cấu trúc. Phép kiểm tra tự động xem giá trị có xuất hiện trong observation hay không yếu hơn việc đánh giá ý nghĩa của nhận định; bước đối chiếu đúng trạm/trường dữ liệu nêu trên được thực hiện riêng. Cả hai bước đều chưa xác nhận giải thích nhân quả hoặc thời điểm bảo trì mà agent đề xuất là hợp lý.

Dữ liệu tổng hợp để kiểm tra lại: [chỉ số từng tình huống](../experiments/agent_smoke/results/dev_degradation_5.csv) và [cấu hình cùng mã băm nguồn](../experiments/agent_smoke/results/dev_degradation_5_provenance.json).

ZIP gốc được lưu riêng, có SHA-256:
`00608341d87614ae9daba4eee3454614ce4c6b3813404414abeea71650097b24`.
Các mã băm nguồn xác định những file đã được dùng trong lần chạy này. Việc chuẩn hóa định dạng code và đóng gói sau đó không được tính là một lần chạy GPU mới.

## Nguồn dữ liệu và mức độ phù hợp

| Nguồn | Vai trò và trạng thái tích hợp | Giới hạn |
|---|---|---|
| [Bosch Production Line Performance](https://www.kaggle.com/competitions/bosch-production-line-performance/data) | Converter hiện có dùng dữ liệu để dựng lại dây chuyền 39 trạm và thống kê sản xuất; toàn bộ đầu vào tập train đã được xử lý trên Kaggle. | Nhãn sản phẩm `Response` không phải nhãn hỏng máy, độ an toàn hành động hoặc lợi ích khi thay đổi hành động. |
| [Open-Meteo Historical Weather / ERA5](https://open-meteo.com/en/docs/historical-weather-api) | Cung cấp đầu vào nhiệt độ ngoài trời tại Dubai theo giờ, giai đoạn 2015–2025, gồm 96.432 giờ liên tục. | Dữ liệu tái phân tích khí tượng không phải nhiệt độ xưởng/motor đo trực tiếp và không chứng minh tính đúng đắn khi vận hành nhà máy thực. |
| [UCI Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) | Nguồn bổ sung được đề xuất, chưa tích hợp hoặc thực nghiệm. Có 2.205 chu kỳ đo trên hệ thống thử nghiệm, gồm dữ liệu cảm biến và nhãn tình trạng linh kiện. | Không thay trực tiếp cho bài toán điều khiển dây chuyền: thiếu đáp án chuẩn về kết quả hành động của twin, cần adapter, cách chia dữ liệu và thiết kế đánh giá riêng. |

Đề xuất hiện tại là **Bosch + ERA5 cho bài toán điều khiển trong simulator đang có**. UCI Hydraulic chỉ là lựa chọn bổ sung cho bài toán riêng về chẩn đoán tình trạng và kiểm tra bằng chứng, cần thống nhất với người hướng dẫn. Không nên thay dữ liệu chỉ để tăng số lượng bộ dữ liệu. UCI công bố bộ Hydraulic theo giấy phép CC BY 4.0, DOI [10.24432/C5CW21](https://doi.org/10.24432/C5CW21).

Cấu trúc dây chuyền và thống kê được suy ra từ dữ liệu; động lực học và tác động của hành động phụ thuộc các giả định trong simulator. Dữ liệu thời tiết được dựng lại độc lập, chưa khẳng định giống hoàn toàn đầu vào gốc của nhóm hướng dẫn. Dữ liệu Bosch và các đầu vào riêng suy ra từ Bosch không được đưa vào phần mã nguồn chia sẻ, tuân theo điều khoản cuộc thi. Các liên kết nguồn được kiểm tra ngày 24/11/2026.

## Công việc nghiên cứu còn lại

1. Thống nhất phương pháp chính, phiên bản benchmark và lựa chọn dữ liệu với người hướng dẫn.
2. Đánh giá lợi ích/độ an toàn của phương án và chất lượng bằng chứng trên tập DEV đa dạng hơn.
3. Cố định prompt, cấu hình giải mã, ngân sách tính toán và giao thức trước khi đánh giá trên tập calibration/test.

Sản phẩm hiện tại là proposer có thể tái sử dụng và kết quả chạy thử tích hợp đã được ghi nhận. Đây chưa phải thuật toán CD-SV hoàn chỉnh hoặc kết quả thực nghiệm đủ để kết luận ở cấp độ paper.
