# smartbai bước 1: baseline chỉ dùng ảnh thật trên VisA. Chốt ngày 2026-11-20, TRƯỚC khi chạy

**Mục đích:** đo hiệu năng phát hiện lỗi giảm thế nào khi có ít ảnh thật. Đây là mốc mà dữ liệu tổng hợp (Omniverse Replicator, bước sau) phải bù được. Bước này không có dữ liệu tổng hợp nào.

## Dữ liệu
- **VisA** (Zou et al., ECCV 2022, CC BY 4.0): 12 loại vật thể, 10,821 ảnh. Dùng đúng các file split chính thức trong `spot-diff/split_csv`.
- Ảnh được resize về 256×256 (bilinear, không crop để không cắt mất lỗi ở mép), chuẩn hoá theo ImageNet.

## Thí nghiệm A: không giám sát, chỉ dùng ảnh bình thường (split `1cls`)
- **Phương pháp:** PatchCore (Roth et al., CVPR 2022).
  - Đặc trưng: WideResNet-50 ImageNet, lấy layer2 + layer3, avg-pool 3×3, rồi chiếu xuống 1024 chiều.
  - Coreset tham lam 10% (chiếu ngẫu nhiên 128 chiều).
  - Điểm ảnh = max khoảng cách láng giềng gần nhất của các patch. Bản đồ lỗi được nội suy lên 256 và làm mượt Gauss với σ = 4.
- **Ngân sách ảnh thật:** 100%, 10%, 2% số ảnh bình thường dùng để train (tối thiểu 5 ảnh). Lấy mẫu ngẫu nhiên theo seed.
- **Test:** test `1cls` (100 bình thường + 100 lỗi mỗi vật thể), giữ cố định.

## Thí nghiệm B: có giám sát với ảnh lỗi thật (split `2cls_highshot`)
- **Phương pháp:** ResNet-50 (torchvision, ImageNet) fine-tune phân loại nhị phân ảnh bình thường / lỗi, train riêng cho từng vật thể.
  - AdamW, lr 1e-4, weight decay 1e-4, cosine; 600 bước, batch 32, lấy mẫu cân bằng 50/50, AMP.
  - Augmentation: lật ngang/dọc, color jitter nhẹ.
  - Số bước cố định, không early stopping, không dùng test để chọn gì.
- **Ngân sách ảnh lỗi thật:** k ∈ {60 (toàn bộ), 20, 5} mỗi vật thể; dùng toàn bộ ảnh bình thường của train.
- **Đối chứng k = 0:** PatchCore train trên ảnh bình thường của train `2cls_highshot`, cùng tập test.
- **Test:** test `2cls_highshot` (40 lỗi + khoảng 400 bình thường mỗi vật thể), giữ cố định.

## Seed và chỉ số
- **Seed:** 3 seed (0, 1, 2) cho việc lấy mẫu ngân sách và khởi tạo train.
- **Chỉ số:** image AUROC và image AP (chính); pixel AUROC cho PatchCore (phụ). AUPRO để bước sau.
- **Báo cáo:** trung bình qua 12 vật thể theo từng (phương pháp, ngân sách), kèm độ lệch chuẩn qua seed.

## Kiểm tra hợp lệ (đặt trước)
- PatchCore 100% trên `1cls` phải có image AUROC trung bình khoảng 0.90–0.96. Các bài công bố báo khoảng 0.91–0.95 tuỳ độ phân giải; mình dùng 256 không crop nên có thể thấp hơn một chút.
- Nếu thấp hơn 0.88 thì coi là lỗi cài đặt, phải sửa trước khi dùng làm baseline.

## Báo cáo
Mọi kết quả đều được ghi lại. Bước 1 không có giả thuyết để kiểm định; nó chỉ định nghĩa baseline và "đường cong ngân sách ảnh thật" cho các bước sau. Bước sau phải so thêm với ảnh lỗi tổng hợp 2D rẻ tiền (CutPaste, DRAEM/NSA), không chỉ với ảnh thật, để biết Omniverse có đáng công không.
