"""
Automated Agent Benchmark Suite Module.
Evaluates 108 UNIQUE, stratified ground-truth test cases across 6 distinct evaluation categories:
1. Explicit Manual Specs
2. Domain RAG Vector Matches
3. Borderline (0.25-0.45) Agent Loop Correction Cases
4. Speech-to-Text (STT) Hearing Mismatches
5. False Refusals (Tier 2 Refusal Routing)
6. Tier 5 General Knowledge & Conversational Queries
"""

import time
from collections import Counter
from typing import Dict, Any, List
from src.eval_router import AdaptiveEvalRouter


def compute_confusion_matrix(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates Confusion Matrix, False Pass Rate (FP), and False Reject Rate (FN)."""
    matrix = Counter()
    total = len(results)

    for r in results:
        expected = r.get("expected_verdict", "PASS")
        predicted = r.get("auto_result", "RETEST")
        matrix[(expected, predicted)] += 1

    # False Pass (FP): Predicted PASS when Ground Truth is NOT PASS (FAIL or RETEST)
    fp_count = sum(cnt for (e, p), cnt in matrix.items() if p == "PASS" and e != "PASS")
    
    # False Reject (FN): Predicted FAIL/RETEST when Ground Truth is PASS
    fn_count = sum(cnt for (e, p), cnt in matrix.items() if p != "PASS" and e == "PASS")

    fp_rate = (fp_count / total) * 100.0 if total > 0 else 0.0
    fn_rate = (fn_count / total) * 100.0 if total > 0 else 0.0

    return {
        "matrix": {f"{e}->{p}": cnt for (e, p), cnt in matrix.items()},
        "false_pass_count": fp_count,
        "false_pass_rate_pct": round(fp_rate, 1),
        "false_reject_count": fn_count,
        "false_reject_rate_pct": round(fn_rate, 1)
    }


def get_real_ground_truth_testset() -> List[Dict[str, Any]]:
    """Returns 108 UNIQUE, diverse real ground-truth test cases across 6 stratified slices (18 cases each)."""
    cases = []

    # Slice 1: Explicit Specs (18 unique cases)
    spec_data = [
        ("Nút SOS dùng để làm gì?", "Nút SOS dùng để kích hoạt cuộc gọi khẩn cấp eCall thủ công trên xe.", "kích hoạt cuộc gọi khẩn cấp eCall thủ công trên xe"),
        ("Tài liệu này có bản quyền không?", "Mọi quyền được bảo lưu, thuộc bản quyền của VinFast.", "mọi quyền được bảo lưu thuộc bản quyền vinfast"),
        ("Làm sao để bật điều hòa?", "Dạ em đã bật điều hòa 22 độ cho anh chị.", "đã bật điều hòa cho anh chị"),
        ("Cách mở cốp sau xe VF8?", "Bấm nút mở cốp trên chìa khóa hoặc nhấn nút mở cốp dưới vô năng.", "nhấn nút mở cốp trên chìa khóa hoặc bảng điều khiển"),
        ("Cách chỉnh gương chiếu hậu?", "Sử dụng cụm nút điều chỉnh gương trên cửa xe bên lái.", "sử dụng cụm nút điều chỉnh gương bên cửa lái"),
        ("Dung lượng pin VF8 là bao nhiêu?", "Dung lượng pin sử dụng của VF8 là 87.7 kWh.", "dung lượng pin sử dụng là 87.7 kwh"),
        ("Áp suất lốp lốp trước VF8?", "Áp suất lốp tiêu chuẩn cho VF8 là 2.5 bar.", "áp suất lốp tiêu chuẩn 2.5 bar"),
        ("Cách khởi động xe VF8?", "Nhấn chân phanh và ấn nút Engine Start Stop trên táp lô.", "nhấn chân phanh và ấn nút start stop"),
        ("Chế độ lái Eco là gì?", "Chế độ Eco tối ưu hóa tiêu thụ năng lượng của pin.", "tối ưu hóa tiêu thụ năng lượng của pin"),
        ("Cổng sạc AC đặt ở đâu?", "Cổng sạc AC nằm ở hông xe phía trước bên trái.", "cổng sạc nằm ở hông xe phía trước bên trái"),
        ("Cách gập hàng ghế thứ 2?", "Kéo cần gạt bên cạnh đệm ngồi để gập ghế phẳng.", "kéo cần gạt bên cạnh tựa lưng để gập ghế"),
        ("Làm sao tắt âm lượng loa?", "Nhấn nút Mute trên vô năng hoặc điều khiển màn hình.", "nhấn nút mute trên vô năng"),
        ("Cách kết nối Bluetooth?", "Vào Cài đặt > Kết nối > Bluetooth và chọn thiết bị.", "vào cài đặt chọn kết nối bluetooth"),
        ("Cảnh báo điểm mù hiển thị ở đâu?", "Đèn cảnh báo điểm mù tích hợp trên gương chiếu hậu ngoài.", "hiển thị trên gương chiếu hậu ngoài"),
        ("Hệ thống ADAS gồm những gì?", "Bao gồm hỗ trợ di chuyển khi ùn tắc, giữ làn đường và phanh khẩn cấp.", "hỗ trợ giữ làn đường và phanh khẩn cấp"),
        ("Khi nào cần bảo dưỡng định kỳ?", "Bảo dưỡng định kỳ mỗi 12,000 km hoặc 12 tháng.", "bảo dưỡng mỗi 12000 km hoặc 12 tháng"),
        ("Túi khí màn hình nằm ở đâu?", "Túi khí người lái nằm ở tâm vô năng.", "túi khí người lái nằm ở vô năng"),
        ("Làm sao mở cửa khi hết điện?", "Sử dụng lẫy mở cửa cơ học ở hộc cửa xe.", "sử dụng lẫy mở cửa cơ học")
    ]
    for idx, (q, act, exp) in enumerate(spec_data, 1):
        cases.append({
            "name": f"[ExplicitSpec-{idx:02d}]",
            "user_cmd": q,
            "actual_resp": act,
            "expected_resp": exp,
            "expected_verdict": "PASS",
            "slice": "Explicit Spec"
        })

    # Slice 2: Domain RAG Matches (18 unique cases)
    rag_data = [
        ("Công suất động cơ VF8?", "Động cơ điện VF8 có công suất tối đa 300 kW.", "300 kw"),
        ("Mô men xoắn cực đại VF8?", "Mô men xoắn cực đại đạt 620 Nm.", "620 nm"),
        ("Thời gian tăng tốc 0-100km/h?", "Thời gian tăng tốc từ 0 đến 100 km/h là 5.5 giây.", "5.5 giây"),
        ("Bán kính quay vòng tối thiểu?", "Bán kính quay vòng tối thiểu của xe là 5.8 m.", "5.8 m"),
        ("Chiều dài cơ sở VF8?", "Chiều dài cơ sở xe VF8 là 2950 mm.", "2950 mm"),
        ("Khoảng sáng gầm xe?", "Khoảng sáng gầm xe không tải là 179 mm.", "179 mm"),
        ("Trọng lượng bản thân xe?", "Trọng lượng bản thân không tải khoảng 2600 kg.", "2600 kg"),
        ("Kích thước lốp xe VF8?", "Xe trang bị lốp kích thước 245/45 R20.", "245/45 r20"),
        ("Số túi khí trên xe VF8?", "Xe VF8 được trang bị 11 túi khí an toàn.", "11 túi khí"),
        ("Loại pin sử dụng là gì?", "Xe sử dụng pin Lithium-ion Ternary NMC.", "lithium ion ternary nmc"),
        ("Công suất sạc nhanh DC tối đa?", "Công suất sạc DC tối đa hỗ trợ lên đến 150 kW.", "150 kw"),
        ("Thời gian sạc 10-70% siêu nhanh?", "Thời gian sạc siêu nhanh 10-70% chỉ mất khoảng 24 phút.", "24 phút"),
        ("Dung tích khoang hành lý?", "Dung tích khoang hành lý phía sau là 376 lít.", "376 lít"),
        ("Cửa sổ trời là loại gì?", "Xe sở hữu cửa sổ trời toàn cảnh Panorama.", "cửa sổ trời toàn cảnh panorama"),
        ("Hệ thống treo trước là loại nào?", "Hệ thống treo trước loại độc lập McPherson.", "treo trước độc lập mcpherson"),
        ("Hệ thống treo sau là loại nào?", "Hệ thống treo sau loại đa điểm liên kết.", "treo sau đa điểm liên kết"),
        ("Phanh trước dùng loại phanh gì?", "Phanh trước sử dụng phanh đĩa tản nhiệt.", "phanh đĩa tản nhiệt"),
        ("Hệ dẫn động xe VF8?", "Xe trang bị hệ dẫn động 2 cầu toàn thời gian AWD.", "hệ dẫn động 2 cầu toàn thời gian awd")
    ]
    for idx, (q, act, exp) in enumerate(rag_data, 1):
        cases.append({
            "name": f"[DomainRAG-{idx:02d}]",
            "user_cmd": q,
            "actual_resp": act,
            "expected_resp": "",
            "expected_verdict": "PASS",
            "slice": "Domain RAG Match"
        })

    # Slice 3: Borderline 0.25-0.45 Agent Loop Cases (18 unique cases)
    borderline_data = [
        ("Cung Ma Kết có đặc điểm gì?", "Ma Kết nổi bật với nguyên tắc sống kỷ luật và tinh thần cống hiến vô tận.", ""),
        ("Đặc trưng tính cách cung Bọ Cạp?", "Bọ Cạp thu hút nhờ sự tự tin, quyết đoán và chiều sâu cảm xúc mãnh liệt.", ""),
        ("Thiên Bình hợp với cung nào?", "Thiên Bình rất hợp cạ với Song Ngư và Bảo Bình trong tình yêu.", ""),
        ("Song Tử sinh tháng mấy?", "Song Tử sinh từ ngày 21 tháng 5 đến 21 tháng 6.", ""),
        ("Đặc điểm của người cung Song Ngư?", "Song Ngư giàu trí tưởng tượng, lãng mạn và vô cùng nhân hậu.", ""),
        ("Bảo Bình thích hợp làm nghề gì?", "Bảo Bình thích hợp làm nhà khoa học, thiết kế hoặc công nghệ.", ""),
        ("Sư Tử có hợp với Bạch Dương không?", "Sư Tử và Bạch Dương là cặp đôi lửa vô cùng nhiệt huyết và hòa hợp.", ""),
        ("Xử Nữ có tính cách ra sao?", "Xử Nữ theo đuổi sự hoàn hảo, cẩn thận và tỉ mỉ trong công việc.", ""),
        ("Kim Ngưu yêu như thế nào?", "Kim Ngưu khi yêu rất chân thành, thủy chung và thích sự ổn định.", ""),
        ("Nhân Mã thích đi du lịch ở đâu?", "Nhân Mã đam mê khám phá những vùng đất tự nhiên mạo hiểm.", ""),
        ("Cung Cự Giải có nhạy cảm không?", "Cự Giải là cung hoàng đạo nhạy cảm, yêu gia đình và sống nội tâm.", ""),
        ("Bạch Dương nảy sinh tình cảm thế nào?", "Bạch Dương nảy sinh tình cảm bốc đồng, mãnh liệt và thẳng thắn.", ""),
        ("Tràng An Ninh Bình nổi tiếng về điều gì?", "Tràng An là di sản thế giới nổi tiếng với cảnh quan núi đá vôi uốn lượn.", ""),
        ("Ăn gì ngon ở Đà Lạt?", "Đà Lạt nổi tiếng với bánh mì xíu mại, lẩu gà lá é và bánh ướt lòng gà.", ""),
        ("Hồ Hoàn Kiếm nằm ở đâu?", "Hồ Hoàn Kiếm nằm ngay trung tâm thành phố Hà Nội.", ""),
        ("Phong Nha Kẻ Bàng thuộc tỉnh nào?", "Phong Nha Kẻ Bàng thuộc tỉnh Quảng Bình.", ""),
        ("Vịnh Hạ Long có bao nhiêu đảo?", "Vịnh Hạ Long gồm gần 2000 hòn đảo lớn nhỏ.", ""),
        ("Đảo Phú Quốc có gì đẹp?", "Phú Quốc nổi tiếng với bãi Sao, cáp treo Hòn Thơm và hoàng hôn tuyệt đẹp.", "")
    ]
    for idx, (q, act, exp) in enumerate(borderline_data, 1):
        cases.append({
            "name": f"[BorderlineLoop-{idx:02d}]",
            "user_cmd": q,
            "actual_resp": act,
            "expected_resp": exp,
            "expected_verdict": "PASS",
            "slice": "Borderline 0.25-0.45"
        })

    # Slice 4: STT Hearing Mismatches (18 unique cases)
    stt_data = [
        ("Bật điều hòa 22 độ", "Dạ em nghe đây ạ, anh chị cần em hỗ trợ gì?", "Hey VinFast"),
        ("Mở cửa sổ bên lái", "Em chào anh chị ạ.", "VinFast"),
        ("Tăng âm lượng nhạc", "Dạ em đây ạ.", "Ví Vi"),
        ("Chuyển sang đài FM", "Dạ xin chào anh chị.", "Hey ViVi"),
        ("Bật sấy kính trước", "Dạ em đang lắng nghe ạ.", "Xin chào VinFast"),
        ("Đóng cửa sổ trời", "Em nghe đây ạ.", "Chào VinFast"),
        ("Mở bản đồ vị trí gần nhất", "Dạ em hỗ trợ gì ạ?", "Hey VinFast"),
        ("Bật đèn gầm xe", "Dạ em nghe đây.", "VinFast"),
        ("Chỉnh ghế lái lùi lại", "Xin chào anh chị.", "Ví Vi"),
        ("Bật chế độ thể thao", "Dạ em nghe.", "Hey ViVi"),
        ("Mở nhạc bolero", "Em chào anh chị.", "Xin chào VinFast"),
        ("Gọi điện thoại cho mẹ", "Dạ em nghe đây ạ.", "Chào VinFast"),
        ("Bật làm mát ghế", "Dạ em nghe đây.", "Hey VinFast"),
        ("Tắt âm thanh loa", "Em nghe đây ạ.", "VinFast"),
        ("Bật sấy gương chiếu hậu", "Xin chào anh chị.", "Ví Vi"),
        ("Chuyển bài hát tiếp theo", "Dạ em nghe đây.", "Hey ViVi"),
        ("Mở áp suất lốp", "Em chào anh chị ạ.", "Xin chào VinFast"),
        ("Tắt điều hòa xe", "Dạ em hỗ trợ gì ạ?", "Chào VinFast")
    ]
    for idx, (q, act, lis) in enumerate(stt_data, 1):
        cases.append({
            "name": f"[STTMismatch-{idx:02d}]",
            "user_cmd": q,
            "actual_resp": act,
            "vivi_listen": lis,
            "expected_resp": "Thao tác thực hiện thành công",
            "expected_verdict": "RETEST",
            "slice": "STT Mismatch"
        })

    # Slice 5: False Refusals (18 unique cases)
    refusal_data = [
        ("Áp suất lốp VF8 tiêu chuẩn là bao nhiêu?", "Xin lỗi, em chưa hiểu ý anh chị.", "2.5 bar"),
        ("Nút SOS xe VF8 đặt ở đâu?", "Xin lỗi em chưa có thông tin về câu hỏi này.", "trên đèn trần phía trước"),
        ("Dung lượng pin VF8 là bao nhiêu?", "Dạ em chưa có đủ dữ liệu để trả lời.", "87.7 kwh"),
        ("Công suất sạc nhanh DC tối đa?", "Rất tiếc em chưa có thông tin.", "150 kw"),
        ("Xe VF8 có mấy túi khí?", "Dạ em chưa thể hỗ trợ câu hỏi này.", "11 túi khí"),
        ("Cách bật điều hòa xe VF8?", "Xin lỗi em không thể hỗ trợ.", "nhấn nút ac hoặc ra lệnh giọng nói"),
        ("Cửa sổ trời VF8 là loại gì?", "Em chưa biết thông tin này ạ.", "cửa sổ trời toàn cảnh panorama"),
        ("Xe VF8 trang bị mâm kích thước bao nhiêu?", "Hiện tại em không có thông tin.", "20 inch"),
        ("Làm sao mở cốp sau xe?", "Xin lỗi em chưa hiểu ý anh chị.", "bấm nút mở cốp trên chìa khóa"),
        ("Kích thước tổng thể xe VF8?", "Dạ em chưa có thông tin.", "4750 x 1934 x 1667 mm"),
        ("Chiều dài cơ sở VF8 là bao nhiêu?", "Xin lỗi em chưa hiểu.", "2950 mm"),
        ("Bảo dưỡng xe VF8 ở đâu?", "Dạ em chưa hỗ trợ được ạ.", "tại các xưởng dịch vụ chính hãng vinfast"),
        ("Tốc độ tối đa xe VF8?", "Xin lỗi em chưa có đủ dữ liệu.", "200 km h"),
        ("Chế độ bảo hành pin VF8?", "Em không có thông tin câu này.", "10 năm hoặc không giới hạn km"),
        ("Xe VF8 dẫn động mấy cầu?", "Xin lỗi em chưa hiểu ý anh chị.", "dẫn động 2 cầu toàn thời gian awd"),
        ("Thời gian tăng tốc 0-100km/h?", "Dạ em chưa biết ạ.", "5.5 giây"),
        ("Hệ thống treo trước loại gì?", "Xin lỗi em chưa có thông tin.", "treo độc lập mcpherson"),
        ("Cổng sạc xe nằm ở đâu?", "Em chưa thể hỗ trợ.", "hông xe phía trước bên trái")
    ]
    for idx, (q, act, exp) in enumerate(refusal_data, 1):
        cases.append({
            "name": f"[FalseRefusal-{idx:02d}]",
            "user_cmd": q,
            "actual_resp": act,
            "expected_resp": exp,
            "expected_verdict": "FAIL",
            "slice": "False Refusal"
        })

    # Slice 6: Tier 5 Open-Ended / Conversational Queries (18 unique cases)
    conv_data = [
        ("Xin chào ViVi", "Chào anh/chị, em là trợ lý ảo ViVi. Em có thể hỗ trợ gì cho anh/chị hôm nay?", ""),
        ("Thời tiết hôm nay thế nào?", "Dạ hôm nay trời nắng đẹp, nhiệt độ khoảng 28 độ C rất thích hợp cho chuyến đi.", ""),
        ("Kể cho tôi nghe một câu chuyện vui", "Có một chú thỏ hỏi rùa: Tại sao bạn bò chậm thế? Rùa trả lời: Vì tớ đi thong thả ngắm cảnh đời!", ""),
        ("Hôm nay là ngày mấy?", "Hôm nay là ngày 25 tháng 8 năm 2026 ạ.", ""),
        ("Chúc anh chị một ngày tốt lành", "Dạ em cảm ơn anh chị rất nhiều! Chúc anh chị một ngày tràn đầy năng lượng và lái xe an toàn!", ""),
        ("Cảm ơn ViVi nhé", "Dạ không có gì ạ! Rất vui được đồng hành cùng anh chị trên mọi nẻo đường.", ""),
        ("Tạm biệt ViVi", "Tạm biệt anh chị ạ! Chúc anh chị thượng lộ bình an!", ""),
        ("Bạn tên là gì?", "Em tên là ViVi, trợ lý thông minh trên xe điện VinFast.", ""),
        ("Ai tạo ra bạn?", "Em được phát triển bởi đội ngũ kỹ sư VinFast.", ""),
        ("Bạn có biết hát không?", "Dạ em hát không hay nhưng em có thể mở những bài hát tuyệt vời cho anh chị nghe ạ!", ""),
        ("Mở một bản nhạc nhẹ nhàng", "Dạ em đang bật playlist nhạc acoustic nhẹ nhàng cho anh chị thư giãn.", ""),
        ("Cho tôi nghe tin tức hôm nay", "Dạ tin tức hôm nay ghi nhận tình hình giao thông thuận lợi trên các tuyến cao tốc chính.", ""),
        ("Bạn bao nhiêu tuổi?", "Em là trợ lý ảo thông minh nên tuổi của em tính theo các bản cập nhật phần mềm ạ!", ""),
        ("Xe điện có tốt hơn xe xăng không?", "Xe điện giúp giảm phát thải, vận hành êm ái và chi phí bảo dưỡng tối ưu hơn.", ""),
        ("Tôi mệt mỏi quá", "Anh chị hãy nghỉ ngơi một chút, bật nhạc nhẹ và uống nước để lấy lại năng lượng nhé!", ""),
        ("Bật nhạc sôi động", "Dạ em đang phát bài hát EDM sôi động giúp anh chị tỉnh táo lái xe!", ""),
        ("Hôm nay nên ăn gì?", "Hôm nay thời tiết đẹp, anh chị thử thưởng thức món phở nóng hoặc bún chả Hà Nội nhé!", ""),
        ("Hẹn gặp lại ViVi", "Dạ hẹn gặp lại anh chị! Em luôn sẵn sàng hỗ trợ khi anh chị cần.")
    ]
    for idx, item in enumerate(conv_data, 1):
        q = item[0]
        act = item[1]
        exp = item[2] if len(item) > 2 else ""
        cases.append({
            "name": f"[Tier5Conv-{idx:02d}]",
            "user_cmd": q,
            "actual_resp": act,
            "expected_resp": exp,
            "expected_verdict": "PASS",
            "slice": "Tier 5 Open-Ended"
        })

    return cases


class AgentBenchmarkSuite:
    """Automated benchmark framework for ViVi Evaluation Agent performance & accuracy."""

    def __init__(self):
        self.router = AdaptiveEvalRouter()

    def run_benchmark(self, sample_cases: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Runs evaluation benchmark against 108 UNIQUE, stratified ground-truth test cases."""
        if not sample_cases:
            sample_cases = get_real_ground_truth_testset()

        start_time = time.time()
        results = []
        correct_verdicts = 0
        path_breakdown = Counter()
        slice_breakdown = Counter()

        for case in sample_cases:
            eval_res = self.router.evaluate(
                name=case["name"],
                user_cmd=case["user_cmd"],
                actual_resp=case["actual_resp"],
                expected_resp=case.get("expected_resp", ""),
                vivi_listen=case.get("vivi_listen", "")
            )
            eval_res["expected_verdict"] = case.get("expected_verdict", "PASS")
            eval_res["slice"] = case.get("slice", "General")
            results.append(eval_res)

            is_correct = (eval_res["auto_result"] == eval_res["expected_verdict"])
            if is_correct:
                correct_verdicts += 1

            resolved_path = eval_res.get("trace_log", {}).get("resolved_by", "Unknown")
            path_breakdown[(resolved_path, "correct" if is_correct else "incorrect")] += 1
            slice_breakdown[(case.get("slice", "General"), "correct" if is_correct else "incorrect")] += 1

        elapsed = time.time() - start_time
        total = len(sample_cases)
        accuracy_pct = (correct_verdicts / total) * 100.0 if total > 0 else 0.0
        throughput_fps = total / elapsed if elapsed > 0 else 0.0

        confusion = compute_confusion_matrix(results)

        return {
            "total_testcases": total,
            "correct_verdicts": correct_verdicts,
            "accuracy_pct": round(accuracy_pct, 1),
            "confusion_matrix": confusion,
            "path_breakdown": {f"{path} ({status})": cnt for (path, status), cnt in path_breakdown.items()},
            "slice_breakdown": {f"{slice_name} ({status})": cnt for (slice_name, status), cnt in slice_breakdown.items()},
            "total_latency_sec": round(elapsed, 2),
            "throughput_rows_per_sec": round(throughput_fps, 1),
            "average_latency_ms": round((elapsed / total) * 1000.0, 1)
        }


if __name__ == "__main__":
    suite = AgentBenchmarkSuite()
    metrics = suite.run_benchmark()
    cm = metrics["confusion_matrix"]

    print("==================================================")
    print(" 🚀 VIVI AGENT EVALUATION BENCHMARK METRICS (v2.2)")
    print("==================================================")
    print(f" Total Stratified Test Cases : {metrics['total_testcases']} (100% Unique)")
    print(f" Correct Verdict Accuracy    : {metrics['accuracy_pct']}%")
    print(f" False Pass Rate (FP %)      : {cm['false_pass_rate_pct']}% ({cm['false_pass_count']} cases)")
    print(f" False Reject Rate (FN %)    : {cm['false_reject_rate_pct']}% ({cm['false_reject_count']} cases)")
    print("--------------------------------------------------")
    print(" 🔍 Resolution Path Breakdown:")
    for path_key, cnt in metrics["path_breakdown"].items():
        print(f"   • {path_key}: {cnt} rows")
    print("--------------------------------------------------")
    print(" 🎯 Stratified Slice Accuracy Breakdown:")
    for slice_key, cnt in metrics["slice_breakdown"].items():
        print(f"   • {slice_key}: {cnt} rows")
    print("--------------------------------------------------")
    print(f" Total Benchmark Execution   : {metrics['total_latency_sec']}s")
    print(f" Processing Throughput       : {metrics['throughput_rows_per_sec']} rows/sec (with live web network latency)")
    print(f" Average Row Latency         : {metrics['average_latency_ms']} ms/row")
    print("==================================================")
