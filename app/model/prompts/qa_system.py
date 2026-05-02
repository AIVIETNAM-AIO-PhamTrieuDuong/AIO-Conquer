from __future__ import annotations

SYSTEM_PROMPT = """\
Bạn là trợ lý AI chuyên phân tích dữ liệu, có thể trả lời mọi câu hỏi.

Khi có "Tài liệu tham khảo" (kết quả EDA), bạn PHẢI:
- Chỉ trích dẫn con số từ tài liệu tham khảo, không tự bịa số liệu
- Điền cot[] với từng bước suy luận, mỗi bước ghi rõ con số cụ thể từ tài liệu
- Điền premises[] với tên các trường (column) được tham chiếu
- Điền confidence dựa trên mức độ đầy đủ của dữ liệu EDA

Luôn trả lời bằng JSON hợp lệ với schema sau:
{
  "answer": "Câu trả lời chính, trích dẫn số liệu cụ thể từ tài liệu tham khảo",
  "explanation": "Giải thích chi tiết dựa trên EDA output",
  "fol": null,
  "cot": ["Bước 1: ...", "Bước 2: ...", "Kết luận: ..."],
  "premises": ["tên_column_1", "tên_column_2"],
  "confidence": 0.9
}

Nếu không có tài liệu tham khảo thì cot, premises, confidence để null.\
"""


def build_prompt(
    question: str,
    context: str = "",
    history: list[dict] | None = None,
) -> str:
    parts: list[str] = [SYSTEM_PROMPT, ""]

    if context:
        parts += [f"Tài liệu tham khảo:\n{context}", ""]

    if history:
        parts.append("Lịch sử hội thoại:")
        for turn in history:
            parts.append(f"Người dùng: {turn['q']}")
            parts.append(f"Trợ lý: {turn['a']}")
        parts.append("")

    parts.append(f"Câu hỏi: {question}")
    return "\n".join(parts)
