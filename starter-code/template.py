"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
import re
from typing import Any, Dict, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""


class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        if self.api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(
                    f"Bạn là chatbot tư vấn du lịch. Hãy trả lời KHÔNG dùng tool hay internet: {user_input}"
                )
                answer = response.text
            except Exception as e:
                answer = f"[Chatbot Baseline] Lỗi khi gọi LLM: {e}"
        else:
            answer = f"[Chatbot Baseline] Trả lời cho: {user_input}"

        return {
            "status": "success",
            "answer": answer,
            "tool_calls": [],  # Baseline không bao giờ sử dụng tool
        }


class ReActAgent:
    """Production-grade ReAct Agent with Tool Registry and Safeguards
    (Rule-based reasoning — KHÔNG cần gọi LLM / API Key)"""

    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    # ---------- Helpers phân tích câu hỏi (thay cho việc gọi LLM) ----------

    def parse_city_code(self, text: str) -> str:
        text_upper = text.upper()
        for code in ["SGN", "HAN", "DAD"]:
            if code in text_upper:
                return code
        if "HÀ NỘI" in text_upper:
            return "HAN"
        if "HỒ CHÍ MINH" in text_upper or "SÀI GÒN" in text_upper:
            return "SGN"
        if "ĐÀ NẴNG" in text_upper:
            return "DAD"
        return "SGN"

    def _parse_flight_request(self, text: str) -> Dict[str, Any]:
        """Rule-based: tách origin/destination/max_price từ câu hỏi."""
        text_upper = text.upper()
        codes_found = []
        for code in ["HAN", "SGN", "DAD"]:
            if code in text_upper and code not in codes_found:
                codes_found.append(code)
        city_map = [
            ("HÀ NỘI", "HAN"),
            ("HỒ CHÍ MINH", "SGN"),
            ("SÀI GÒN", "SGN"),
            ("ĐÀ NẴNG", "DAD"),
        ]
        for name, code in city_map:
            if name in text_upper and code not in codes_found:
                codes_found.append(code)

        origin = codes_found[0] if len(codes_found) >= 1 else "HAN"
        destination = codes_found[1] if len(codes_found) >= 2 else "SGN"

        price_match = re.search(r"(\d+(?:[.,]\d+)?)\s*tri[eệ]u", text, re.IGNORECASE)
        max_price = (
            int(float(price_match.group(1).replace(",", ".")) * 1_000_000)
            if price_match
            else 5_000_000
        )
        return {"origin": origin, "destination": destination, "max_price": max_price}

    def _determine_needed_actions(self, user_input: str) -> List[str]:
        """Xác định trước danh sách tool cần gọi để trả lời đầy đủ câu hỏi."""
        lower = user_input.lower()

        # Câu hỏi về chính sách/quy định không cần tool nào, kể cả khi có nhắc "vé"
        faq_markers = ["chính sách", "đổi trả", "quy định", "điều khoản"]
        if any(marker in lower for marker in faq_markers):
            return []

        actions = []
        if "chuyến bay" in lower or "tìm vé" in lower or "đặt vé" in lower or "giá vé" in lower:
            actions.append("get_flight_info")
        if "thời tiết" in lower or "mặc" in lower or "trang phục" in lower:
            actions.append("get_weather_forecast")
        return actions

    def _build_faq_answer(self, user_input: str) -> str:
        """Không có tool nào phù hợp -> trả lời chung chung (FAQ)."""
        return (
            "Xin lỗi, tôi chưa có công cụ tra cứu chính sách của Vinpearl. "
            "Vui lòng liên hệ tổng đài chăm sóc khách hàng Vinpearl để được tư vấn "
            f"chi tiết cho câu hỏi: \"{user_input}\""
        )

    def _build_final_answer(self, flight_result, weather_result, destination_code: str) -> str:
        """Tổng hợp câu trả lời cuối, dạng đánh số dễ đọc (không cần LLM)."""
        sections = []

        if flight_result is not None:
            lines = ["1. Thông tin chuyến bay:"]
            if flight_result:
                for f in flight_result:
                    airline = f.get("airline", "N/A")
                    flight_no = f.get("flight_number") or f.get("flight_no", "")
                    dep_time = f.get("departure_time", "")
                    price = f.get("price_vnd", 0)
                    label = f"{airline} ({flight_no})" if flight_no else airline
                    lines.append(f"   - {label}: {dep_time} - Giá: {price:,} VNĐ")
            else:
                lines.append("   - Không tìm thấy chuyến bay nào phù hợp với yêu cầu.")
            sections.append("\n".join(lines))

        if weather_result is not None:
            lines = ["Thông tin thời tiết & trang phục:"]
            if isinstance(weather_result, dict) and "error" not in weather_result:
                city = weather_result.get("city", destination_code)
                temp = weather_result.get("temperature_c", weather_result.get("temp_c", "?"))
                condition = weather_result.get("condition", "")
                recommendation = weather_result.get("recommendation") or weather_result.get(
                    "suggestion", ""
                )
                lines.append(f"   - Thời tiết tại {city}: {temp}°C ({condition}).")
                lines.append(f"   - Gợi ý trang phục: {recommendation}")
            else:
                lines.append(f"   - Không có dữ liệu thời tiết cho {destination_code}.")

            prefix = "2. " if flight_result is not None else "1. "
            sections.append(prefix + "\n".join(lines))

        return "\n\n".join(sections) if sections else "Không có đủ thông tin để trả lời."

    def _execute_action(self, action: Dict[str, Any]) -> Any:
        """Thực thi tool trong TOOL_MAP (Safeguard: kiểm tra tool có tồn tại trong Registry)."""
        tool_name = action.get("name")
        if tool_name not in TOOL_MAP:
            return {"error": f"Tool '{tool_name}' không tồn tại trong TOOL_MAP."}
        try:
            return TOOL_MAP[tool_name](**action.get("args", {}))
        except Exception as e:
            return {"error": str(e)}

    # ---------- Vòng lặp chính ----------

    def run(self, user_input: str) -> Dict[str, Any]:
        # TODO 1: Khởi tạo mảng lưu lịch sử conversation / traces
        self.trace = []
        needed_actions = self._determine_needed_actions(user_input)

        # Trường hợp không cần tool nào (FAQ) -> trả lời trực tiếp, 1 iteration
        if not needed_actions:
            iteration = 1
            final_answer = self._build_faq_answer(user_input)
            self.trace.append(
                {
                    "iteration": iteration,
                    "thought": "Câu hỏi này không liên quan tới các tool hiện có, tôi sẽ trả lời trực tiếp.",
                    "final_answer": final_answer,
                }
            )
            return {
                "status": "completed",
                "iterations": iteration,
                "trace": self.trace,
                "answer": final_answer,
            }

        flight_result = None
        weather_result = None
        destination_code = None
        iteration = 0
        status = "completed"

        # TODO 2: Thiết lập vòng lặp, giới hạn bởi self.max_iterations (safeguard)
        for idx, action_name in enumerate(needed_actions):
            if iteration >= self.max_iterations:
                status = "max_iterations_reached"
                break
            iteration += 1

            # TODO 3: Phân tích Thought / Action — rule-based cho từng loại tool
            if action_name == "get_flight_info":
                params = self._parse_flight_request(user_input)
                destination_code = params["destination"]
                thought = (
                    f"Tôi cần tìm chuyến bay từ {params['origin']} đến {params['destination']} "
                    f"giá dưới {params['max_price']:,} VND."
                )
                action = {"name": "get_flight_info", "args": params}
            else:  # get_weather_forecast
                destination_code = destination_code or self.parse_city_code(user_input)
                thought = f"Tôi cần kiểm tra thông tin thời tiết tại {destination_code}."
                action = {"name": "get_weather_forecast", "args": {"city_code": destination_code}}

            # TODO 4: Thực thi Tool trong TOOL_MAP nếu có Action
            result = self._execute_action(action)

            if action["name"] == "get_flight_info":
                flight_result = result
            else:
                weather_result = result

            is_last_action = idx == len(needed_actions) - 1
            if is_last_action and len(needed_actions) == 1:
                # Chỉ cần đúng 1 tool -> gộp luôn Final Answer vào cùng iteration này
                final_answer = self._build_final_answer(flight_result, weather_result, destination_code)
                self.trace.append(
                    {
                        "iteration": iteration,
                        "thought": thought,
                        "action": action,
                        "observation": result,
                        "final_answer": final_answer,
                    }
                )
                return {
                    "status": "completed",
                    "iterations": iteration,
                    "trace": self.trace,
                    "answer": final_answer,
                }

            # TODO 5: Ghi lại Observation, lặp lại cho tới khi ra Final Answer
            self.trace.append(
                {
                    "iteration": iteration,
                    "thought": thought,
                    "action": action,
                    "observation": result,
                }
            )

        # Cần >1 tool -> dành riêng 1 iteration cuối để tổng hợp câu trả lời
        if status != "max_iterations_reached" and iteration >= self.max_iterations:
            status = "max_iterations_reached"

        if status == "max_iterations_reached":
            partial_answer = self._build_final_answer(flight_result, weather_result, destination_code)
            return {
                "status": "max_iterations_reached",
                "iterations": iteration,
                "trace": self.trace,
                "answer": partial_answer,
            }

        iteration += 1
        final_answer = self._build_final_answer(flight_result, weather_result, destination_code)
        self.trace.append(
            {
                "iteration": iteration,
                "thought": "Tôi đã thu thập đủ thông tin để trả lời khách hàng.",
                "final_answer": final_answer,
            }
        )
        return {
            "status": "completed",
            "iterations": iteration,
            "trace": self.trace,
            "answer": final_answer,
        }


def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    chatbot_result = chatbot.query(user_query)
    print(json.dumps(chatbot_result, indent=2, ensure_ascii=False))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    agent_result = agent.run(user_query)
    print(json.dumps(agent_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()