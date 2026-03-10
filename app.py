import streamlit as st
import os
import re
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

model = ChatOpenAI(
    model="deepseek/deepseek-r1",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0
) 
search = TavilySearchResults()

class ReWOO(TypedDict):
    task: str
    plan_string: str # output cúa planner 
    steps: List[Any] # Danh sách các bước đã parse từ plan
    results: Dict[str, str] # Dictionary lưu kết quả của từng bước
    result: str # Kết quả cuối cùng sau khi thực hiện tất cả các bước

# PLANNER 
prompt = """Đối với nhiệm vụ sau, hãy lập kế hoạch để giải quyết vấn đề từng bước.
Đối với mỗi kế hoạch (plan), hãy chỉ rõ công cụ bên ngoài cần sử dụng cùng với input của công cụ để thu thập thông tin.
Bạn có thể lưu thông tin thu được vào một biến #E để có thể sử dụng lại ở các bước sau.
(Plan, #E1, Plan, #E2, Plan, ...)

Các công cụ có thể sử dụng:

(1) Google[input]: Worker dùng để tìm kiếm kết quả từ Google.
Hữu ích khi bạn cần tìm câu trả lời ngắn gọn và chính xác về một chủ đề cụ thể.
Input phải là một truy vấn tìm kiếm.

(2) LLM[input]: Một mô hình ngôn ngữ đã được huấn luyện trước giống như bạn.
Hữu ích khi bạn cần sử dụng kiến thức chung và suy luận thông thường.
Ưu tiên sử dụng khi bạn tự tin có thể giải quyết vấn đề bằng suy luận.
Input có thể là bất kỳ hướng dẫn nào.

Ví dụ:

Task: Thomas, Toby và Rebecca làm tổng cộng 157 giờ trong một tuần.
Thomas làm x giờ.
Toby làm ít hơn 10 giờ so với gấp đôi số giờ Thomas làm,
và Rebecca làm ít hơn Toby 8 giờ.
Rebecca đã làm bao nhiêu giờ?

Plan: Vì Thomas làm x giờ, hãy chuyển bài toán thành biểu thức đại số và giải bằng Wolfram Alpha.
#E1 = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]

Plan: Tìm số giờ Thomas đã làm.
#E2 = LLM[What is x, given #E1]

Plan: Tính số giờ Rebecca đã làm.
#E3 = Calculator[(2 * #E2 − 10) − 8]

Begin!
Hãy mô tả kế hoạch của bạn với đầy đủ chi tiết.
Mỗi Plan phải đi kèm đúng một biến #E.

Task: {task}"""

regex_pattern = r"Plan:\s*(.*?)\s*\n\s*(#E\d+)\s*=\s*([A-Za-z_]\w*)\s*\[(.*?)\]"
planner_prompt = ChatPromptTemplate.from_messages([("system", prompt)])
planner = planner_prompt | model

def get_plan(state: ReWOO):
    result = planner.invoke({"task": state["task"]})
    matches = re.findall(regex_pattern, result.content, re.DOTALL)
    return {"steps": matches, "plan_string": result.content}

# WORKER (TOOL EXECUTION)
def _get_current_task(state: ReWOO):
    if state.get("results") is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    return len(state["results"]) + 1

def tool_execution(state: ReWOO):
    _step = _get_current_task(state)
    _, step_name, tool, tool_input = state["steps"][_step - 1]
    
    _results = state.get("results") or {}
    
    for k, v in _results.items():
        tool_input = tool_input.replace(k, v)
        
    if tool == "Google":
        raw_result = search.invoke(tool_input)
        tool_result = str(raw_result)
    elif tool == "LLM":
        raw_result = model.invoke(tool_input)
        tool_result = raw_result.content
    else:
        tool_result = "Tool not found or unsupported."

    _results[step_name] = tool_result
    return {"results": _results}

# SOLVER 
solve_prompt = """Hãy giải quyết nhiệm vụ hoặc vấn đề sau đây. Để giải quyết vấn đề, chúng tôi đã lập một Kế hoạch (Plan) từng bước và truy xuất các Bằng chứng (Evidence) tương ứng cho mỗi bước. Hãy sử dụng chúng một cách cẩn trọng vì các bằng chứng dài có thể chứa thông tin không liên quan.

{plan}

Bây giờ, hãy giải quyết câu hỏi hoặc nhiệm vụ dựa trên các Bằng chứng đã được cung cấp ở trên. Trả lời trực tiếp kết quả, không thêm từ ngữ dư thừa.

Task: {task}
Response:"""

def solve(state: ReWOO):
    plan = ""
    for _plan, step_name, tool, tool_input in state["steps"]:
        _results = state.get("results") or {}
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
            step_name = step_name.replace(k, v)
        plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]\nKết quả thu được: {_results.get(step_name, '')}\n\n"
        
    prompt_formatted = solve_prompt.format(plan=plan, task=state["task"])
    result = model.invoke(prompt_formatted)
    return {"result": result.content}

# ROUTER & GRAPH COMPILATION 
def _route(state):
    _step = _get_current_task(state)
    if _step is None:
        return "solve"
    return "tool"

def format_runtime_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()

    if "401" in message or "authenticationerror" in lowered or "user not found" in lowered:
        return (
            "OpenRouter đang trả về lỗi 401. Kiểm tra lại biến môi trường "
            "OPENROUTER_API_KEY trong file .env hoặc terminal hiện tại. "
            "Key hiện tại có thể sai, hết hạn, hoặc không thuộc đúng tài khoản OpenRouter."
        )

    if "tavily" in lowered and ("api" in lowered or "auth" in lowered or "key" in lowered):
        return "Tavily API lỗi. Kiểm tra lại TAVILY_API_KEY trong môi trường chạy Streamlit."

    return f"Đã xảy ra lỗi khi chạy agent: {message}"

# Build graph (stateless, không có checkpointer)
graph = StateGraph(ReWOO)
graph.add_node("plan", get_plan)
graph.add_node("tool", tool_execution)
graph.add_node("solve", solve)
graph.add_edge(START, "plan")
graph.add_edge("plan", "tool")
graph.add_conditional_edges("tool", _route)
graph.add_edge("solve", END)

# ── UI CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ReWOO Agent", page_icon="🤖", layout="centered")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 760px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Nền tổng thể */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 1rem; }

/* Section wrapper */
.panel {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
    backdrop-filter: blur(8px);
}

/* Badge trạng thái */
.badge-wait  { display:inline-block; background:#f59e0b22; color:#fbbf24;
               border:1px solid #f59e0b55; border-radius:99px;
               padding:4px 14px; font-size:0.82rem; font-weight:600; }
.badge-done  { display:inline-block; background:#10b98122; color:#34d399;
               border:1px solid #10b98155; border-radius:99px;
               padding:4px 14px; font-size:0.82rem; font-weight:600; }

/* Input */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.94) !important;
    border: 1px solid rgba(255,255,255,0.30) !important;
    border-radius: 10px !important;
    color: #0f172a !important;
    caret-color: #4f46e5 !important;
    font-size: 1rem;
    padding: 10px 14px !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

/* Textarea */
textarea {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #e0e0e0 !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.88rem !important;
}

/* Buttons */
[data-testid="stButton"] > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    border: none !important;
}
[data-testid="stButton"] > button:first-child {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}
[data-testid="stButton"] > button:hover {
    filter: brightness(1.15) !important;
    transform: translateY(-1px) !important;
}

/* Evidence items */
.evidence-item {
    background: rgba(255,255,255,0.05);
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.88rem;
    word-break: break-word;
}
.evidence-key {
    color: #a78bfa;
    font-weight: 700;
    font-family: monospace;
}

/* Result box */
.result-box {
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 12px;
    padding: 20px 24px;
    font-size: 1.05rem;
    line-height: 1.7;
    color: #d1fae5;
}

/* Step pills */
.step-pill {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 99px;
    padding: 2px 12px;
    font-size: 0.78rem;
    color: #a5b4fc;
    margin-right: 6px;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 12px 0 4px 0;">
    <div style="font-size:3rem;">🤖</div>
    <h1 style="font-size:2.2rem; font-weight:800; margin:0;
               background: linear-gradient(90deg,#818cf8,#c084fc);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        ReWOO Agent
    </h1>
    <p style="color:#94a3b8; margin-top:6px; font-size:0.95rem;">
        Lập kế hoạch · Phê duyệt · Thực thi — Không cần quan sát lặp lại
    </p>
</div>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────────────────────────
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
if "app" not in st.session_state:
    st.session_state.app = graph.compile(
        checkpointer=st.session_state.memory,
        interrupt_after=["plan"]
    )
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "rewoo_session_1"
if "last_error" not in st.session_state:
    st.session_state.last_error = ""

app = st.session_state.app
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ── INPUT ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📝 Nhập nhiệm vụ</div>', unsafe_allow_html=True)
task = st.text_input(
    "Nhập câu hỏi hoặc nhiệm vụ",
    placeholder="Ví dụ: Đội nào vô địch Ngoại hạng Anh năm 2023 và HLV của họ bao nhiêu tuổi?",
)

btn_plan = st.button("Lên Kế Hoạch 🚀", use_container_width=True)

if st.session_state.last_error:
    st.error(st.session_state.last_error)

if btn_plan:
    if not OPENROUTER_API_KEY:
        st.session_state.last_error = (
            "Chưa tìm thấy OPENROUTER_API_KEY. Thêm key vào file .env hoặc môi trường chạy Streamlit."
        )
    elif task:
        try:
            st.session_state.last_error = ""
            with st.spinner("🧠 DeepSeek đang phân tích và lập kế hoạch..."):
                app.invoke({"task": task}, config)
        except Exception as exc:
            st.session_state.last_error = format_runtime_error(exc)
    else:
        st.warning("Vui lòng nhập câu hỏi!")

# ── STATE ──────────────────────────────────────────────────────────────────────
current_state = app.get_state(config)

# ── PENDING APPROVAL ───────────────────────────────────────────────────────────
if current_state.next == ('tool',):
    raw_plan_string = current_state.values.get("plan_string", "")
    clean_plan_string = re.sub(r'<think>.*?</think>', '', raw_plan_string, flags=re.DOTALL).strip()

    steps = current_state.values.get("steps", [])

    st.markdown(
        '<span class="badge-wait">⏳ Chờ phê duyệt</span>'
        '<span style="font-size:1.15rem; font-weight:700; margin-left:12px;">Kế hoạch đề xuất</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Hiển thị các bước dạng pill summary
    if steps:
        pills = "".join(
            f'<span class="step-pill">{s[1]} · {s[2]}</span>' for s in steps
        )
        st.markdown(f"<div style='margin-bottom:12px'>{pills}</div>", unsafe_allow_html=True)

    edited_plan_string = st.text_area(
        "Chỉnh sửa kế hoạch (tuỳ chọn) hoặc giữ nguyên và bấm Phê duyệt:",
        value=clean_plan_string,
        height=280,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("✅ Phê duyệt & Thực thi", use_container_width=True):
            new_matches = re.findall(regex_pattern, edited_plan_string, re.DOTALL)
            app.update_state(config, {"plan_string": edited_plan_string, "steps": new_matches})
            try:
                st.session_state.last_error = ""
                with st.spinner("⚙️ Đang thực thi các công cụ..."):
                    app.invoke(None, config)
            except Exception as exc:
                st.session_state.last_error = format_runtime_error(exc)
            st.rerun()
    with col2:
        if st.button("❌ Huỷ", use_container_width=True):
            st.session_state.thread_id = st.session_state.thread_id + "_new"
            st.session_state.last_error = ""
            st.rerun()

# ── RESULT ─────────────────────────────────────────────────────────────────────
elif current_state.values.get("result"):
    st.markdown(
        '<span class="badge-done">✅ Hoàn thành</span>'
        '<span style="font-size:1.15rem; font-weight:700; margin-left:12px;">Kết quả</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    raw_result = current_state.values.get("result", "")
    clean_result = re.sub(r'<think>.*?</think>', '', raw_result, flags=re.DOTALL).strip()
    st.markdown(f'<div class="result-box">{clean_result}</div>', unsafe_allow_html=True)

    st.markdown("")
    with st.expander("🔍 Chi tiết bằng chứng (Evidence)"):
        results_dict = current_state.values.get("results", {})
        for key, val in results_dict.items():
            st.markdown(
                f'<div class="evidence-item"><span class="evidence-key">{key}</span><br>{val}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    if st.button("🔄 Làm nhiệm vụ mới", use_container_width=True):
        st.session_state.thread_id = st.session_state.thread_id + "_new"
        st.session_state.last_error = ""
        st.rerun()