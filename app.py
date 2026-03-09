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

model = ChatOpenAI(
    model="deepseek/deepseek-r1",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
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

# Build graph (stateless, không có checkpointer)
graph = StateGraph(ReWOO)
graph.add_node("plan", get_plan)
graph.add_node("tool", tool_execution)
graph.add_node("solve", solve)
graph.add_edge(START, "plan")
graph.add_edge("plan", "tool")
graph.add_conditional_edges("tool", _route)
graph.add_edge("solve", END)

# UI 
st.set_page_config(page_title="ReWOO Agent", page_icon="🤖", layout="centered")
st.title("🤖 ReWOO Agent")
st.markdown("Hệ thống sẽ lập kế hoạch và xin phép bạn phê duyệt hoặc chỉnh sửa trước khi bắt đầu tìm kiếm.")

# Lưu memory và app vào session_state để tồn tại qua mỗi lần rerun
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
if "app" not in st.session_state:
    st.session_state.app = graph.compile(
        checkpointer=st.session_state.memory, 
        interrupt_after=["plan"]
    )

app = st.session_state.app

# Khởi tạo thread_id để LangGraph nhớ ngữ cảnh phiên làm việc
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "rewoo_session_1"

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Nhận yêu cầu từ người dùng
task = st.text_input("Nhập câu hỏi/nhiệm vụ của bạn:", placeholder="Ví dụ: Đội nào vô địch giải Ngoại hạng Anh năm 2023 và HLV của họ bao nhiêu tuổi?")

if st.button("Lên Kế Hoạch 🚀"):
    if task:
        with st.spinner("DeepSeek đang suy nghĩ lộ trình..."):
            app.invoke({"task": task}, config)
    else:
        st.warning("Vui lòng nhập câu hỏi!")

# Lấy trạng thái hiện tại của đồ thị
current_state = app.get_state(config)

# Xử lý Logic Hiển thị dựa trên State của Đồ thị
if current_state.next == ('tool',):
    st.warning("⚠️ Hệ thống đang chờ bạn phê duyệt kế hoạch!")
    
    st.subheader("Kế hoạch được đề xuất:")
    
    # Xoá chuỗi <think>...</think>=
    raw_plan_string = current_state.values.get("plan_string", "")
    clean_plan_string = re.sub(r'<think>.*?</think>', '', raw_plan_string, flags=re.DOTALL).strip()
    
    # Cho phép người dùng chỉnh sửa trực tiếp text của kế hoạch
    edited_plan_string = st.text_area(
        "Bạn có thể sửa trực tiếp nội dung dưới đây, hoặc giữ nguyên và bấm Phê duyệt.", 
        value=clean_plan_string, 
        height=300
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Phê duyệt & Thực thi"):
            # Cập nhật lại state với kế hoạch đã sửa (nếu có)
            new_matches = re.findall(regex_pattern, edited_plan_string, re.DOTALL)
            app.update_state(config, {"plan_string": edited_plan_string, "steps": new_matches})
                
            with st.spinner("Đang thực thi các công cụ. Quá trình này có thể mất chút thời gian..."):
                # Resume (Chạy tiếp) đồ thị bằng cách truyền `None` vào invoke
                app.invoke(None, config)
            st.rerun() # Tải lại trang để hiện kết quả

    with col2:
        if st.button("❌ Huỷ bỏ"):
            # Reset lại state bằng cách đổi thread_id
            st.session_state.thread_id = st.session_state.thread_id + "_new"
            st.rerun()

# Nếu đồ thị đã hoàn thành (không có node tiếp theo và có kết quả)
elif current_state.values.get("result"):
    st.success("🎉 Nhiệm vụ đã hoàn thành!")
    
    with st.expander("Xem chi tiết quá trình thu thập bằng chứng (Evidence)"):
        results_dict = current_state.values.get("results", {})
        for key, val in results_dict.items():
            st.markdown(f"**{key}**: {val}")
            
    st.subheader("💡 Kết luận cuối cùng:")
    
    # Làm sạch <think> của Solver (nếu có)
    raw_result = current_state.values.get("result", "")
    clean_result = re.sub(r'<think>.*?</think>', '', raw_result, flags=re.DOTALL).strip()
    st.info(clean_result)
    
    if st.button("Làm nhiệm vụ mới"):
        st.session_state.thread_id = st.session_state.thread_id + "_new"
        st.rerun()