import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE") 

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = ROOT / "benchmark_questions.json"
OUTPUT_DIR = ROOT / "benchmark_outputs"

PLAN_REGEX = r"Plan:\s*(.*?)\s*\n\s*(#E\d+)\s*=\s*([A-Za-z_]\w*)\s*\[(.*?)\]"

REWOO_PLANNER_PROMPT = """Bạn là planner cho hệ thống ReWOO.
Tạo kế hoạch bằng đúng format sau:
Plan: <reason>
#E1 = Tool[input]

Chỉ được dùng 2 tool: Google[input] và LLM[input].
Dùng 3 đến 5 bước. Có thể tham chiếu #E1, #E2.
Không viết gì ngoài các dòng Plan/#E.

Task: {task}
"""

REWOO_SOLVER_PROMPT = """Hãy trả lời câu hỏi cuối cùng dựa trên kế hoạch và bằng chứng bên dưới.

{plan}

Task: {task}

Trả lời thật ngắn gọn và trực tiếp.
"""

REACT_SYSTEM_PROMPT = """Bạn là một ReAct agent.
Bạn giải bằng vòng lặp Action/Observation.
Chỉ được dùng các action sau:
- Google[query]
- LLM[instruction]

Quy tắc:
- Mỗi lượt chỉ trả về đúng 1 dòng duy nhất.
- Hoặc: Action: Google[...]
- Hoặc: Action: LLM[...]
- Hoặc: Final Answer: ...
- Không thêm giải thích nào khác.
"""

# Tính token usage 
@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated: bool = False

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.estimated = self.estimated or other.estimated


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def ensure_env() -> None:
    load_dotenv()
    missing = []
    if not os.getenv("OPENROUTER_API_KEY"):
        missing.append("OPENROUTER_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")


def build_model_config() -> Dict[str, Any]:
    model_name = os.getenv("BENCHMARK_MODEL") or os.getenv("OPENROUTER_MODEL") or "deepseek/deepseek-chat"
    max_tokens = int(os.getenv("BENCHMARK_MAX_TOKENS", "384"))
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    return {
        "client": client,
        "model_name": model_name,
        "max_tokens": max_tokens,
        "temperature": 0,
    }


def build_search() -> TavilySearchResults:
    return TavilySearchResults(max_results=5)

# Nếu API không trả về usage, ước tính token dựa trên độ dài text (giả sử trung bình 4 ký tự/token)
def rough_token_estimate(text: str) -> int:
    cleaned = text or ""
    if not cleaned:
        return 0
    return max(1, math.ceil(len(cleaned) / 4))

# Cắt chuỗi text dài để tránh evidence quá dài, tiết kiệm token khi nhét search result vào prompt 
def clip_text(text: str, limit: int = 700) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def usage_from_response(response: Any, prompt_text: str = "", content: str = "") -> Usage:
    usage = Usage()

    response_usage = getattr(response, "usage", None)
    if response_usage:
        usage.prompt_tokens = int(getattr(response_usage, "prompt_tokens", 0) or 0)
        usage.completion_tokens = int(getattr(response_usage, "completion_tokens", 0) or 0)
        usage.total_tokens = int(getattr(response_usage, "total_tokens", 0) or 0)

    if usage.total_tokens == 0:
        usage.prompt_tokens = rough_token_estimate(prompt_text)
        usage.completion_tokens = rough_token_estimate(content)
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        usage.estimated = True

    return usage

# Gọi model với prompt, trả về content đã strip think và usage
def invoke_model(model_config: Dict[str, Any], prompt_text: str) -> Tuple[str, Usage]:
    response = model_config["client"].chat.completions.create(
        model=model_config["model_name"],
        temperature=model_config["temperature"],
        max_tokens=model_config["max_tokens"],
        messages=[{"role": "user", "content": prompt_text}],
    )
    content = strip_think(response.choices[0].message.content or "")
    usage = usage_from_response(response, prompt_text, content)
    return content, usage

# Chuyển kết quả tool thành string để nhét vào prompt, ưu tiên format JSON nếu có thể
def stringify_tool_result(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, ensure_ascii=False)
    except TypeError:
        return str(raw)

# Chạy truy vấn Google Search qua Tavily, trích xuất answer và top 3 results để làm evidence
def run_google(search: TavilySearchResults, query: str) -> str:
    raw = search.invoke(query)

    if isinstance(raw, dict):
        parts: List[str] = []
        if raw.get("answer"):
            parts.append(f"Answer: {raw['answer']}")
        for item in raw.get("results", [])[:3]:
            title = item.get("title", "")
            content = clip_text(item.get("content", ""), 220)
            if title or content:
                parts.append(f"- {title}: {content}")
        if parts:
            return clip_text("\n".join(parts), 700)

    if isinstance(raw, list):
        parts = []
        for item in raw[:3]:
            if isinstance(item, dict):
                title = item.get("title", "")
                content = clip_text(item.get("content", ""), 220)
                if title or content:
                    parts.append(f"- {title}: {content}")
            else:
                parts.append(clip_text(str(item), 220))
        if parts:
            return clip_text("\n".join(parts), 700)

    return clip_text(stringify_tool_result(raw), 700)


def run_llm_tool(model_config: Dict[str, Any], instruction: str) -> Tuple[str, Usage]:
    return invoke_model(model_config, instruction)


def run_rewoo(question: str, model_config: Dict[str, Any], search: TavilySearchResults) -> Dict[str, Any]:
    started = time.perf_counter()
    usage = Usage()
    llm_calls = 0
    tool_calls = 0

    planner_prompt = REWOO_PLANNER_PROMPT.format(task=question)
    plan_text, planner_usage = invoke_model(model_config, planner_prompt)
    usage.add(planner_usage)
    llm_calls += 1

    steps = re.findall(PLAN_REGEX, plan_text, re.DOTALL)
    if not steps:
        raise RuntimeError("ReWOO planner did not return parseable Plan/#E steps.")

    results: Dict[str, str] = {}
    for _, step_name, tool_name, tool_input in steps:
        resolved_input = tool_input
        for key, value in results.items():
            resolved_input = resolved_input.replace(key, value)

        tool_calls += 1
        if tool_name == "Google":
            results[step_name] = run_google(search, resolved_input)
        elif tool_name == "LLM":
            tool_result, tool_usage = run_llm_tool(model_config, resolved_input)
            results[step_name] = tool_result
            usage.add(tool_usage)
            llm_calls += 1
        else:
            raise RuntimeError(f"ReWOO used unsupported tool: {tool_name}")

    plan_block_lines: List[str] = []
    for plan_reason, step_name, tool_name, tool_input in steps:
        resolved_input = tool_input
        for key, value in results.items():
            resolved_input = resolved_input.replace(key, value)
        plan_block_lines.append(f"Plan: {plan_reason}")
        plan_block_lines.append(f"{step_name} = {tool_name}[{resolved_input}]")
        plan_block_lines.append(f"Evidence: {clip_text(results.get(step_name, ''), 450)}")
        plan_block_lines.append("")

    solver_prompt = REWOO_SOLVER_PROMPT.format(plan="\n".join(plan_block_lines), task=question)
    final_answer, solver_usage = invoke_model(model_config, solver_prompt)
    usage.add(solver_usage)
    llm_calls += 1

    elapsed = time.perf_counter() - started
    return {
        "answer": final_answer,
        "time_sec": elapsed,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "estimated_tokens": usage.estimated,
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "plan": plan_text,
    }


def parse_react_reply(reply: str) -> Tuple[str | None, Tuple[str, str] | None]:
    cleaned = strip_think(reply)
    final_match = re.search(r"Final Answer\s*:\s*(.+)", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if final_match:
        return final_match.group(1).strip(), None

    action_match = re.search(r"Action\s*:\s*(Google|LLM)\[(.*?)\]", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if action_match:
        tool = action_match.group(1)
        tool_input = action_match.group(2).strip()
        return None, (tool, tool_input)

    return cleaned, None


def run_react(question: str, model_config: Dict[str, Any], search: TavilySearchResults, max_steps: int = 6) -> Dict[str, Any]:
    started = time.perf_counter()
    usage = Usage()
    llm_calls = 0
    tool_calls = 0
    scratchpad = "" # Dùng để lưu lại toàn bộ lịch sử Action/Observation, giúp model có đủ ngữ cảnh để đưa ra quyết định ở mỗi bước
    final_answer = ""

    for step_index in range(1, max_steps + 1):
        react_prompt = (
            f"{REACT_SYSTEM_PROMPT}\n\n"
            f"Question: {question}\n\n"
            f"Current scratchpad:\n{scratchpad}\n\n"
            "Trả về đúng 1 dòng tiếp theo."
        )
        reply, reply_usage = invoke_model(model_config, react_prompt)
        usage.add(reply_usage)
        llm_calls += 1

        parsed_final, parsed_action = parse_react_reply(reply)
        if parsed_action is None and parsed_final:
            final_answer = parsed_final
            scratchpad += f"Step {step_index} Model Output:\n{reply}\n\n"
            break

        if parsed_action is None:
            final_answer = parsed_final or reply
            scratchpad += f"Step {step_index} Model Output:\n{reply}\n\n"
            break

        tool_name, tool_input = parsed_action
        tool_calls += 1
        if tool_name == "Google":
            observation = run_google(search, tool_input)
        else:
            observation, tool_usage = run_llm_tool(model_config, tool_input)
            usage.add(tool_usage)
            llm_calls += 1

        scratchpad += (
            f"Step {step_index}:\n"
            f"{reply}\n"
            f"Observation: {clip_text(observation, 450)}\n\n"
        )
    else: # Nếu đã chạy hết max_steps mà vẫn chưa có Final Answer, gọi model thêm 1 lần nữa với prompt nhắc phải trả lời Final Answer dựa trên toàn bộ scratchpad
        final_prompt = (
            f"{REACT_SYSTEM_PROMPT}\n\nQuestion: {question}\n\n"
            f"Current scratchpad:\n{scratchpad}\n\n"
            "Dựa trên toàn bộ scratchpad, hãy trả về đúng một dòng Final Answer: <đáp án>."
        )
        final_answer, final_usage = invoke_model(model_config, final_prompt)
        usage.add(final_usage)
        llm_calls += 1
        parsed_final, _ = parse_react_reply(final_answer)
        final_answer = parsed_final or final_answer

    elapsed = time.perf_counter() - started
    return {
        "answer": final_answer,
        "time_sec": elapsed,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "estimated_tokens": usage.estimated,
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "scratchpad": scratchpad,
    }


def build_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    ordered_columns = [
        "question_id",
        "system",
        "time_sec",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "llm_calls",
        "tool_calls",
        "estimated_tokens",
        "answer",
        "status",
        "error",
        "question",
    ]
    return df[ordered_columns]


def save_summary(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    grouped = df.groupby("system", dropna=False).agg(
        total_time_sec=("time_sec", "sum"),
        avg_time_sec=("time_sec", "mean"),
        total_tokens=("total_tokens", "sum"),
        avg_tokens=("total_tokens", "mean"),
        total_llm_calls=("llm_calls", "sum"),
        success_count=("status", lambda s: int((s == "ok").sum())),
    )
    summary["by_system"] = grouped.round(3).to_dict(orient="index")

    if {"ReAct", "ReWOO"}.issubset(grouped.index):
        react = grouped.loc["ReAct"]
        rewoo = grouped.loc["ReWOO"]
        token_saved = float(react["total_tokens"] - rewoo["total_tokens"])
        time_saved = float(react["total_time_sec"] - rewoo["total_time_sec"])
        summary["savings"] = {
            "token_saved": round(token_saved, 3),
            "token_saved_pct": round((token_saved / react["total_tokens"] * 100) if react["total_tokens"] else 0, 2),
            "time_saved_sec": round(time_saved, 3),
            "time_saved_pct": round((time_saved / react["total_time_sec"] * 100) if react["total_time_sec"] else 0, 2),
        }

    summary_path = OUTPUT_DIR / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def save_chart(df: pd.DataFrame, summary: Dict[str, Any]) -> None:
    ok_df = df[df["status"] == "ok"].copy()
    if ok_df.empty:
        return

    pivot_tokens = ok_df.pivot(index="question_id", columns="system", values="total_tokens").fillna(0)
    pivot_time = ok_df.pivot(index="question_id", columns="system", values="time_sec").fillna(0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 16))

    pivot_tokens.plot(kind="bar", ax=axes[0], color=["#f97316", "#6366f1"])
    axes[0].set_title("Tong token moi cau hoi")
    axes[0].set_xlabel("Question ID")
    axes[0].set_ylabel("Tokens")
    axes[0].grid(axis="y", alpha=0.25)

    pivot_time.plot(kind="bar", ax=axes[1], color=["#f97316", "#6366f1"])
    axes[1].set_title("Thoi gian phan hoi moi cau hoi")
    axes[1].set_xlabel("Question ID")
    axes[1].set_ylabel("Seconds")
    axes[1].grid(axis="y", alpha=0.25)

    if "savings" in summary:
        savings = summary["savings"]
        axes[2].bar(
            ["Token saved", "Time saved"],
            [savings["token_saved_pct"], savings["time_saved_pct"]],
            color=["#10b981", "#06b6d4"],
        )
        axes[2].set_ylim(0, max(5, savings["token_saved_pct"], savings["time_saved_pct"]) * 1.25)
        axes[2].set_ylabel("Percent")
        axes[2].set_title("Ty le tiet kiem cua ReWOO so voi ReAct")
        for index, value in enumerate([savings["token_saved_pct"], savings["time_saved_pct"]]):
            axes[2].text(index, value + 0.5, f"{value:.2f}%", ha="center", va="bottom")
    else:
        axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "benchmark_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ensure_env()
    OUTPUT_DIR.mkdir(exist_ok=True)

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    question_limit = int(os.getenv("BENCHMARK_LIMIT", "0"))
    if question_limit > 0:
        questions = questions[:question_limit]

    model_config = build_model_config()
    search = build_search()

    all_rows: List[Dict[str, Any]] = []

    for item in questions:
        question_id = item["id"]
        question = item["question"]
        print(f"Running {question_id}: {question}")

        for system_name, runner in (("ReAct", run_react), ("ReWOO", run_rewoo)):
            try:
                result = runner(question, model_config, search)
                row = {
                    "question_id": question_id,
                    "system": system_name,
                    "time_sec": round(result["time_sec"], 3),
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["total_tokens"],
                    "llm_calls": result["llm_calls"],
                    "tool_calls": result["tool_calls"],
                    "estimated_tokens": result["estimated_tokens"],
                    "answer": result["answer"],
                    "status": "ok",
                    "error": "",
                    "question": question,
                }
            except Exception as exc:
                row = {
                    "question_id": question_id,
                    "system": system_name,
                    "time_sec": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "llm_calls": 0,
                    "tool_calls": 0,
                    "estimated_tokens": False,
                    "answer": "",
                    "status": "error",
                    "error": str(exc),
                    "question": question,
                }
            all_rows.append(row)
            print(
                f"  {system_name}: status={row['status']} time={row['time_sec']}s "
                f"tokens={row['total_tokens']}"
            )

    df = build_dataframe(all_rows)
    csv_path = OUTPUT_DIR / "benchmark_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = save_summary(df)
    save_chart(df, summary)

    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved results to: {csv_path}")
    print(f"Saved summary to: {OUTPUT_DIR / 'benchmark_summary.json'}")
    print(f"Saved chart to: {OUTPUT_DIR / 'benchmark_comparison.png'}")


if __name__ == "__main__":
    main()