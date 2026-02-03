### 使用 colab 運行此程式碼 ###
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel  
from langchain_openai import ChatOpenAI
import json
import time
 
# 1.設定 LLM
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="vllm-token",
    model="Qwen/Qwen3-VL-8B-Instruct",
    temperature=0,
    max_tokens=50,
)

# 2.設定 Prompt Template 模板 (parser 格式注入 Prompt)
One_Prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位繁體中文的醫生。請用「專業、條理清晰」的風格輸出。"),
    ("human", "{topic}")
])
Two_Prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位繁體中文的溫柔姊姊。請用「輕鬆口吻，引導互動，溫柔」的風格輸出。"),
    ("human", "{topic}")
])

parser = StrOutputParser()

# 3.建立 Chain (Prompt + LLM + Parser)
One_Chain= One_Prompt | llm | parser
Two_Chain= Two_Prompt | llm | parser

multi_post_agent = RunnableParallel(
    professional=One_Chain,
    casual=Two_Chain,
)


async def run_streaming(topic: str):
    print("\n=== Streaming Output ===")
    async for chunk in multi_post_agent.astream({"topic": topic}):
        print(json.dumps(chunk, ensure_ascii=False), flush=True)

def run_batch(topic: str):
    print("\n=== Batch Output ===")
    start = time.perf_counter()
    result = multi_post_agent.invoke({"topic": topic})
    elapsed = time.perf_counter() - start

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nBatch elapsed: {elapsed:.3f} seconds")

    with open("batch_time.log", "a", encoding="utf-8") as f:
        f.write(f"topic={topic}\telapsed={elapsed:.3f}s\n")

topic = input("請輸入主題：").strip()
await run_streaming(topic)
run_batch(topic)
