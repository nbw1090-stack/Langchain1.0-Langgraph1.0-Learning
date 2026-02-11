"""
LangChain 1.0 - Agent 执行循环（ReAct 模式）
============================================

本模块重点讲解：
1. ReAct 执行循环的详细过程（Reason → Act → Observe）
2. 流式输出（streaming）
3. 查看中间步骤
4. 理解消息流转

⚠️ API 说明：
- 本模块使用 `create_react_agent`（来自 langgraph.prebuilt）
- 这是 LangGraph 1.0 的预构建图，明确使用 ReAct 架构
- 适合学习 Agent 执行循环的底层原理

🔄 ReAct 循环：
   Reason（推理）→ Act（行动）→ Observe（观察）→ 循环直到完成
"""

import os
import sys

# 添加工具目录到路径
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(parent_dir, '04_custom_tools', 'tools'))

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent  # ✅ LangGraph 预构建 ReAct Agent
from calculator import calculator
from weather import get_weather

# 加载环境变量
load_dotenv()
GROQ_API_KEY = os.getenv("deepseek_api")
model = init_chat_model(
    "deepseek-r1",
    model_provider="openai",
    api_key=GROQ_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"            
)

# ============================================================================
# 示例 1：理解执行循环 - 查看完整消息历史
# ============================================================================
def example_1_understand_loop():
    """
    示例1：查看 Agent 执行循环的每一步

    关键：response['messages'] 包含完整的对话历史
    """
    print("\n" + "="*70)
    print("示例 1：Agent 执行循环详解")
    print("="*70)

    agent = create_agent(
        model=model,
        tools=[calculator],
    system_prompt="你是一个有帮助的助手。"
    )

    print("\n问题：25 乘以 8 等于多少？")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "25 乘以 8 等于多少？"}]
    })

    print("\n完整消息历史：")
    for i, msg in enumerate(response['messages'], 1):
        print(f"\n{'='*60}")
        print(f"消息 {i}: {msg.__class__.__name__}")
        print(f"{'='*60}")

        if hasattr(msg, 'content') and msg.content:
            print(f"内容: {msg.content}")

        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"工具调用:")
            for tc in msg.tool_calls:
                print(f"  - 工具: {tc['name']}")
                print(f"  - 参数: {tc['args']}")

        if hasattr(msg, 'name') and msg.name:
            print(f"工具名: {msg.name}")

    print("\n\n执行流程：")
    print("""
    1. HumanMessage    → 用户问题
    2. AIMessage       → AI 决定调用工具（包含 tool_calls）
    3. ToolMessage     → 工具执行结果
    4. AIMessage       → AI 基于结果生成最终答案
    """)

    print("\n关键点：")
    print("  - Agent 自动完成这个循环")
    print("  - 所有步骤都记录在 messages 中")
    print("  - 最后一条消息是最终答案")

# ============================================================================
# 示例 2：流式输出（Streaming）
# ============================================================================
def example_2_streaming():
    """
    示例2：实时查看 Agent 的输出

    使用 .stream() 方法
    """
    print("\n" + "="*70)
    print("示例 2：流式输出")
    print("="*70)

    agent = create_agent(
        model=model,
        tools=[calculator, get_weather],
    system_prompt="你是一个有帮助的助手。"
    )

    print("\n问题：北京天气如何？然后计算 10 加 20")
    print("\n流式输出（实时显示）：")
    print("-" * 70)

    # 使用 stream 方法
    for chunk in agent.stream({
        "messages": [{"role": "user", "content": "北京天气如何？"}]
    }):
        # chunk 是字典，包含更新的状态
        if 'model' in chunk and 'messages' in chunk["model"]:
            # 获取最新的消息
            latest_msg = chunk["model"]['messages'][-1]

            # 如果是 AI 的最终回答
            if hasattr(latest_msg, 'content') and latest_msg.content:
                if not hasattr(latest_msg, 'tool_calls') or not latest_msg.tool_calls:
                    print(f"\n最终回答: {latest_msg.content}")

    print("\n关键点：")
    print("  - stream() 返回生成器，逐步返回结果")
    print("  - 用于实时显示进度")
    print("  - 适合长时间运行的任务")

# ============================================================================
# 示例 3：多步骤执行
# ============================================================================
def example_3_multi_step():
    """
    示例3：Agent 执行多个工具调用

    理解复杂任务的执行过程
    """
    print("\n" + "="*70)
    print("示例 3：多步骤执行")
    print("="*70)

    agent = create_agent(
        model=model,
        tools=[calculator],
        system_prompt="你是一个数学助手。当遇到复杂计算时，分步骤计算。"
    )

    print("\n问题：先算 10 加 20，然后把结果乘以 3")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "先算 10 加 20，然后把结果乘以 3"}]
    })

    # 统计工具调用次数
    tool_calls_count = 0
    print(response['messages'])
    for msg in response['messages']:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls_count += len(msg.tool_calls)

    print(f"\n工具调用次数: {tool_calls_count}")
    print(f"最终答案: {response['messages'][-1].content}")

    print("\n关键点：")
    print("  - Agent 可以多次调用工具")
    print("  - 每次调用的结果会影响下一步")
    print("  - 直到得到最终答案")

# ============================================================================
# 示例 4：查看中间状态
# ============================================================================
def example_4_inspect_state():
    """
    示例4：在执行过程中查看状态

    使用 stream 并检查每个 chunk
    """
    print("\n" + "="*70)
    print("示例 4：查看中间状态")
    print("="*70)

    agent = create_agent(
        model=model,
        tools=[calculator],
    system_prompt="你是一个有帮助的助手。"
    )

    print("\n问题：100 除以 5 等于多少？")
    print("\n执行步骤：")

    step = 0
    for chunk in agent.stream({
        "messages": [{"role": "user", "content": "100 除以 5 等于多少？"}]
    }):
        step += 1
        print(f"\n步骤 {step}:")

        if 'model' in chunk and 'messages' in chunk['model']:
            latest = chunk['model']['messages'][-1]
            msg_type = latest.__class__.__name__
            print(f"  类型: {msg_type}")

            if hasattr(latest, 'tool_calls') and latest.tool_calls:
                print(f"  工具调用: {latest.tool_calls[0]['name']}")
            elif hasattr(latest, 'content') and latest.content:
                content_preview = latest.content[:50] if len(latest.content) > 50 else latest.content
                print(f"  内容: {content_preview}...")

    print("\n关键点：")
    print("  - stream 让你看到每个步骤")
    print("  - 可以用于调试")
    print("  - 可以用于进度显示")

# ============================================================================
# 示例 5：理解消息类型
# ============================================================================
def example_5_message_types():
    """
    示例5：详解各种消息类型

    Agent 执行循环中的消息类型
    """
    print("\n" + "="*70)
    print("示例 5：消息类型详解")
    print("="*70)

    agent = create_agent(
        model=model,
        tools=[get_weather],
    system_prompt="你是一个有帮助的助手。"
    )

    response = agent.invoke({
        "messages": [{"role": "user", "content": "上海天气如何？"}]
    })

    print("\n消息类型分析：")
    for msg in response['messages']:
        msg_type = msg.__class__.__name__

        if msg_type == "HumanMessage":
            print(f"\n[HumanMessage] 用户输入")
            print(f"  内容: {msg.content}")

        elif msg_type == "AIMessage":
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"\n[AIMessage] AI 决定调用工具")
                print(f"  工具: {msg.tool_calls[0]['name']}")
                print(f"  参数: {msg.tool_calls[0]['args']}")
            else:
                print(f"\n[AIMessage] AI 的最终回答")
                print(f"  内容: {msg.content}")

        elif msg_type == "ToolMessage":
            print(f"\n[ToolMessage] 工具执行结果")
            print(f"  工具: {msg.name}")
            print(f"  结果: {msg.content}")

    print("\n\n消息类型总结：")
    print("""
    HumanMessage  → 用户的输入
    AIMessage     → AI 的输出（可能包含 tool_calls 或最终答案）
    ToolMessage   → 工具的执行结果
    SystemMessage → 系统指令（通过 prompt 参数设置）
    """)

# ============================================================================
# 示例 6：执行循环最佳实践
# ============================================================================
def example_6_best_practices():
    """
    示例6：使用执行循环的最佳实践
    """
    print("\n" + "="*70)
    print("示例 6：执行循环最佳实践")
    print("="*70)

    print("""
最佳实践：

1. 获取最终答案
   final_answer = response['messages'][-1].content

2. 查看是否使用了工具
   used_tools = [
       msg.tool_calls[0]['name']
       for msg in response['messages']
       if hasattr(msg, 'tool_calls') and msg.tool_calls
   ]

3. 流式输出用于用户体验
   for chunk in agent.stream(input):
       # 实时显示给用户

4. 调试时查看完整历史
   for msg in response['messages']:
       print(msg)

5. 错误处理
   try:
       response = agent.invoke(input)
   except Exception as e:
       print(f"Agent 执行错误: {e}")

6. API 注意事项（LangChain 1.0）
   - 使用 create_agent（LangChain 1.0 新 API）
   - LangChain 1.0 新 API
    """)

    # 实际示例
    print("\n实际示例：")
    system_prompt="你是一个有帮助的助手。"
    agent = create_agent(model=model, tools=[calculator])

    try:
        response = agent.invoke({
            "messages": [{"role": "user", "content": "5 加 3"}]
        })

        # 获取最终答案
        final_answer = response['messages'][-1].content
        print(f"最终答案: {final_answer}")

        # 查看使用的工具
        used_tools = [
            msg.tool_calls[0]['name']
            for msg in response['messages']
            if hasattr(msg, 'tool_calls') and msg.tool_calls
        ]
        print(f"使用的工具: {used_tools}")

    except Exception as e:
        print(f"错误: {e}")

# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "="*70)
    print(" LangChain 1.0 - Agent 执行循环")
    print("="*70)

    try:
        example_1_understand_loop()
        input("\n按 Enter 继续...")

        example_2_streaming()
        input("\n按 Enter 继续...")

        example_3_multi_step()
        input("\n按 Enter 继续...")

        example_4_inspect_state()
        input("\n按 Enter 继续...")

        example_5_message_types()
        input("\n按 Enter 继续...")

        example_6_best_practices()

        print("\n" + "="*70)
        print(" 完成！")
        print("="*70)
        print("\n核心要点：")
        print("  ✅ Agent 执行循环：问题 → 工具调用 → 结果 → 答案")
        print("  ✅ messages 记录完整历史")
        print("  ✅ stream() 用于实时输出")
        print("  ✅ 理解 HumanMessage、AIMessage、ToolMessage")
        print("  ✅ 使用 create_agent（LangChain 1.0 API）")
        print("\n阶段一（基础）完成！")
        print("  已学习：模型调用、提示词、消息、工具、Agent")
        print("\n下一阶段：")
        print("  phase2_practical - 内存、中间件、结构化输出")

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
