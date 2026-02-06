"""
LangChain 1.0 - Middleware Basics (中间件基础)
使用中间件修改消息未成功
==============================================

本模块重点讲解：
1. 什么是中间件（Middleware）
2. before_model 和 after_model 钩子
3. 自定义中间件的创建
4. 多个中间件的组合
5. 内置中间件的使用
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.agents.middleware import AgentMiddleware
from langgraph.checkpoint.memory import InMemorySaver

# 加载环境变量
load_dotenv()
GROQ_API_KEY = os.getenv("deepseek_api")
model = init_chat_model(
    "deepseek-r1",
    model_provider="openai",
    api_key=GROQ_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    streaming=True,
)

# ============================================================================
# 示例 1：最简单的中间件
"""
  关键规则

  1. 必须继承 AgentMiddleware ← 这个固定
  2. 方法名固定 (before_model, after_model) ← 这个固定
  3. 类名随意 ← 这个不固定

  LangGraph 只看：
  - 是否继承 AgentMiddleware？
  - 是否有 before_model / after_model 方法？
  """
# ============================================================================
class LoggingMiddleware(AgentMiddleware): # ✅ 类名随意
    """
    日志中间件 - 记录每次模型调用

    before_model: 模型调用前执行
    after_model: 模型响应后执行
    """

    def before_model(self, state, runtime):
        """模型调用前"""
        print("\n[中间件] before_model: 准备调用模型")
        print(f"[中间件] 当前消息数: {len(state.get('messages', []))}")
        return None  # 返回 None 表示继续正常流程

    def after_model(self, state, runtime):
        """模型响应后"""
        print("[中间件] after_model: 模型已响应")
        last_message = state.get('messages', [])[-1]
        print(f"[中间件] 响应类型: {last_message.__class__.__name__}")
        return None  # 返回 None 表示不修改状态

def example_1_basic_middleware():
    """
    示例1：基础中间件 - 日志记录

    展示 before_model 和 after_model 的基本用法
    """
    print("\n" + "="*70)
    print("示例 1：基础中间件 - 日志记录")
    print("="*70)

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个有帮助的助手。",
        middleware=[LoggingMiddleware()]  # 添加中间件
    )

    print("\n用户: 你好")
    response = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
    print(f"Agent: {response['messages'][-1].content}")

    print("\n关键点：")
    print("  - before_model 在模型调用前执行")
    print("  - after_model 在模型响应后执行")
    print("  - 返回 None 表示不修改状态")

# ============================================================================
# 示例 2：修改状态的中间件
# 中间件计数与config配置的线程无关，在一个进程中的agent调用这个中间件对象都会进行++
# ============================================================================
class CallCounterMiddleware(AgentMiddleware):
    """
    计数中间件 - 统计模型调用次数

    在中间件内部维护计数器（简单版本）
    """

    def __init__(self):
        super().__init__()
        self.count = 0  # 简单计数器

    def after_model(self, state, runtime):
        """模型响应后，增加计数"""
        self.count += 1
        print(f"\n[计数器] 模型调用次数: {self.count}")
        return None  # 不修改 state

def example_2_state_modification():
    """
    示例2：中间件内部状态 - 计数器

    展示如何在中间件内部维护状态（不依赖 Agent state）
    """
    print("\n" + "="*70)
    print("示例 2：中间件内部状态 - 模型调用计数")
    print("="*70)

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个有帮助的助手。",
        middleware=[CallCounterMiddleware()],
        checkpointer=InMemorySaver()
    )
    agent1 = create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个有帮助的助手。",
        middleware=[CallCounterMiddleware()],
        checkpointer=InMemorySaver()
    )

    config = {"configurable": {"thread_id": "counter_test"}}
    config1 = {"configurable": {"thread_id": "counter_test1"}}

    print("\n第一次调用:")
    agent.invoke({"messages": [{"role": "user", "content": "你好"}]}, config)

    print("\n第二次调用:")
    agent1.invoke({"messages": [{"role": "user", "content": "今天天气"}]}, config1)

    print("\n第三次调用:")
    agent.invoke({"messages": [{"role": "user", "content": "谢谢"}]}, config)

    print("\n关键点：")
    print("  - 中间件内部维护计数器（self.count）")
    print("  - 不依赖 Agent state（更可靠）")
    print("  - 返回 None 表示不修改 Agent 状态")

# ============================================================================
# 示例 3：修改状态信息中间件
# 修改messages的信息是对原始messages的拼接，并不是直接替换
# ============================================================================
class MessageTrimmerMiddleware(AgentMiddleware):
    """
    消息修剪中间件 - 限制发送给模型的消息数量

    重要说明：
    - 中间件只能控制传递给模型的消息，不能改变最终返回的完整对话历史
    - agent.invoke() 返回的 response['messages'] 仍然包含原始所有消息

    """

    def before_model(self, state, runtime):
        """模型调用前，修改消息"""
        messages = state.get('messages', [])
        messages=[{"role": "user", "content":"你好，我叫张三"}]
        return{"messages": messages}

    def after_model(self, state, runtime):
        """模型调用后，修改消息"""
        messages = state.get('messages', [])
        print("返回消息长度为：", len(messages))
        print("返回消息内容为：", messages)
        messages=[{"role": "user", "content":"你好，我叫李四"}]
        return{"messages": messages}

def example_3_message_trimming():
    """
    示例3：消息修剪 - 理解中间件的行为

    展示中间件修剪的实际效果：
    1. 中间件修剪的是发送给模型的消息
    2. agent.invoke() 返回的仍然包含所有原始消息
    3. 演示如何在调用 agent 前修剪消息
    """
    print("\n" + "="*70)
    print("示例 3：消息修剪 - 理解中间件行为")
    print("="*70)

    print("\n[说明1] 中间件修剪的是发送给模型的消息，不是返回的消息\n")

    # 场景1：使用中间件修剪（发送给模型的消息会变少）
    print("--- 场景1：中间件修剪 ---")
    middleware = MessageTrimmerMiddleware()
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个有帮助的助手。",
        middleware=[middleware]
    )

    # 创建6条消息
    name = ["aa", "bb", "cc", "dd"]
    messages = [{"role": "user", "content": f"我有多个名字，{i}是我其中一个名字"} for i in name]
    messages.append({"role": "user", "content": "请问我有几个名字，分别是什么"})
    print(f"原始消息数: {len(messages)}")
    # 调用 agent
    print(messages)
    print("-------")
    response = agent.invoke({"messages": messages})
    print("【完整 response】", response["messages"])

    print(f"\n【说明】中间件行为分析：")
    print(f"  1. before_model 不会减少发送给模型的消息，只会添加before_model中新增的消息")

# ============================================================================
# 示例 4：输出验证中间件
# ============================================================================
class OutputValidationMiddleware(AgentMiddleware):
    """
    输出验证中间件 - 检查响应长度

    after_model 验证输出
    """

    def __init__(self, max_length=2):
        super().__init__()
        self.max_length = max_length

    def after_model(self, state, runtime):
        """模型响应后，验证输出"""
        messages = state.get('messages', [])
        if not messages:
            return None

        last_message = messages[-1]
        content = getattr(last_message, 'content', '')

        if len(content) > self.max_length:
            print(f"\n[警告] 响应过长 ({len(content)} 字符)，已截断到 {self.max_length}")
            messages[-1].content = messages[-1].content[:self.max_length]
            return {"messages": messages}
        return None

def example_4_output_validation():
    """
    示例4：输出验证 - 检查响应质量

    展示如何验证模型输出
    """
    print("\n" + "="*70)
    print("示例 4：输出验证 - 响应长度检查")
    print("="*70)

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个有帮助的助手。",
        middleware=[OutputValidationMiddleware(max_length=5)]
    )

    print("\n用户: 请详细介绍 Python 编程语言的历史、特点和应用")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "请详细介绍 Python 编程语言的历史、特点和应用"}]
    })
    print(f"Agent: {response['messages'][-1].content}...")

    print("\n关键点：")
    print("  - after_model 可以验证输出")
    print("  - 可以实现重试、截断等逻辑")
    print("  - 保证输出质量")

# ============================================================================
# 示例 5：多个中间件组合
# ============================================================================
class TimingMiddleware(AgentMiddleware):
    """计时中间件"""

    def before_model(self, state, runtime):
        import time
        # 记录开始时间（实际应该用 runtime 的上下文管理）
        print("\n[计时] 开始调用模型...")
        return None

    def after_model(self, state, runtime):
        print("[计时] 模型调用完成")
        return None

def example_5_multiple_middleware():
    """
    示例5：多个中间件 - 执行顺序

    展示中间件的执行顺序：
    - before_model: 正序（1→2→3）
    - after_model: 逆序（3→2→1）
    """
    print("\n" + "="*70)
    print("示例 5：多个中间件 - 执行顺序")
    print("="*70)

    class Middleware1(AgentMiddleware):
        def before_model(self, state, runtime):
            print("[中间件1] before_model")
            return None
        def after_model(self, state, runtime):
            print("[中间件1] after_model")
            return None

    class Middleware2(AgentMiddleware):
        def before_model(self, state, runtime):
            print("[中间件2] before_model")
            return None
        def after_model(self, state, runtime):
            print("[中间件2] after_model")
            return None

    class Middleware3(AgentMiddleware):
        def before_model(self, state, runtime):
            print("[中间件3] before_model")
            return None
        def after_model(self, state, runtime):
            print("[中间件3] after_model")
            return None

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个有帮助的助手。",
        middleware=[Middleware1(), Middleware2(), Middleware3()]
    )

    print("\n执行一次调用，观察顺序：")
    agent.invoke({"messages": [{"role": "user", "content": "测试"}]})

    print("\n关键点：")
    print("  - before_model: 正序执行（1→2→3）")
    print("  - after_model: 逆序执行（3→2→1）")
    print("  - 类似洋葱模型：1→2→3→模型→3→2→1")

# ============================================================================
# 示例 6：条件跳转（高级）
# ============================================================================
class MaxCallsMiddleware(AgentMiddleware):
    """
    最大调用限制中间件

    通过抛出异常来阻止模型调用（更可靠的方式）
    """

    def __init__(self, max_calls=3):
        super().__init__()
        self.max_calls = max_calls
        self.count = 0  # 简单计数器

    def before_model(self, state, runtime):
        """检查调用次数，超过限制则抛出异常"""
        if self.count >= self.max_calls:
            print(f"\n[限制] 已达到最大调用次数 {self.max_calls}，停止调用")
            # 抛出自定义异常来阻止继续执行
            raise ValueError(f"已达到最大调用次数限制: {self.max_calls}")

        print(f"[限制] 当前调用次数: {self.count}/{self.max_calls}")
        return None

    def after_model(self, state, runtime):
        """增加计数"""
        self.count += 1
        print("次数+1")
        return None

def example_6_conditional_jump():
    """
    示例6：调用限制 - 通过异常阻止调用

    展示如何使用异常来阻止模型调用（比 jump_to 更可靠）
    """
    print("\n" + "="*70)
    print("示例 6：调用限制 - 最大调用次数")
    print("="*70)

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个有帮助的助手。",
        middleware=[MaxCallsMiddleware(max_calls=2)],
        checkpointer=InMemorySaver()
    )

    config = {"configurable": {"thread_id": "limit_test"}}

    for i in range(4):
        print(f"\n--- 第 {i+1} 次尝试调用 ---")
        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": f"消息{i+1}"}]},
                config
            )
            if response.get('messages'):
                print(f"响应: {response['messages'][-1].content[:50]}...")
        except ValueError as e:
            print(f"[已阻止] {e}")
        except Exception as e:
            print(f"调用失败: {e}")

    print("\n关键点：")
    print("  - before_model 中抛出异常可以阻止模型调用")
    print("  - 比 jump_to 更可靠（在 LangChain 1.0 中）")
    print("  - 中间件内部维护计数（self.count）")
    print("  - 用于实现熔断、限流等逻辑")
    print("\n注意：")
    print("  - jump_to 在 middleware 中可能不按预期工作")
    print("  - 推荐用异常来实现流程控制")

# ============================================================================
# 示例 7：内置中间件使用
# ============================================================================
def example_7_builtin_middleware():
    """
    示例7：内置中间件 - SummarizationMiddleware

    使用 LangChain 提供的内置中间件
    """
    print("\n" + "="*70)
    print("示例 7：内置中间件 - 自动摘要")
    print("="*70)

    from langchain.agents.middleware import SummarizationMiddleware

    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="你是一个有帮助的助手。",
        middleware=[
            SummarizationMiddleware(
                model="groq:llama-3.3-70b-versatile",
                trigger=('tokens', 200)
            )
        ],
        checkpointer=InMemorySaver()
    )

    config = {"configurable": {"thread_id": "summary_test"}}

    # 连续多轮对话
    conversations = [
        "介绍一下 Python",
        "它有哪些特点？",
        "主要应用在哪些领域？",
        "和 Java 的区别是什么？"
    ]

    for i, msg in enumerate(conversations):
        print(f"\n--- 第 {i+1} 轮对话 ---")
        print(f"用户: {msg}")
        response = agent.invoke({"messages": [{"role": "user", "content": msg}]}, config)
        print(f"Agent: {response['messages'][-1].content[:100]}...")
        print(f"总消息数: {len(response['messages'])}")

    print("\n关键点：")
    print("  - SummarizationMiddleware 自动摘要旧消息")
    print("  - 防止消息历史无限增长")
    print("  - 第 08 章详细学习过")

# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "="*70)
    print(" LangChain 1.0 - Middleware Basics (中间件)")
    print("="*70)

    try:
        example_1_basic_middleware()
        input("\n按 Enter 继续...")

        example_2_state_modification()
        input("\n按 Enter 继续...")

        example_3_message_trimming()
        input("\n按 Enter 继续...")

        example_4_output_validation()
        input("\n按 Enter 继续...")

        example_5_multiple_middleware()
        input("\n按 Enter 继续...")

        example_6_conditional_jump()
        input("\n按 Enter 继续...")

        example_7_builtin_middleware()

        print("\n" + "="*70)
        print(" 完成！")
        print("="*70)
        print("\n核心要点：")
        print("  1. 继承 AgentMiddleware 创建自定义中间件")
        print("  2. before_model - 模型调用前执行")
        print("  3. after_model - 模型响应后执行")
        print("  4. 返回 None - 不修改状态")
        print("  5. 返回 dict - 更新状态")
        print("  6. 抛出异常 - 阻止执行（流程控制）")
        print("  7. 执行顺序：before 正序，after 逆序")
        print("  8. 推荐：在中间件内部维护状态（self.xxx）更可靠")
        print("\n下一步：")
        print("  11_structured_output - 结构化输出")

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
