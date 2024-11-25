from autogen import AssistantAgent
from autogen.coding.local_commandline_code_executor import LocalCommandLineCodeExecutor

from config import LLM_CONFIG


def create_coding_agent() -> AssistantAgent:
    # define the agent
    agent = AssistantAgent(
        name="Coding Agent",
        system_message="You are a helpful AI assistant. "
                       "You can help with writing Python code, "
                       "Verify the correctness of the code, "
                       "Display the output in a code block, "
                       "Return 'TERMINATE' when the task is done.",
        llm_config=LLM_CONFIG,
    )

    # add the tools to the agent
    # agent.register_for_llm(name="calculator", description="A simple calculator")(calculator)

    return agent


def create_user_proxy():
    user_proxy = AssistantAgent(
        name="User",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
        human_input_mode="NEVER",
        code_execution_config={
            "executor": LocalCommandLineCodeExecutor(work_dir="coding"),
        }
    )
    # user_proxy.register_for_execution(name="calculator")(calculator)
    return user_proxy


def main():
    user_proxy = create_user_proxy()
    calculator_agent = create_coding_agent()
    chat_result = user_proxy.initiate_chat(calculator_agent, cache=None,
                                           message="Write a Python function that takes a list of numbers and returns the average of the numbers.")
    print(chat_result)


if __name__ == "__main__":
    main()
