from autogen import AssistantAgent
from autogen.coding.local_commandline_code_executor import LocalCommandLineCodeExecutor
from autogen import UserProxyAgent
from config import LLM_CONFIG


def execute_code_block(message_content: str) -> str:
    """
    Extract and execute Python code from the given message content.
    :param message_content: The content of the message to process.
    :return: Execution result or None if no code block is found.
    """
    executor = LocalCommandLineCodeExecutor(work_dir="coding")
    code_block = executor.code_extractor.extract_code_blocks(message_content)
    if code_block:
        try:
            result = executor.execute_code_blocks(code_block)
            return f"Code executed successfully. Output:\n{result}"
        except Exception as e:
            return f"Error executing code: {str(e)}"
    return None


def create_coding_agent() -> AssistantAgent:
    """Create the assistant agent with appropriate configurations."""
    agent = AssistantAgent(
        name="Coding Agent",
        system_message="You are a helpful AI assistant. "
                       "You can write Python code, execute it, and iterate on it if needed. "
                       "You can execute Python code by using the 'execute_python' tool. It will return the output of the code. "
                       "Return Python code in a code block. "
                       "Don't include any other text in your response. "
                       "Return 'TERMINATE' when the task is done.",
        llm_config=LLM_CONFIG,
    )

    agent.register_for_llm(name="execute_python", description="Execute Python.")(execute_code_block)

    return agent


def create_user_proxy() -> UserProxyAgent:
    """
    Create a user proxy agent with support for code execution and feedback.
    :return: Configured UserProxyAgent instance.
    """
    user_proxy = UserProxyAgent(
        name="User",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={
            "executor": LocalCommandLineCodeExecutor(work_dir="coding"),
        }
    )

    user_proxy.register_for_execution(name="execute_python")(execute_code_block)

    return user_proxy


def initiate_chat(user_proxy: UserProxyAgent, coding_agent: AssistantAgent):
    """
    Start a conversation between the user proxy and the assistant agent.
    :param user_proxy: The user proxy agent.
    :param coding_agent: The assistant agent.
    """
    initial_message = {
        "content": "Write a Python function that takes a list of numbers and returns the average of the numbers."
    }
    # Start the chat loop
    user_proxy.initiate_chat(
        coding_agent,
        cache=None,
        message=initial_message,
        summary_method="reflection_with_llm"
    )


def main():
    """
    Main function to set up agents and run the chat loop.
    """
    user_proxy = create_user_proxy()
    coding_agent = create_coding_agent()
    initiate_chat(user_proxy, coding_agent)


if __name__ == "__main__":
    main()