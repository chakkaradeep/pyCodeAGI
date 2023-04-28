import configparser
import os
import re
import ast
from typing import List, Dict, Any

from langchain import LLMChain
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate
from pydantic import BaseModel

# Read API keys from config file
config = configparser.ConfigParser()
config.read('config.ini')
os.environ["OPENAI_API_KEY"] = config.get('API_KEYS', 'OPENAI-API_KEY')

output_file = "output_steps.txt"
code_file = "app.py"


class GeneratePyCodeChain(LLMChain):
    """
    The main LLM Chain class that runs every step.
    """

    @classmethod
    def create_chain(cls, verbose: bool = False) -> LLMChain:
        system_template = ("""
            You are code generation AI proficient in Python.\n
            Your goal is to build console-based Python app.\n 
            {instructions}.""")
        system_prompt_template = PromptTemplate(template=system_template, input_variables=["instructions"])
        system_message_prompt = SystemMessagePromptTemplate(prompt=system_prompt_template)

        user_template = "{tasks}"
        user_prompt_template = PromptTemplate(template=user_template, input_variables=["tasks"])
        user_message_prompt = HumanMessagePromptTemplate(prompt=user_prompt_template)

        prompt = ChatPromptTemplate.from_messages([system_message_prompt, user_message_prompt])
        llm = ChatOpenAI(model_name="gpt-4",
                         temperature=0.35,
                         request_timeout=180)
        chain_instance = cls(prompt=prompt, llm=llm)
        return chain_instance


class PyCodeAGI(Chain, BaseModel):
    """
    Our AGI that performs the MAGIC!
    """
    llm_chain: GeneratePyCodeChain

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        objective = inputs["objective"]
        print("\033[93m" + "\n" + "*****OBJECTIVE*****" + "\033[0m")
        print(objective.strip())
        with open(output_file, "a") as f:
            f.write(f"Objective: \n {objective.strip()}\n\n")

        print("\033[93m" + "*****DESCRIPTION*****" + "\033[0m")
        instructions = "Users will interact with the app in a console terminal."
        tasks = f"""            
            Create a concise description for the console-based Python app: {objective}\n
            Use your expertise to envision the app's purpose and functionality.
        """
        self.llm_chain.llm.max_tokens = 200
        description = self.llm_chain.run(instructions=instructions, tasks=tasks)
        print(description.strip())
        with open(output_file, "a") as f:
            f.write(f"Description: \n {description.strip()}\n\n")

        print("\033[93m" + "*****ARCHITECTURE*****" + "\033[0m")
        instructions = f"""
            You are given the app name and description.\n
            App Name:\n
            {objective}\n
            Description: \n
            {description}
            """
        tasks = f"""
            Create a concise app architecture you can use to build the UX flow.\n
            Outline the components and structure of the code.\n
            Present the app architecture in an ordered list.
            """
        self.llm_chain.llm.max_tokens = 350
        architecture = self.llm_chain.run(instructions=instructions, tasks=tasks)
        print(architecture.strip())
        with open(output_file, "a") as f:
            f.write(f"Architecture: \n {architecture.strip()}\n\n")

        print("\033[93m" + "*****UX FLOW*****" + "\033[0m")
        instructions = f"""
                    You are given the app name, description and architecture.\n
                    App Name:\n
                    {objective}\n
                    Description: \n
                    {description}\n
                    Architecture:\n
                    {architecture}
                    """
        tasks = f"""
                Create a concise UX flow that you can use to build code flow.\n
                Present the UX flow an ordered list.
                """
        self.llm_chain.llm.max_tokens = 700
        uxflow = self.llm_chain.run(instructions=instructions, tasks=tasks)
        print(uxflow.strip())
        with open(output_file, "a") as f:
            f.write(f"UX Flow: \n {uxflow.strip()}\n\n")

        print("\033[93m" + "*****CODE FLOW*****" + "\033[0m")
        instructions = f"""
                        You are given the app name, description, architecture and UX flow.\n
                        App Name:\n
                        {objective}\n
                        Description: \n
                        {description}\n
                        Architecture:\n
                        {architecture}\n
                        UX Flow:\n
                        {uxflow}
                        """
        tasks = f"""
            Create a concise code flow you can use to write code.\n
            Outline the code components and structure.\n
            Present the code flow in an ordered list.
            """
        self.llm_chain.llm.max_tokens = 700
        codeflow = self.llm_chain.run(instructions=instructions, tasks=tasks)
        print(codeflow.strip())
        with open(output_file, "a") as f:
            f.write(f"Code Flow: \n {codeflow.strip()}\n\n")

        print("\033[93m" + "*****APP CODE*****" + "\033[0m")
        instructions = f"""
                        You are given the app name, description, architecture, UX flow and code flow.\n
                        App Name:\n
                        {objective}\n
                        Description: \n
                        {description}\n
                        Architecture:\n
                        {architecture}\n
                        UX Flow:\n
                        {uxflow}
                        Code Flow:\n
                        {codeflow}
                        """
        tasks = f"""
            Write the Python code for the app in a single python file.\n
            Avoid using database for backend storage, instead use in-memory options.\n
            Exclude environment setup, testing, debugging, and deployment tasks.
            """
        self.llm_chain.llm.max_tokens = 4000
        appcode = self.llm_chain.run(instructions=instructions, tasks=tasks)
        print(appcode.strip())
        with open(output_file, "a") as f:
            f.write(f"App Code: \n {appcode.strip()}")

        print("\033[93m" + "\n*****SAVING CODE TO FILE*****\n" + "\033[0m")
        code_match = re.search(r'```python(.*?)```', appcode.strip(), re.DOTALL)
        code_content = code_match.group(1).strip()
        try:
            ast.parse(code_content)
            print("Generated code is AWESOME!")
            with open(code_file, "w") as f:
                f.write(code_content)
            print(f"Code saved to {code_file}.")
        except SyntaxError as e:
            print("OOPS! Something wrong with the code")
            print(f"\nSyntax Error: {e}\n")
            print("Try running the code generator again!")

        print("\033[93m" + "\n*****THANK YOU*****\n" + "\033[0m")

        return {}

    @classmethod
    def create_llm_chain(cls, verbose: bool = False) -> "PyCodeAGI":
        llm_chain = GeneratePyCodeChain.create_chain(verbose=verbose)
        return cls(llm_chain=llm_chain)


if __name__ == "__main__":
    # Delete output files
    if os.path.exists(output_file):
        os.remove(output_file)

    if os.path.exists(code_file):
        os.remove(code_file)

    # Initialize our agent
    pycode_agi = PyCodeAGI.create_llm_chain()

    # Get the user input
    objective = input(f"\nWhat app do you want me to build: ")

    # Run the agent and witness the MAGIC!
    pycode_agi({"objective": objective})
