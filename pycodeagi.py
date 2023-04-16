import configparser
import os
from typing import List, Optional, Dict, Any

from langchain import OpenAI, LLMChain
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from langchain.prompts.prompt import PromptTemplate
from pydantic import BaseModel

# Read API keys from config file
config = configparser.ConfigParser()
config.read('config.ini')
os.environ["OPENAI_API_KEY"] = config.get('API_KEYS', 'OPENAI-API_KEY')


class CreateCodingTasksChain(LLMChain):
    """
    LLM Chain to create coding tasks given an objective.
    """
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        tasks_creation_template = (
            """
            You are task creation AI. 
            You have knowledge on Python programming.   
            Describe the Python code in plain English you will use to build '{objective}' console app.  
            The tasks will only include what is required to write code. 
            Do not create tasks for debugging, testing and deployment.       
            Based on the results, create new tasks, a maximum of 15 tasks, to be completed to generate the Python code.
            Return the tasks as numbered list.   
            """
        )
        prompt = PromptTemplate(
            template=tasks_creation_template,
            input_variables=["objective"]
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


# class RateCodeTasksChain(LLMChain):
    """
    LLM Chain to rate the tasks to accomplish the objective.
    """
    # raise NotImplementedError("Rate Code Tasks Chain is not implemented yet.")


# class RefineCodeTasksChain(LLMChain):
    """
    LLM Chain to refine the highest scored tasks to accomplish the objective.
    """
    # raise NotImplementedError("Refine Code Instructions Chain is not implemented yet.")


class GenerateCodeChain(LLMChain):
    """
    LLM Chain to generate Python code given an objective and instructions.
    """
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        code_creation_template = (
            """
            You are an AI tool that excels in writing Python code.
            You will create an python console called {objective}.
            You will write code based on the following instructions: {instructions}
            Return the Python code.
            
            Python:
            import
            """
        )
        prompt = PromptTemplate(
            template=code_creation_template,
            input_variables=["objective", "instructions"]
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


def execute_create_task(execution_chain: LLMChain, objective: str) -> str:
    """
    Executes the chain to create tasks.
    :param execution_chain:
    :param objective:
    :return: response
    """
    return execution_chain.run(objective=objective)


def execute_code_gen_task(execution_chain: LLMChain, objective: str, instructions: str) -> str:
    """
    Executes the chain to create code.
    :param execution_chain:
    :param objective:
    :param instructions:
    :return: response
    """
    return execution_chain.run(objective=objective, instructions=instructions)


class PyCodeAGI(Chain, BaseModel):
    """
    Our AGI that performs the MAGIC!
    """
    create_coding_tasks_chain: CreateCodingTasksChain
    generate_coding_chain: GenerateCodeChain

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        objective = inputs["objective"]
        print("\033[95m\033[1m" + "\n*****OBJECTIVE*****\n" + "\033[0m\033[0m")
        print(objective)

        code_instructions = execute_create_task(self.create_coding_tasks_chain, objective)
        print("\033[95m\033[1m" + "\n*****CODING TASKS*****\n" + "\033[0m\033[0m")
        print(code_instructions)

        generated_code = execute_code_gen_task(self.generate_coding_chain,
                                               objective=objective,
                                               instructions=code_instructions)

        print("\033[95m\033[1m" + "\n*****CODE CREATED*****\n" + "\033[0m\033[0m")
        print(generated_code)

        print("\033[95m\033[1m" + "\n*****THANK YOU*****\n" + "\033[0m\033[0m")

        return {}

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False, **kwargs) -> "PyCodeAGI":
        create_coding_tasks_chain = CreateCodingTasksChain.from_llm(
            llm,
            verbose=verbose)
        generate_coding_chain = GenerateCodeChain.from_llm(
            llm,
            verbose=verbose)
        return cls(
            create_coding_tasks_chain=create_coding_tasks_chain,
            generate_coding_chain=generate_coding_chain,
            **kwargs,
        )


# TODO: Get user input here
objective = "weather app"
llm = OpenAI(temperature=0.8, max_tokens=300)
verbose = True
max_iterations: Optional[int] = 3

# Initialize our agent
pycode_agi = PyCodeAGI.from_llm(
    llm=llm,
    verbose=verbose,
    max_iterations=max_iterations
)

# Run the agent and witness the MAGIC!
pycode_agi({"objective": objective})
