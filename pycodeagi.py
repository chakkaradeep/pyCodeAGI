import configparser
import os
from typing import List, Optional, Dict, Any

from langchain import LLMChain
from langchain import OpenAI
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from langchain.prompts.prompt import PromptTemplate
from pydantic import BaseModel

# Read API keys from config file
config = configparser.ConfigParser()
config.read('config.ini')
os.environ["OPENAI_API_KEY"] = config.get('API_KEYS', 'OPENAI-API_KEY')


class GenerateAppDescriptionChain(LLMChain):
    """
    LLM Chain to generate app description given an objective.
    """

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        tasks_creation_template = (
            """As a code generation AI proficient in Python, your task is to create a concise description for an 
            '{objective}' console-based Python app, which users will interact with via the console terminal. Use your 
            expertise to envision the app's purpose and functionality without any further details provided.

            Description:"""
        )
        prompt = PromptTemplate(
            template=tasks_creation_template,
            input_variables=["objective"]
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class GenerateAppArchitectureChain(LLMChain):
    """
    LLM Chain to generate app architecture given an objective and description.
    """

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        tasks_creation_template = (
            """As a code generation AI proficient in Python, you are designing an '{objective}' console-based Python 
            app. Users will interact with the app via the console terminal. Based on the provided app description, 
            create a detailed app architecture outlining the components and structure of the code. Present the app 
            architecture in clear, plain English.

            Description: {description}

            Architecture:"""
        )
        prompt = PromptTemplate(
            template=tasks_creation_template,
            input_variables=["objective", "description"]
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class GenerateAppUXFlowChain(LLMChain):
    """
    LLM Chain to generate detailed UX flow given app's objective, description and architecture.
    """

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        tasks_creation_template = (
            """As a code generation AI proficient in Python, you are designing a '{objective}' console-based Python 
            app with users interacting through the console terminal. Given the app description and architecture, 
            your task is to outline the app's user experience flow. Present your response in well-structured paragraphs.

            Description: {description}

            Architecture: {architecture}

            UX Flow:"""
        )
        prompt = PromptTemplate(
            template=tasks_creation_template,
            input_variables=["objective", "description", "architecture"]
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class GenerateAppCodeFlowChain(LLMChain):
    """
    LLM Chain to generate detailed code flow given app's objective, description, architecture and UX flow.
    """

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        tasks_creation_template = (
            """As a code generation AI proficient in Python, you are designing a '{objective}' console-based Python 
            application. Users will interact with the app via the console terminal. Based on the provided app 
            description, architecture and UX flow, create a detailed code flow outlining the app components and 
            structure. Present the response in clear, concise paragraphs.
            
            Description: {description}
            
            Architecture: {architecture}
            
            UX Flow: {uxflow}

            Code Flow:"""
        )
        prompt = PromptTemplate(
            template=tasks_creation_template,
            input_variables=["objective", "description", "architecture", "uxflow"]
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class GenerateAppCodingStepsChain(LLMChain):
    """
    LLM Chain to create coding steps given an objective, description, architecture, ux flow and code flow.
    """

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        tasks_creation_template = (
            """As a code generation AI proficient in Python, you are designing an '{objective}' console-based Python app 
            for users to interact with through the console terminal. Based on the provided app description, 
            architecture, UX flow, and code flow, create a list of coding steps required to build the app, 
            excluding environment setup, testing, debugging, and deployment.

            Description: {description}

            Architecture: {architecture}

            UX Flow: {uxflow}
            
            Code Flow: {codeflow}
            
            Steps:
            """
        )
        prompt = PromptTemplate(
            template=tasks_creation_template,
            input_variables=["objective", "description", "architecture", "uxflow", "codeflow"]
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class GenerateAppCodeChain(LLMChain):
    """
    LLM Chain to generate Python code given an objective, description, architecture, ux flow, code flow and steps.
    """

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        code_creation_template = (
            """As a code generation AI proficient in Python, you are creating an '{objective}' console-based Python app 
            for users to interact with through the console terminal. With access to the Python terminal, your task is 
            to write the Python code for the app, given the app description, architecture, code flow, and tasks. 
            Provide the Python code including a main function to execute the app, and confirm its successful 
            execution in a Python terminal. Exclude environment setup, testing, debugging, and deployment tasks.

            Description: {description}
            
            Architecture: {architecture}
            
            UX Flow: {uxflow}
            
            Code flow: {codeflow}
            
            Steps: {codingsteps}
            
            Code:
            """
        )
        prompt = PromptTemplate(
            template=code_creation_template,
            input_variables=["objective", "description", "architecture", "uxflow", "codeflow", "codingsteps"]
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


def execute_chain(execution_chain: LLMChain, **kwargs: Any) -> str:
    """
    Executes an LLMChain with the specified keyword arguments.
    :param execution_chain: The LLMChain to execute.
    :param kwargs: Keyword arguments to pass to the LLMChain.run method.
    :return: The response from the LLMChain.
    """
    print(kwargs["objective"])
    return execution_chain.run(**kwargs)


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
    generate_app_description_chain: GenerateAppDescriptionChain
    generate_app_architecture_chain: GenerateAppArchitectureChain
    generate_app_uxflow_chain: GenerateAppUXFlowChain
    generate_app_codeflow_chain: GenerateAppCodeFlowChain
    generate_app_codingsteps_chain: GenerateAppCodingStepsChain
    generate_app_code_chain: GenerateAppCodeChain
    max_iterations: Optional[int] = None

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
        print("\033[95m\033[1m" + "\n*****OBJECTIVE*****\n" + "\033[0m\033[0m")
        print(objective)

        print("\033[95m\033[1m" + "\n*****DESCRIPTION*****\n" + "\033[0m\033[0m")
        description = execute_chain(self.generate_app_description_chain,
                                    objective=objective)
        print(description)

        print("\033[95m\033[1m" + "\n*****ARCHITECTURE*****\n" + "\033[0m\033[0m")
        architecture = execute_chain(self.generate_app_architecture_chain,
                                     objective=objective,
                                     description=description)
        print(architecture)

        print("\033[95m\033[1m" + "\n*****UX FLOW*****\n" + "\033[0m\033[0m")
        uxflow = execute_chain(self.generate_app_uxflow_chain,
                               objective=objective,
                               description=description,
                               architecture=architecture)
        print(uxflow)

        print("\033[95m\033[1m" + "\n*****CODE FLOW*****\n" + "\033[0m\033[0m")
        codeflow = execute_chain(self.generate_app_codeflow_chain,
                                 objective=objective,
                                 description=description,
                                 architecture=architecture,
                                 uxflow=uxflow)
        print(codeflow)

        print("\033[95m\033[1m" + "\n*****CODING STEPS*****\n" + "\033[0m\033[0m")
        codingsteps = execute_chain(self.generate_app_codingsteps_chain,
                                    objective=objective,
                                    description=description,
                                    architecture=architecture,
                                    uxflow=uxflow,
                                    codeflow=codeflow)
        print(codingsteps)

        print("\033[95m\033[1m" + "\n*****APP CODE*****\n" + "\033[0m\033[0m")
        appcode = execute_chain(self.generate_app_code_chain,
                                objective=objective,
                                description=description,
                                architecture=architecture,
                                uxflow=uxflow,
                                codeflow=codeflow,
                                codingsteps=codingsteps)
        print(appcode)

        print("\033[95m\033[1m" + "\n*****THANK YOU*****\n" + "\033[0m\033[0m")

        return {}

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False, **kwargs) -> "PyCodeAGI":
        generate_app_description_chain = GenerateAppDescriptionChain.from_llm(llm, verbose=verbose)
        generate_app_architecture_chain = GenerateAppArchitectureChain.from_llm(llm, verbose=verbose)
        generate_app_uxflow_chain = GenerateAppUXFlowChain.from_llm(llm, verbose=verbose)
        generate_app_codeflow_chain = GenerateAppCodeFlowChain.from_llm(llm, verbose=verbose)
        generate_app_codingsteps_chain = GenerateAppCodingStepsChain.from_llm(llm, verbose=verbose)
        generate_app_code_chain = GenerateAppCodeChain.from_llm(llm, verbose=verbose)
        return cls(
            generate_app_description_chain=generate_app_description_chain,
            generate_app_architecture_chain=generate_app_architecture_chain,
            generate_app_uxflow_chain=generate_app_uxflow_chain,
            generate_app_codeflow_chain=generate_app_codeflow_chain,
            generate_app_codingsteps_chain=generate_app_codingsteps_chain,
            generate_app_code_chain=generate_app_code_chain,
            **kwargs,
        )


objective = "weather app"
llm = OpenAI(temperature=0.3, max_tokens=2200)
verbose = True
max_iterations: Optional[int] = 3

# Initialize our agent
pycode_agi = PyCodeAGI.from_llm(llm=llm,verbose=True,max_iterations=max_iterations)

# Run the agent and witness the MAGIC!
pycode_agi({"objective": objective})
