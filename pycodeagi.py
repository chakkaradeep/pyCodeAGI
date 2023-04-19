import configparser
import os
from typing import List, Dict, Any

from langchain import LLMChain
from langchain import OpenAI
from langchain.chains.base import Chain
from langchain.prompts.prompt import PromptTemplate
from pydantic import BaseModel

# Read API keys from config file
config = configparser.ConfigParser()
config.read('config.ini')
os.environ["OPENAI_API_KEY"] = config.get('API_KEYS', 'OPENAI-API_KEY')


class GeneratePyCodeChain(LLMChain):
    """
    The main LLM Chain class that runs every step.
    """
    @classmethod
    def create_chain(cls, verbose: bool = False) -> LLMChain:
        prompt_template = ("""
            You are code generation AI proficient in Python.\n
            Your task is to build a '{objective}' console-based Python app.\n 
            {maincontent}.\n
            {outcome}:""")
        prompt = PromptTemplate(template=prompt_template, input_variables=["objective", "maincontent", "outcome"])

        llm = OpenAI(model_name="text-davinci-003",
                     temperature=0.3)

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
        print("\033[93m" + "*****OBJECTIVE*****" + "\033[0m")
        print(objective.strip())

        print("\033[93m" + "*****DESCRIPTION*****" + "\033[0m")
        maincontent = """
                Your task is to create a concise description for the console-based Python app.\n
                Users will interact with the app in a console terminal.\n
                Use your expertise to envision the app's purpose and functionality.
                """
        outcome = "Description"
        self.llm_chain.llm.max_tokens = 200
        description = self.llm_chain.run(objective=objective,
                                         maincontent=maincontent,
                                         outcome=outcome)
        print(description.strip())

        print("\033[93m" + "*****ARCHITECTURE*****" + "\033[0m")
        maincontent = f"""
            Based on the provided app description, create a detailed app architecture.\n
            Outline the components and structure of the code.\n
            Present the app architecture in an ordered list.\n
            Description: {description}
            """
        outcome = "Architecture"
        self.llm_chain.llm.max_tokens = 350
        architecture = self.llm_chain.run(objective=objective,
                                          maincontent=maincontent,
                                          outcome=outcome)
        print(architecture.strip())

        print("\033[93m" + "*****UX FLOW*****" + "\033[0m")
        maincontent = f"""
                Based on the app description and architecture outline the app UX flow.\n
                Present the UX flow an ordered list.\n
                Description: {description}\n
                Architecture: {architecture}"""
        outcome = "UX Flow"
        self.llm_chain.llm.max_tokens = 400
        uxflow = self.llm_chain.run(objective=objective,
                                    maincontent=maincontent,
                                    outcome=outcome)
        print(uxflow.strip())

        print("\033[93m" + "*****CODE FLOW*****" + "\033[0m")
        maincontent = f"""
            Based on the app description, architecture and UX flow, create a detailed code flow.\n
            Outline the code components and structure.\n
            Present the code flow in an ordered list.\n
            Description: {description}\n
            Architecture: {architecture}\n
            UX Flow: {uxflow}"""
        outcome = "Code Flow"
        self.llm_chain.llm.max_tokens = 400
        codeflow = self.llm_chain.run(objective=objective,
                                      maincontent=maincontent,
                                      outcome=outcome)
        print(codeflow.strip())

        print("\033[93m" + "*****CODING STEPS*****" + "\033[0m")
        maincontent = f"""
            You are provided with the app description, architecture, UX flow, and code flow.\n
            Create an ordered list of coding steps required to build the app.\n
            Exclude environment setup, testing, debugging, and deployment steps.\n
            Description: {description}\n
            Architecture: {architecture}\n
            UX Flow: {uxflow}\n
            Code Flow: {codeflow}"""
        outcome = "Coding Steps"
        self.llm_chain.llm.max_tokens = 400
        codingsteps = self.llm_chain.run(objective=objective,
                                         maincontent=maincontent,
                                         outcome=outcome)
        print(codingsteps.strip())

        print("\033[93m" + "*****APP CODE*****" + "\033[0m")
        maincontent = f"""
            With access to the Python terminal, your task is to write the Python code for the app.\n
            You are given the app description, architecture, code flow, and tasks.\n
            Write the Python code with a main function to execute the app in a console terminal.\n
            Avoid using database for backend storage, instead use in-memory options.
            Exclude environment setup, testing, debugging, and deployment tasks.\n
            Description: {description}\n
            Architecture: {architecture}\n
            UX Flow: {uxflow}\n
            Code Flow: {codeflow}\n
            Coding Steps: {codingsteps}'"""
        outcome = "App Code"
        self.llm_chain.llm.max_tokens = 3000
        appcode = self.llm_chain.run(objective=objective,
                                     maincontent=maincontent,
                                     outcome=outcome)
        print(appcode.strip())

        print("\033[93m" + "\n*****THANK YOU*****\n" + "\033[0m")

        return {}

    @classmethod
    def create_llm_chain(cls, verbose: bool = False) -> "PyCodeAGI":
        llm_chain = GeneratePyCodeChain.create_chain(verbose=verbose)
        return cls(llm_chain=llm_chain)


objective = "calculator app"

# Initialize our agent
pycode_agi = PyCodeAGI.create_llm_chain()
# Run the agent and witness the MAGIC!
pycode_agi({"objective": objective})