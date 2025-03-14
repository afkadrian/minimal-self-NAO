from langchain.tools import BaseTool, Tool
from langchain.agents import AgentExecutor, create_react_agent

from langchain_openai import ChatOpenAI
#from langchain import LangChain
#from lanchain.models import GPT4
from langchain.prompts import PromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.messages import HumanMessage
from langchain.agents.output_parsers.json import JSONAgentOutputParser

import requests
import os

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

from typing import Dict, List, Optional, Sequence, Union
import base64
from simple_colors import get_color_code
from prompts import IMAGE_ANALYSIS_PROMPT, NAO_MOTION_PROMPT, MIRROR_TEST_PROMPT
from llm import gpt4

from datetime import datetime

# Server URL for the robot's API
server_url = "http://localhost:5000"


image_dir = "./Images/"
run_id = 0
os.mkdir(image_dir)

image_id = 0
# Tool: CaptureImage
class CaptureImage(BaseTool):
    name: str = "capture_image"
    description: str = (
        "Captures images using the robot's head and body cameras."
    )
    server_url: str = None

    def __init__(self, server_url: str):
        super().__init__()
        self.server_url = server_url

    def _run(self, _="Capturing images with head and body cameras.") -> str:
        try:
            global image_id
            payload = {"id": image_id}
            response = requests.post(self.server_url, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return "Images captured."
            if response.json()["status"] == "success":
                return response.json()["image_id"]
            else:
                return response.json()["message"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error capturing image: {e}")

    async def _arun(self, input):
        raise NotImplementedError("This tool does not support async execution.")

# Tool: Image2Text
class Image2Text(BaseTool):
    name: str = "image_to_text"
    description: str = ("Analyzes the images captured by the head and the body camera of NAO and generates a textual description.")
    llm: ChatOpenAI = None
    prompt: str = None

    def __init__(self, llm: ChatOpenAI, prompt: str):
        super().__init__()
        self.llm = llm
        self.prompt = prompt

    def load_image(self, image_path):
        
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
            return image_base64

    def _run(self, _="Analyze the captured images.") -> str:
        global image_id
        try:
            id = int(image_id)
            head_cam = os.path.join(f"{image_dir}run_{run_id}/", f"head_cam{id}.jpeg")
            body_cam = os.path.join(f"{image_dir}run_{run_id}/", f"body_cam{id}.jpeg")
            if not os.path.exists(head_cam) or not os.path.exists(body_cam):
                return f"Error: Image file {image_id} not found at {image_dir}run{run_id}/"
            #image_id += 1
            # Proceed with image processing
            encoded_img_head = self.load_image(head_cam)
            encoded_img_body = self.load_image(body_cam)

            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img_head}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img_body}"}}
                ]
            )

            #formatted_prompt = self.prompt_template.format(image_url=f"data:image/jpeg;base64,{encoded_img}")
            description = self.llm.invoke([message])
            image_id += 1
            #print(f"\n{description.content}\n")
            return description.content
        except Exception as e:
            return f"Error analyzing image: {e}"

    async def _arun(self, input = "Analyze the captured images.") -> str:
        raise NotImplementedError("This tool does not support async execution.")

# Tool: GenerateMotion
class GenerateMotion(BaseTool):
    name: str = "generate_motion"
    description: str = (
        "Creates a motor command to control the NAO robot based on a textual description of a simple motion."
    )
    llm: ChatOpenAI = None
    prompt: str = None
    prompt_template: List[Dict] = None
    server_url: str = None

    def __init__(self, server_url: str, llm: ChatOpenAI):
        super().__init__()
        self.server_url = server_url
        self.llm = llm
        self.prompt = NAO_MOTION_PROMPT
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> List[Dict]:
        try:
            return [
                {"role": "system", "content": "You are a robot motion interpreter for the NAO humanoid robot."},
                {"role": "user", "content": self.prompt},
            ]
        except Exception as e:
            raise Exception(f"Error loading prompt template: {e}")


    def _send_to_server(self, code: str) -> dict:
        try:
            payload = {"code": f"{code}"}
            response = requests.post(self.server_url, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error sending data to server: {e}")


    def _run(self, input_text: str) -> str:
        try:
            # Generate the prompt
            messages = self.prompt_template.copy()
            messages.append({"role": "user", "content": input_text})

            # Generate motor commands using the prompt template
            #formatted_prompt = self.prompt_template.format(input_text=input_text)
            response = self.llm.invoke(messages)
            code = response.content
            server_response = self._send_to_server(code=code)
            #time.sleep(2)
            #return server_response['positions']
            return f"Generated code\n{code}"
        except Exception as e:
            return f"\nError generating motor commands: {e}\n"

    async def _arun(self, input_text: str):
        raise NotImplementedError("This tool does not support async execution.")

from langchain.agents.format_scratchpad import format_log_to_str, log_to_messages

# Initialize the agent and tools
def initialize_agent_and_tools():
    # Initialize the tools
    capture_image = CaptureImage(server_url=f"{server_url}/capture_image")
    image_to_text = Image2Text(llm=gpt4, prompt=IMAGE_ANALYSIS_PROMPT)
    generate_motion = GenerateMotion(server_url=f"{server_url}/set_joints", llm=gpt4)
    tools = [
        Tool(
            name="capture_image",
            func=capture_image._run,
            description=capture_image.description,
        ),
        Tool(
            name="image_to_text",
            func=image_to_text._run,
            description=image_to_text.description,
        ),
        Tool(
            name="generate_motion",
            func=generate_motion._run,
            description=generate_motion.description,
        ),
    ]

    # Initialize the agent
    msr_prompt = PromptTemplate.from_template(template=MIRROR_TEST_PROMPT)
    agent = create_react_agent(gpt4, tools, msr_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True, max_iterations=100, return_intermediate_steps=True)

    return agent_executor

# Main loop for agent interaction
def main():
    while True:
        data = None
        global run_id
        global image_id
        image_id = 0
        while not data or not data["status"] == "success":
            run_id = input("\nEnter the run id: ")
            if run_id.lower() in ["exit", "quit"]:
                print("Exiting the agent.")
                return
            try:
                payload = {"id": run_id}
                response = requests.post(f"{server_url}/run_id", json=payload)
                response.raise_for_status()  # Raise an exception for HTTP errors
                data = response.json()
            except requests.exceptions.RequestException as e:
                raise Exception(f"Error sending data to server: {e}")
            print(f"<===== SERVER RESPONSE =====>")
            print(data["message"])


        print(f"<===== STARTING RUN {run_id} =====>")

        agent = initialize_agent_and_tools()
        response = agent.invoke({"input": "your task is to determine if you are in control of a humanoid robot named NAO"})

        messages = format_log_to_str(response["intermediate_steps"])


        with open(f"Experiments/{run_id}.txt", "w") as file:
            file.write(f"Thought: {messages}\nFinal response:\n{response['output']}")
            #[file.write(f"{msg}\n") for msg in messages]

        #print(messages)
        print("Final response:")
        print(response['output'])


if __name__ == "__main__":
    main()
