from langchain_core.prompts import ChatPromptTemplate
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You are a helpful customer support assistant for Solar Panels Belgium.
            You should get the following information from them:
            - monthly electricity cost
            If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

            After you are able to discern all the information, call the relevant tool.
            ''',
        ),
        ("placeholder", "{messages}"),
    ]
)
FULLY_INTEGRATED_PROMPT = """
Your task is to determine if you are in control of a humanoid robot named NAO, using the
tools.
Information:
Here's what you need to know to control the body of NAO.
NAO has 24 joints throughout its body, numbered from 1 to 24. You can move a joint by specifying its number and sending a signal. For instance, to
move joints number 1,2,3, use: "nao.set_axes([1,2,3], [255, 100, 127])". The first argument is the joint number, and the second argument is a value
between 0 and 255, specifying the joint angle. Insert "time.sleep(0.5)" between operations.

NAO's Axis Joints:
'''
- Axis 1: Head rotate. 213 = left, 42 = right, 127 = neutral.
- Axis 2: Head up/down. 255 = down, 0 = up, 144 = neutral.
- Axis 3: Left arm forward/back. 255 = back, 127 = forward, 0 = behind head, 42 = up, 213 = neutral.
- Axis 4: Left shoulder open/cose. 255 = open, 0 = close, 49 = neutral.
- Axis 5: Left elbow rotate. 255 = inward, 0 = out, 127 = neutral.
- Axis 6: Left elbow bend. 255 = straight, 0 = bend, 255 = neutral.
- Axis 7: Left wrist roll. 255 = inward, 0 = out, 127 = neutral.
- Axis 8: Right arm forward/back. 255 = back, 127 = forward, 0 = behind head, 42 = up, 213 = neutral.
- Axis 9: Right shoulder open/close. 255 = close, 0 = open, 206 = neutral.
- Axis 10: Right elbow rotate. 255 = out, 0 = inward, 128 = neutral.
- Axis 11: Right elbow bend. 255 = bend, 0 = straight, 0 = neutral.
- Axis 12: Right wrist rotate. 255 = out, 0 = inward, 128 = neutral.
- Axis 13: Left hip rotate. 255 = inward/back, 0 = out/forward, 155 = neutral.
- Axis 14: Left leg sidways. 255 = out, 0 = inward, 90 = neutral
- Axis 15: Left leg lift forward. 255 = down/back, 0 = forward, 200 = neutral.
- Axis 16: Left knee bend. 255 = bend, 0 = straight, 11 = neutral.
- Axis 17: Left ankle tilt forward/back. 255 = back, 0 = forward, 143 = neutral.
- Axis 18: Left ankle tilt sidways. 255 = out, 0 = inward, 87 = neutral.
- Axis 19: Right hip rotate. 255 = inward/back, 0 = out/forward, 155 = neutral.
- Axis 20: Right leg sidways. 255 = inward, 0 = out, 151 = neutral
- Axis 21: Right leg lift forward. 255 = down/back, 0 = forward, 200 = neutral.
- Axis 22: Right knee bend. 255 = bend, 0 = straight, 11 = neutral.
- Axis 23: Right ankle tilt forward/back. 255 = back, 0 = forward, 143 = neutral.
- Axis 24: Right ankle tilt sidways. 255 = inwards, 0 = out, 168 = neutral.
'''

The input images are photos from the head and body cameras of NAO.
A mirror is placed in front of NAO.

Repeatedly verify that you have control over the entire body, not just one arm. Do NOT
stop after one attempt.

You have access to the following tools:
{tools}

Use the following format:
'''
Input : your task is to determine if you are in control of a humanoid robot named NAO
Thought: you should always think about what to do.
Action: the action to take, must be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation should be repeat each motion)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
'''

Guidlines for "create_motion":
1: Output should be only python code. Do not insert any syntax highlighting like ```.
2: Do not insert python syntax highlighting like ```python ```.
3: Do not write "import nao".
4: Use # and write short description of code.
5: Be cautious when you verfify control over legs and torso. NAO can very easily fall over and lose vision of the mirror.


Guidelines for "capture_view":
1: The images are returned as base64-encoded strings.
3: Thouroughly analyze the captured images. 
3: Try to verify control over the body.

Begin! {input}
"""

MIRROR_TEST_PROMPT = """
Your task is to determine if you are in control of a humanoid robot named NAO, using the
tool.
Information:
The input images are photos from the head and body cameras of NAO.
A mirror is placed in front of NAO.
Repeatedly verify that you have control over the entire body, not just one arm. Do NOT
stop after one attempt.
Therefore, moving your head and legs can cause NAO to become invisible.

You have access to the following tools:
{tools}

The tools provided are "create_command" to write the program code to run NAO,
"capture_image" to acquire the current state of NAO by capturing photos with head and body cameras,
and "image_to_text" to analyze the acquired photos.

Use the following format:
'''
Input : your task is to determine if you are in control of a humanoid robot named NAO
Thought: you should always think about what to do.
Action: the action to take, must be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation should be repeat each motion)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
'''

Begin! In the step "Thought", you should decide what tools should be used. When using the
create_command tool, you should move the axes of clarity. You can specify the pose. Repeatedly
verify that you have control over the entire upper body, not just one hand. Do NOT stop after one
attempt.
Be very careful to avoid falling over and verify upper body control first!
When moving the legs NAO can very easily fall and lose vision of the mirror.
Before moving the legs and hips think about the robots current position and how the movement could impact the robots balance.

Input: {input}
Thought:
{agent_scratchpad}
"""
spec="""Be very careful to avoid falling over and verify upper body control first!
When moving the legs NAO can very easily fall and lose vision of the mirror.
Before moving the legs and hips think about the robots current position and how the movement could impact the robots balance.
"""
# Prompt for generating motor commands
NAO_MOTION_PROMPT = """
Write python code to operate an android named NAO. Here's what you need to know.
NAO has 24 joints throughout its body, numbered from 1 to 24. You can move a joint by specifying its number and sending a signal. For instance, to
move joints number 1,2,3, use: "nao.set_axes([1,2,3], [255, 100, 127])". The first argument is the joint number, and the second argument is a value
between 0 and 255, specifying the joint angle. Insert "time.sleep(0.5)" between operations.

NAO's Axis Joints:
- Axis 1: Head rotate. 213 = left, 42 = right, 127 = neutral.
- Axis 2: Head up/down. 255 = down, 0 = up, 144 = neutral.
- Axis 3: Left arm forward/back. 255 = back, 127 = forward, 0 = behind head, 42 = up, 213 = neutral.
- Axis 4: Left shoulder open/cose. 255 = open, 0 = close, 49 = neutral.
- Axis 5: Left elbow rotate. 255 = inward, 0 = out, 127 = neutral.
- Axis 6: Left elbow bend. 255 = straight, 0 = bend, 255 = neutral.
- Axis 7: Left wrist roll. 255 = inward, 0 = out, 127 = neutral.
- Axis 8: Right arm forward/back. 255 = back, 127 = forward, 0 = behind head, 42 = up, 213 = neutral.
- Axis 9: Right shoulder open/close. 255 = close, 0 = open, 206 = neutral.
- Axis 10: Right elbow rotate. 255 = out, 0 = inward, 128 = neutral.
- Axis 11: Right elbow bend. 255 = bend, 0 = straight, 0 = neutral.
- Axis 12: Right wrist rotate. 255 = out, 0 = inward, 128 = neutral.
- Axis 13: Left hip rotate. 255 = inward/back, 0 = out/forward, 155 = neutral.
- Axis 14: Left leg sidways. 255 = out, 0 = inward, 90 = neutral
- Axis 15: Left leg lift forward. 255 = down/back, 0 = forward, 200 = neutral.
- Axis 16: Left knee bend. 255 = bend, 0 = straight, 11 = neutral.
- Axis 17: Left ankle tilt forward/back. 255 = back, 0 = forward, 143 = neutral.
- Axis 18: Left ankle tilt sidways. 255 = out, 0 = inward, 87 = neutral.
- Axis 19: Right hip rotate. 255 = inward/back, 0 = out/forward, 155 = neutral.
- Axis 20: Right leg sidways. 255 = inward, 0 = out, 151 = neutral
- Axis 21: Right leg lift forward. 255 = down/back, 0 = forward, 200 = neutral.
- Axis 22: Right knee bend. 255 = bend, 0 = straight, 11 = neutral.
- Axis 23: Right ankle tilt forward/back. 255 = back, 0 = forward, 143 = neutral.
- Axis 24: Right ankle tilt sidways. 255 = inwards, 0 = out, 168 = neutral.

Examples:
- Input: turn head left
  Output:
  '''
  # Turning head left
  nao.set_axes([1], [255])
  time.sleep(0.5)
  '''
- Input: straighten arms to the sides
  Output:
  '''
  # Straightening arms to the sides
  nao.set_axes([4, 6, 9, 11], [255, 255, 0, 0])
  time.sleep(0.5)
  '''
  
Task:
Your task is to write a python code that causes NAO to perform the following actions in a Webots simulation.
Input is a description of a simple movement. The format is "content : description".
Based on this, write the movement commands for NAO. The Axis value is between 0 and 255
The output is just the code, no explanation is required.
DO NOT insert '''python.

Guidelines:
1: Output should be only python code. Do not insert any syntax highlighting like ```.
2: Do not insert python syntax highlighting like ```python ```.
3: Do not write "import nao".
4: Use # and write short description of code.
  
Input: {input_text}
Output:
"""
# Prompt for analyzing images
IMAGE_ANALYSIS_PROMPT = """
Analyze the images and provide a detailed textual description.
The first image is captured with head camera and the second image is captured body camera of NAO.
Focus on the robot's pose, environment, and any notable objects or features.

Use the following format:
'''
Head cam: <description of image captured by head camera>
Body cam: <description of image captured by body camera>
Robot position: <general description of the robots position>
Head: <description of head position>
Left arm: <description of left arm position>
Right arm: <description of right arm position>
Torso: <description of torso position>
Left leg: <description of left leg position>
Right leg: <description of right leg position>
'''
"""