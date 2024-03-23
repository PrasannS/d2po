# file with different prompt formats for datasets to handle converting between stuff
import re

# standard prompt used for most generators by default
def apfarmstyle(question, response=""):
    return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+question+"\n\n### Response:"+response

# standard prompt used for most RMs by default
def qaform(q, r=""):
    return "Question: " + q + "\n\nAnswer: " + r

# tulu format (TODO these methods look refactorable, see if it's worth later)
def tuluform(q,r=""):
    return "<user>\n"+q+"\n<assistant>\n"+r
    
def splitter(inp):
    # qa style
    if ("Question:" in inp) and "Answer:" in inp: 
        instruction, response = inp.split("\n\nAnswer: ")
        instruction = instruction[len("Question: "):]
    # alpacafarm style
    if "### Instruction:" in inp:
        instruction_match = re.search(r'### Instruction:\n(.*?)(### Response:|\Z)', inp, re.DOTALL)
        instruction = instruction_match.group(1).strip() if instruction_match else inp
        # Extract Response
        response_match = re.search(r'### Response:.*?(.*?)(### |\Z)', inp, re.DOTALL)
        response = response_match.group(1).strip() if response_match else ""
    # tulu style
    if "<user>" in inp: 
        q = inp[len("<user>\n"):]
        instruction, response = q.split("\n<assistant>\n")
    
    return instruction, response
    
# given an input in some style, figure out what style it is, convert it to a new style
def convert_prompstlye(inp, newstyle=apfarmstyle):
    instruction, response  = splitter(inp)
    return newstyle(instruction, response)
