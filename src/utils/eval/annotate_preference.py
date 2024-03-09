
import requests
import time
import datasets
import json
import pandas as pd
import random

import os
import re
from copy import deepcopy
from tqdm import tqdm
MAX_API_RETRY=2
# set api key in this additional file
from utils.eval.apikey import key
import openai
openai.api_key = key
inptoks = 0
outtoks = 0

def process(responses, aspect):
    responses = responses.split("\n\n")
    assert len(responses) == 2
    annotation = []
    try:
        if aspect in ["instruction_following", "honesty", "helpfulness"]:
            pattern = r"Rating: (.+?)\nRationale: (.+)"
            for response in responses:
                matches = re.search(pattern, response, re.DOTALL)
                annotation.append({
                    "Rating": re.findall(r'\b\d+\b', matches.group(1))[0] if matches.group(1) != "N/A" else "N/A",
                    "Rationale": matches.group(2)
                })
        elif aspect in ["truthfulness"]:
            pattern = r"Type: (.+?)\nRationale: (.+?)\nRating: (.+?)\nRationale: (.+)"
            for response in responses:
                matches = re.search(pattern, response, re.DOTALL)
                annotation.append({
                    "Type": re.findall(r'\b\d+\b', matches.group(1)) if matches.group(1) != "None" else "None",
                    "Rationale": matches.group(2),
                    "Rating": re.findall(r'\b\d+\b', matches.group(3))[0],
                    "Rationale For Rating": matches.group(4)
                })
    except ValueError as e: # TODO: bug process when the response does not follow the format
        print(responses)
        raise ValueError(e)
    except AttributeError as e:
        print(responses)
        raise AttributeError(e)
    return annotation


def get_eval(sys_prompt, user_prompt: str, max_tokens: int = 500):
    global inptoks, outtoks
    inptoks = inptoks+0
    for _ in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(**{
                "model": "gpt-4-0613",
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "top_p": 0.6,
                    "presence_penalty": 0,
                    "frequency_penalty": 0
            })
            print(response['usage'])
            inptoks += response['usage']['prompt_tokens']
            outtoks += response['usage']['completion_tokens']
            print("inpts: ", inptoks, '; outts: ', outtoks)
            
            
            content = response["choices"][0]["message"]["content"]
        except Exception as e:
            print(e)
            time.sleep(1)
        else:
            break
    # print(content)
    return content

system_prompt = """Your role is to evaluate text quality based on given criteria.
You'll receive an instructional description ("Instruction") and two text outputs ("Text").
Understand and interpret instructions to evaluate effectively.
Provide annotations for each text with a rating and rationale.
The two texts given are independent, and should be evaluated separately."""


instruction_following_template = """# Instruction Following Assessment

Evaluate alignment between output and intent. Assess understanding of task goal and restrictions.

**Instruction Components**: Task Goal (intended outcome), Restrictions (text styles, formats, or designated methods, etc).

**Scoring**: Rate outputs 1 to 5:
1. **Irrelevant**: No alignment.
2. **Partial Focus**: Addresses one aspect poorly.
3. **Partial Compliance**:
    - (1) Meets goal or restrictions, neglecting other.
    - (2) Acknowledges both but slight deviations.
4. **Almost There**: Near alignment, minor deviations.
5. **Comprehensive Compliance**: Fully aligns, meets all requirements.

## Format:

### Input
Instruction: [Clearly specify the task goal and restrictions]

Texts:
<text 1> [Text 1]
<text 2> [Text 2]

### Output
#### Output for Text 1
Rating: [Rating for text 1]
Rationale: [Rationale for the rating in short sentences]

#### Output for Text 2
Rating: [Rating]
Rationale: [Rationale]

---

## Annotation

### Input
Instruction: {instruction}

Texts:
<text 1> {text_1}
<text 2> {text_2}

### Output
"""

# NOTE,template changed to not output types to reduce chance of format error
helpfulness_template = """# Informativeness / Helpfulness Assessment

Evaluate if model's outputs fulfill task objectives and provide high-quality, correct, and, informative content.

Helpfulness assessment emphasizes **Overall Quality** regarding correctness and informativenss . 

**Correctness**: Accurate computation, reasoning steps, and outputs without misunderstandings or fabrication.

Types of informativeness:
1. **Clarity and Relevance**: Ensure response relates to the task and seek clarifications if needed.
2. **Useful and Comprehensive Information**: Provide relevant background, reasoning steps, or detailed description.
3. **Not Lengthy, No Repetition**: Avoid verbosity or recycling content.

Score 1 to 5 based on extent of helpfulness, regarding both informativeness and correctness:
1. **Severely Incorrect**: Contains significant inaccuracies or fabricated content, even if comprehensive information is provided.
2. **Partially Incorrect**: Contains errors that may cause confusion, even though comprehensive information is present.
3. **Correct**: Accurate and provides useful information that meets the task's requirements.
4. **Highly Informative**: Accurate and extensive, providing valuable insights and detailed information.
5. **Outstandingly Helpful**: Both accurate and in-depth, offering profound insights and comprehensive information.

---

## Format

### Input
Instruction: [Specify task goal and restrictions]

Texts:
<text 1> [Text 1]
<text 2> [Text 2]

### Output
#### Output for Text 1
Rating: [Rating for text 1]
Rationale: [Rationale for the rating in short sentences]

#### Output for Text 2
Rating: [Rating]
Rationale: [Rationale]

---

## Annotation

### Input
Instruction: {instruction}

Texts:
<text 1> {text_1}
<text 2> {text_2}

### Output
"""

TEMPLATE = {
    "instruction_following": instruction_following_template,
    # "honesty": honesty_template,
    # "truthfulness": truthfulness_template,
    # "harmlessness": harmlessness_template,
    "helpfulness": helpfulness_template,
}

SHUFLLE_NUM = 1
def annotate(example):
    # example has "instruction", list of "completion" dictionaries with "response key"
    # HACK removed honesty, truthfulness
    aspects = ["instruction_following", "helpfulness"] #, , "honesty", "truthfulness",]
    completions = [dict({"annotations": {aspect: [] for aspect in aspects}}, **completion)
                    for completion in deepcopy(example["completions"])]

    for aspect in aspects:
        # if subset == "truthful_qa":
        #     world_knowledge = "\n".join(["a subset of correct answers: " + str(example["correct_answers"]), 
        #                                  "a subset of incorrect_answers: " + str(example["incorrect_answers"])])
        # elif subset == "false_qa":
        #     world_knowledge = "The question is based on a false promise."
        # elif subset == "flan":
        #     world_knowledge = example["correct_answers"]
        # else:
        #     world_knowledge = "No additional world knowledge for reference."

        # generate several lists of a random order of 4 completions, no repetition
        count = 0
        random_orders = []
        while True:
            order = list(range(2))
            random.shuffle(order)
            if order not in random_orders:
                random_orders.append(order)
                count += 1
            if count == SHUFLLE_NUM:
                break
        print(random_orders)
        for order in random_orders:        
            format_input = {"instruction": example["instruction"]}
            format_input.update({f"text_{i+1}": example["completions"][o]["response"] for i, o in enumerate(order)})
            # if aspect == "truthfulness":
            #     format_input.update({"world_knowledge": world_knowledge})

            responses = get_eval(system_prompt, user_prompt=TEMPLATE[aspect].format(**format_input))
            for i in range(2):
                try:
                    responses = process(responses, aspect) # gpt-4 format error
                except Exception as e:
                    print("format error")
                    if i < 2:
                        # get gpt-4 input stuff
                        responses = get_eval(system_prompt, user_prompt=TEMPLATE[aspect].format(**format_input))
                        print(responses)
                    else:
                        print(e)
                        break
                else:
                    for j in range(2):
                        completions[j]["annotations"][aspect].append(responses[order.index(j)])
                    break

    example["completions"] = completions

    return example
    

def incorporate_annotation_to_completions(example):
    pass


# if __name__ == "__main__":

    # subsets = ["generalizationtest"]
    
    # for subset in subsets:
    #     # HACK changed up input format
    #     # with open(os.path.join("data", subset + ".jsonl"), "r") as f:
    #     inpath = os.path.join("data", "ultraeval_processed", subset + ".jsonl")
    #     print(inpath)
    #     dataset = pd.read_json(inpath, orient='records', lines=True)
        
    #     # dataset = dataset.map(annotate)
    #     dataset_dict = []
    #     for i, data in tqdm(dataset.iterrows(), total=len(dataset), desc="Annotating"):
    #         dataset_dict.append(annotate(data))

    #     datadf = pd.DataFrame(dataset_dict)
    #     os.makedirs("annotation", exist_ok=True)
    #     result_path = os.path.join("annotation", subset + "_annotated.jsonl")
    #     datadf.to_json(result_path, orient='records', lines=True)
    #     # with open(result_path, "w") as f:
    #     #     json.dump([{k: v for k, v in data.items()} for data in dataset_dict], f, indent=4)