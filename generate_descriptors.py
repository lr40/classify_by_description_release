import os
import openai
import json
import tqdm

from transformers import AutoTokenizer
import transformers
import torch

import itertools

from descriptor_strings import stringtolist

openai.api_key = None #FILL IN YOUR OWN HERE


def generate_prompt(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a television in a photo?
A: There are several useful visual features to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful features for distinguishing a {category_name} in a photo?
A: There are several useful visual features to tell there is a {category_name} in a photo:
-
"""

# generator 
def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))
        
def obtain_descriptors_and_save(filename, class_list, use_llama2=True):
    responses = {}
    descriptors = {}
    
    prompts = [generate_prompt(category.replace('_', ' ')) for category in class_list]
    
    # most efficient way is to partition all prompts into the max size that can be concurrently queried from the OpenAI API
    if use_llama2:
        model = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model)
        print("Loading model...")
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=400,
        )
        responses = [pipeline(prompt_partition) for prompt_partition in tqdm.tqdm(partition(prompts, 20))]
        responses = [item for sublist in responses for item in sublist]
        response_texts = [resp[0]["generated_text"][len(prompts[i]):] for i,resp in enumerate(responses)]
    else:
        responses = [openai.Completion.create(model="text-davinci-003",
                                                prompt=prompt_partition,
                                                temperature=0.,
                                                max_tokens=100,
                                                ) for prompt_partition in partition(prompts, 20)]
        response_texts = [r["text"] for resp in responses for r in resp['choices']]
    descriptors_list = [stringtolist(response_text) for response_text in response_texts]
    descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    # save descriptors to json file
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(descriptors, fp)
    

#read the cub classes from /export/home/ru86qer/datasets/cub_unmodified/CUB_200_2011
def read_cub_classes():
    cub_classes = []
    with open('/export/home/ru86qer/datasets/cub_unmodified/CUB_200_2011/classes.txt', 'r') as f:
        cub_classes = [line.strip().split(' ')[1].split('.')[1] for line in f.readlines()]
    return cub_classes

cub_classes = read_cub_classes()
obtain_descriptors_and_save('llama2_cub', cub_classes, use_llama2=True)