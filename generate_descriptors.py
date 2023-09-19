import os
import openai
import json
import tqdm

from transformers import AutoTokenizer, LlamaForCausalLM
import transformers
import torch

import itertools

from descriptor_strings import stringtolist

openai.api_key = "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"


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

def generate_prompt_abstract(prompt: str, category_name: str):
    return prompt.format(category_name=category_name)

# generator 
def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))
        
def obtain_descriptors_and_save(savename, class_list, prompt,answer_length, top_k, batch_size, use_generate, use_pipeline, use_llama2=True):
    responses = {}
    descriptors = {}
    
    prompts = [generate_prompt_abstract(prompt,category.replace('_', ' ')) for category in class_list]
    
    # most efficient way is to partition all prompts into the max size that can be concurrently queried from the OpenAI API
    if use_llama2:
        if use_pipeline:
            model= "meta-llama/Llama-2-7b-chat-hf"
            tokenizer = AutoTokenizer.from_pretrained(model)
            print("Cuda available: ", torch.cuda.is_available())
            pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                torch_dtype=torch.float16,
                device_map="auto",
                #device = 1,
                do_sample=True,
                top_k=top_k,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens = answer_length # max_length=len(prompt.split())+answer_length
            )
            responses = []
            for i in tqdm.tqdm(range(0, len(prompts), batch_size)):
                batch = prompts[i:i+batch_size]
                batch_responses = pipeline(batch)
                responses.extend(batch_responses)
            responses = [item for sublist in responses for item in sublist]
            response_texts = [resp["generated_text"][len(prompts[i]):] for i,resp in enumerate(responses)]
        if use_generate:
            model_name= "meta-llama/Llama-2-7b-chat-hf"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("Cuda available: ", torch.cuda.is_available())
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                device_map = "auto",
            )


            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to('cuda')
            # Generate
            generate_ids = model.generate(inputs.input_ids, max_length=len(inputs.input_ids[0])+answer_length, top_k=top_k, num_return_sequences=1, do_sample=True)

            decoded_responses = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            response_texts = [resp[len(prompt):] for resp in decoded_responses.split('\n-')]

    else:
        responses = [openai.Completion.create(model="text-davinci-003",
                                                prompt=prompt_partition,
                                                temperature=0.,
                                                max_tokens=100,
                                                ) for prompt_partition in partition(prompts, 20)]
        response_texts = [r["text"] for resp in responses for r in resp['choices']]
    descriptors_list = [stringtolist("\n-"+response_text) for response_text in response_texts]
    descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    descriptors_metadata = {'prompt': prompt, 'answer_length': answer_length, 'top_k': top_k}

    # save descriptors to json file
    if not savename.endswith('.json'):
        savename += '.json'
    with open(savename, 'w') as fp:
        json.dump(descriptors, fp, indent=4)
     
    with open(os.path.join("descriptors_meta_info", os.path.split(savename)[1]).replace('.json', '_metadata.json'), 'w') as fp:
        json.dump(descriptors_metadata, fp, indent=4)
    
    whole_output = {cat: output for cat, output in zip(class_list, response_texts)}
    with open(os.path.join("descriptors_meta_info", os.path.split(savename)[1]).replace('.json', '_whole_outputs.json'), 'w') as fp:
        json.dump(whole_output, fp, indent=4)
    

#read the cub classes from /export/home/ru86qer/datasets/cub_unmodified/CUB_200_2011
def read_classes():
    classes = []
    with open('/export/scratch/ru86qer/datasets/stanford_cars/classes.txt', 'r') as f:
        #classes = [line.strip().split(' ')[1].split('.')[1] for line in f.readlines()]
        classes = [line.strip() for line in f.readlines()]
    return classes

classes = read_classes()

path = '/export/home/ru86qer/classify_by_description_release/cars_descriptors'

answer_length = 100    #auch optional problemlos als Liste umschreibbar mit individualisierten Werten pro Prompt

top_k = 1             #auch optional problemlos als Liste umschreibbar mit individualisierten Werten pro Prompt

batch_size = 20

prompt_index = 0

with open('prompts_cars/prompt'+"_{0}.txt".format(prompt_index), 'r') as f:
    prompt = f.read()


for i in range(0, 10000):
    savename = os.path.join(path, f'descriptors_cars_prompt{prompt_index}_run_{i}.json')
    if not os.path.exists(savename):
        break

obtain_descriptors_and_save(os.path.join(path,savename), classes, prompt, answer_length, top_k, batch_size, use_generate=False, use_pipeline=True, use_llama2=True)


