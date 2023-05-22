import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import json
import torch
import click
from pathlib import Path
from typing import List
import openai

def request(
    prompt: str,
    engine='davinci',
    max_tokens=60,
    temperature=1.0,
    top_p=1.0,
    n=1,
    stop='\n',
    presence_penalty=0.0,
    frequency_penalty=0.0,
    ):
    # retry request (handles connection errors, timeouts, and overloaded API)
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            break
        except Exception as e:
            tqdm.write(str(e))
            tqdm.write("Retrying...")
            import time
            time.sleep(60)
    
    generations = [gen['text'].lstrip() for gen in response['choices']]
    generations = [_ for _ in generations if _ != '']
    return generations

def prompt_format(prompt_path: str, keywords: List[str], query: str):
    with open(prompt_path) as f:
        context_string = f.read().strip('\n')
    if keywords is not None:
        n = np.random.choice(range(1, len(keywords)+1))      # number of keywords
        keywords = random.sample(keywords, n)                # subset of keywords
        context_string = context_string.replace('{keywords}', ', '.join(keywords))
    if query is not None:
        context_string = context_string.replace('{question}', query)
    return context_string


@click.command()
@click.option('--task', type=str, default='obqa')
@click.option('--input_path', type=str, default='data/obqa/train.obqa.json')
@click.option('--output_path', type=str, default='data/obqa/knowledge/knowledge_gpt3.train.obqa.json')
@click.option('--prompt_path', type=str, default='prompt/obqa_prompt.txt')
@click.option('--num_knowledge', type=int, default=10)
@click.option('--top_p', default=0.5, type=float)
@click.option('--temperature', default=0.7, type=float)
@click.option('--max_tokens', default=60, type=int)
@click.option('--n', default=None, type=int)
def main(
    task: str,
    input_path: str,
    output_path: str,
    prompt_path: str,
    num_knowledge: int,
    top_p: float,
    temperature: float,
    max_tokens: int,
    n: int,
):
    # read examples for inference
    eval_df = pd.read_json(input_path)

    # generate knowledge!
    generated_examples = []

    for i, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        context_string = prompt_format(
            prompt_path,
            keywords=None,
            query=row['query'])
        knowledges = request(
            context_string,
            n=num_knowledge,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens)

        row['knowledges'] = list(set(knowledges))
        generated_examples.append(row.to_dict())

    with open(output_path, 'w') as fo:
        json.dump(generated_examples, fo, indent=4)


if __name__ == '__main__':
    main()
