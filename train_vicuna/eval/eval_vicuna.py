from dataclasses import dataclass, field
import json
import time
import numpy as np
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother #código para evitar overconfidence no modelo
from .similarity_calculator import SimilaritiesExtractor
from inference.vicuna_inference import VicunaGenerator
import fire
from tqdm import tqdm
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def get_context_conversation(data_eval: list, window_size: str, roles: list) -> tuple:
    context_to_inference = {}
    answer_to_eval = {}
    for i in data_eval:
        id = i['id']
        conv = i['conversations']
        context_conversation = []
        answer_conversation = []
        if window_size == 'all':
            window_size_int = len(conv)-1
        else:
            window_size_int = int(window_size)
        for j in range(window_size_int,len(conv),2):
            context_conversation.append(conv[j-window_size_int:j])
            answer_conversation.append(conv[j])    
        context_to_inference[id] = context_conversation
        answer_to_eval[id] = answer_conversation

    return context_to_inference, answer_to_eval

def get_time_inference(inferences_time:list, window_size: str) -> str:
    inferences_np = np.array(inferences_time)
    mean = np.mean(inferences_np)
    max = np.max(inferences_np)
    min = np.min(inferences_np)

    return f"Inference Time to {window_size}: Mean: {mean} | Max: {max} | Min: {min}"
    
def save_json(data: dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(data, f)

def make_inference(
        data_path: str = 'data',
        base_model: str = 'hellollel/vicuna-7b',
        model_adapted: str = '',
        output_results: str = 'results/',
        device_map:str = "auto",
        roles: str = "['client','agent']",
        steps: str = ['1','3','5','7','all']
):
    
    vicuna_generator = VicunaGenerator(base_model, model_adapted, device_map)

    print(f"Loading data from {data_path}...")
    data_eval = json.load(open(data_path, 'r'))
    roles = eval(roles)  

    convs = {}
    answers = {}
    times = {}
    predicts = {}

    for step in steps:
        print(f"Preparing data to size context {step}")
        convs[step], answers[step] = get_context_conversation(data_eval, step, roles)

        print(f"Starting inference to size context {step}")
        actual_time = []
        convs_predict = {}
        for session_id in tqdm(convs[step].keys()):
            actual_predict = []          
            for chat_list in convs[step][session_id]:
                start_time = time.time()
                predict = vicuna_generator.generate_answer(chat_list)
                end_time = time.time()
                actual_time.append(end_time-start_time)
                actual_predict.append(predict)
            convs_predict[session_id] = actual_predict           
        print(f"Finished inference to size context {step}")
        print(get_time_inference(actual_time, step))
        predicts[step] = convs_predict
        times[step] = actual_time
        
    print(f"Saving results in {output_results}")
    
    save_json(predicts, output_results + 'predicts.json')
    save_json(answers, output_results + 'answers.json')

    print(f"Importing transformer for similarity calculation")
    similarity_calculator = SimilaritiesExtractor()
    similarities = {}

    for step in steps:
        step_similarities = {}
        print(f"Calculating similarities to size context {step}")
        for session_id in tqdm(convs[step].keys()):
            reference_answers = [answer['value'].strip() for answer in answers[step][session_id]]
            predictions_answers = [predict.strip() for predict in predicts[step][session_id]]

            step_similarities[session_id] = similarity_calculator.get_list_similarities(references = reference_answers, predictions = predictions_answers)
        print(f"Finished calculating similarities to size context {step}")
        print(f"Saving similarities in {output_results} to size context {step}")
        similarities[step] = step_similarities
    
    save_json(similarities, output_results + f'similarities_{step}.json')

    
if __name__ == "__main__":
    fire.Fire(make_inference) 
   