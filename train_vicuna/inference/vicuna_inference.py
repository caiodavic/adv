from peft import PeftModel
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother #código para evitar overconfidence no modelo
import sys
import fire
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
PROMPT_TEMPLATE = f"""
A chat between a costumer who needs help related to a company1s products and an agent who is willing to help as best as possible.
The agent gives helpful, detailed and polite answers to the user's questions.

### Instruction: 
The last messages of the conversation were these, you must reply to the last message from the client. Generate only one response being the agent according to the examples after ### Chat.

### Chat:
[CHAT]

### Response
"""

class VicunaGenerator:
    def __init__(self, base_model:str, 
                model_adapted: str="", 
                device_map:str = "auto", 
                temperature:float =0.1,
                top_p: float = 0.8):
        self.base_model = base_model
        self.model_adapted = model_adapted
        self.temperature = temperature
        self.top_p = top_p

        print(f"Importing base model {self.base_model}...")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        )
        print(f"Importing tokenizer from {self.base_model}...")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)
    
        if model_adapted != "":
            print(model_adapted)
            print(f"Importing tuned model {self.model_adapted.split('/')[-1]}...")
            self.model = PeftModel.from_pretrained(self.model,model_adapted)
    
    def create_prompt(self, chat: str) -> str:
        return PROMPT_TEMPLATE.replace('[CHAT]', chat)

    def format_chat(self, chat: list) -> str:
        chat_str = ""
        for i in chat:
            value = i['value'].replace('\n','')
            chat_str+= f"{i['from']}: {value}\n"
        return chat_str
    
    @torch.inference_mode()
    def predict(self, prompt: str) -> str:
        inputs = self.tokenizer([prompt])
        output_ids = self.model.generate(
            torch.as_tensor(inputs.input_ids).cuda(),
            do_sample=True,
            temperature=self.temperature,
            max_new_tokens=150,
            top_p=0.8
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        skip_echo_len = len(prompt)
        outputs = outputs[skip_echo_len:]
        return outputs
    
    def generate_answer(self, chat_list: list) -> str:
        chat = self.format_chat(chat_list)
        prompt = self.create_prompt(chat)
        answer = self.predict(prompt)
        return answer

    