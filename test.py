import time
import torch
from transformers import AutoTokenizer
from modeling_llama import LlamaForCausalLM
from decoding import exact_self_speculative_generate

th_stop_draft_essg_autoth  = 0.6

tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-chat-hf')
model = LlamaForCausalLM.from_pretrained('NousResearch/Llama-2-7b-chat-hf',
    torch_dtype=torch.bfloat16,
    load_in_8bit=True
)

prompt = '<s>[INST] Tell me something interesting about the sun and the moon. [/INST]'
input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
print(tokenizer.decode(input_ids[0]), end='\n\n')

start_time = time.time()
result = exact_self_speculative_generate(
    model, tokenizer, input_ids,
    max_tokens=2000, max_step_draft=8,
    th_stop_draft=th_stop_draft_essg_autoth,
    early_stop=True,
    auto_th_stop_draft=True,
    auto_parameters=[1,0.5,0.9,1e-2,0.9],
    do_sample=False, do_sample_draft=False,
    top_k=1, top_p=1, temperature=0
)
cnt_tokens = len(result['generate_ids'][0])
# print(result['generate_ids'])
# print(result['matchness'])
time_delta = time.time() - start_time
print('e2e speed:', time_delta, cnt_tokens, cnt_tokens / time_delta)
