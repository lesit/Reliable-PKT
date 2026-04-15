import torch
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

features = ("Description", "Concepts")
def _parse_response(response):
    lines = [l.strip() for l in response.split('\n') if l.strip()]

    extracted = {feat:"" for feat in features}

    for line in lines:
        for feat in features:
            i_feat = line.find(feat+":")
            if i_feat>=0:
                extracted[feat] = line[i_feat+len(feat)+1:].strip()

    for v in extracted.values():
        if len(v) == 0:
            return None, None

    return extracted.values()

def extract_eckt_features(student_code):
    # return description, concepts, is_retry
    
    student_code = student_code.replace("\\n", "\n")
    """
    학생 코드에서 ECKT 피처(지식 개념 리스트 및 의도 요약)를 추출합니다.
    """
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a CS education expert. Your task is to analyze student code for Knowledge Tracing research. "
                "You must generate a problem description and knowledge concepts for each submission. "
                "Your output must be exactly TWO lines:\n"
                "Line 1: Problem Description\n"
                "Line 2: Knowledge Concepts\n"
                "Do not include any markdown, or conversational filler."
            ),
        },
        {
            "role": "user", 
            "content": (
                "[Code Submission]\n"
                f"{student_code}\n\n"
                
                "[Task]\n"
                "Based on the code submission above, generate the following two items which are Knowledge Concepts and Problem Description. "
                "- Description: \{A one-sentence concise description of the original problem.\}\n"
                "- Concepts: \{3-5 core programming concepts, comma-separated.\}\n\n"
                "Respond in exactly two lines. Example format:\n"
                "Description: Calculate the area of a circle given its radius.\n"
                "Concepts: Arithmetic Operations, Boolean Logic, Math Library"
            )
        }
    ]

    # 채팅 템플릿 적용 및 추론
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    for idx, temp in enumerate([0.1, 0.7]):
        do_sample = idx > 0
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256,
                temperature=temp if do_sample else None, # 일관된 출력을 위해 낮은 온도로 설정
                do_sample=False,
                top_p=0.9 if do_sample else None
            )
        
        # 생성된 텍스트 디코딩
        response = tokenizer.batch_decode(
            [g[len(m):] for m, g in zip(model_inputs.input_ids, generated_ids)],
            skip_special_tokens=True
        )[0]

        description, concepts = _parse_response(response)
        if description is not None:
            return description, concepts, idx>0

    return None, None, True
