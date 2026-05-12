import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()

def generate_qa_pairs():
    client = openai.OpenAI()
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    # We will just take the first 10k chars of each file to avoid huge contexts
    text1 = ""
    with open(os.path.join(data_dir, "sample_01.md"), "r", encoding="utf-8") as f:
        text1 = f.read()[:10000]
        
    text2 = ""
    with open(os.path.join(data_dir, "sample_02.md"), "r", encoding="utf-8") as f:
        text2 = f.read()[:10000]
        
    prompt = f"""
    Dựa vào các đoạn văn bản sau, hãy tạo ra 20 câu hỏi và câu trả lời (ground truth) bằng tiếng Việt.
    10 câu hỏi cho Văn bản 1 (Báo cáo tài chính).
    10 câu hỏi cho Văn bản 2 (Nghị định bảo vệ dữ liệu cá nhân).
    
    Trả về định dạng JSON là một list các object:
    [
      {{"question": "câu hỏi...", "ground_truth": "câu trả lời..."}},
      ...
    ]
    Chỉ trả về JSON hợp lệ, không có markdown block hay text thừa.
    
    Văn bản 1:
    {text1}
    
    Văn bản 2:
    {text2}
    """
    
    print("Sending request to OpenAI...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    content = response.choices[0].message.content.strip()
    if content.startswith("```json"):
        content = content[7:-3]
    elif content.startswith("```"):
        content = content[3:-3]
        
    qa_list = json.loads(content)
    
    with open(os.path.join(os.path.dirname(__file__), "test_set.json"), "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)
        
    print(f"Generated {len(qa_list)} QA pairs and saved to test_set.json")

if __name__ == "__main__":
    generate_qa_pairs()
