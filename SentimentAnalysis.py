import streamlit as st #creates webapp
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

#classifier
classifier = pipeline("sentiment-analysis")

#responder
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

st.title("Sentiment Analysis & Business Response") #creates webapp
text = st.text_input("Enter a sentence:") #gets text input

if text:
    result = classifier(text)[0]
    st.write(f"**Sentiment:** {result['label']}")
    st.write(f"**Confidence:** {result['score']}")

    prompt = "respond as the business owner to the following review: " + text
    messages = [
        {"role": "user", "content": prompt}
    ]

    response = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt = True,
        enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )

    full_response_decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        index=len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    st.write(f"**Response: **{content}")