#%%
!pip install sentencepiece
#%%
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#%%
model_name = "Helsinki-NLP/opus-mt-en-fr"
#%%
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#%%
tokenizer = AutoTokenizer.from_pretrained(model_name)
#%%
source_text = "Hello, how are you?" 
inputs = tokenizer(source_text, return_tensors="pt")
outputs = model.generate(inputs.input_ids)
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translated_text)
