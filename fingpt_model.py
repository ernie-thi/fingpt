import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Überprüfe, ob CUDA verfügbar ist
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Geht mit cuda, man muss also nix extra installiern :D ")
else:
    device = torch.device("cpu")
    print("Geht net mit Cuda.. Hoffentlich is dei CPU besser wie meine :D")

# Pfad zum Modell (musst anpassen)
model_path = "C:\\Users\\rapni\\Github\\python_playarounds\\finGPT\\content\\finetuned_model"

# Basis Config
config = AutoConfig.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)

# Initialisierung von Basis-Modell
model = AutoModelForCausalLM.from_config(config)

# LoRA-Konfiguration
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=True,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query_key_value"],
    bias="none",
)

# Wende LoRA-Anpassungen an
model = get_peft_model(model, lora_config)

# Verschiebe das Modell auf GPU wenns geht 
model = model.to(device)

# Lade die trainierten Gewichte
adapter_weights = torch.load(f"{model_path}")
model.load_state_dict(adapter_weights, strict=False)

# Modell in den eval mode
model.eval()

# Hier kommt Frage rein: r
input_text = "Wie ist die Coca Cola Aktie bewertet? Ist sie ein Kauf wert?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# max_new tokens == wieviele tokens lang die antwort sein darf
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

# Dekodiere die Vorhersage in einen lesbaren Text
predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(predicted_text)
