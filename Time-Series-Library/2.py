from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained(
    "SanctumAI/Meta-Llama-3-8B-Instruct-GGUF",
    model_file="meta-llama-3-8b-instruct.Q8_0.gguf",
    model_type="llama",
    gpu_layers=50)

print(llm("AI is going to"))

