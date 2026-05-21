---
title: "Building a Generative Data Agent System: From Data Selection to Deployment"
date: 2025-05-19
categories:
  - project
  - machine learning
tags:
  - data agent
  - LLM
  - fine-tuning
  - deployment
---

This post documents my work on a **Generative Data Agent System** – a system that allows non-expert users to obtain end-to-end data analysis pipelines, mathematical problem abstraction, and predictive models simply by prompting a language model with natural language questions. The project is based on the **DataMind-12K** dataset and the **Qwen3.5-0.8B** model.

---

## Task 1: Literature Review on Data Selection

**Requirement:** Read the survey *"A Survey on Data Selection for Language Models"* (focus on the *Data Selection for Instruction-Tuning and Multitask Training* part). Illustrate three methods that you consider most effective for data selection in the data science and mathematical modeling domain.

### Three Effective Methods

1. **Complexity-based Filtering**  
   - Each trajectory (question + reasoning + code) is assigned a complexity score based on metrics like reasoning steps, code length, or the number of API calls.  
   - High-complexity samples are retained because they force the model to learn deeper reasoning and more sophisticated code generation, which is crucial for data science tasks.  
   - *Why effective:* Data science often involves multi-step reasoning; simple queries are less informative for fine-tuning.

2. **Trajectory Deduplication**  
   - Remove highly similar trajectories by computing sentence embeddings (e.g., using Sentence-BERT) and clustering. Only one sample per cluster is kept.  
   - *Why effective:* Reduces redundancy and ensures diversity in the training data, preventing the model from overfitting to repetitive patterns.

3. **Reward-based Selection**  
   - Use a reward model (e.g., trained on human preferences or task completion success) to score each trajectory. Only trajectories with high rewards are selected.  
   - *Why effective:* Directly optimizes for trajectories that lead to correct and efficient data analysis workflows, aligning with the end goal.

---

## Task 2: Data Processing and Sampling

**Requirement:**  
- Download `datamind_12k.json` from the DataMind-12K repository.  
- Select **2k samples** for training and **500 samples** for validation using the methods from Task 1 (not random sampling).  
- Prepare the data in the format required by Qwen model training.

### Implementation

I wrote a Python script to:

1. Load the full JSON file.
2. Apply complexity-based filtering (using a heuristic based on trajectory length and number of code blocks).
3. Apply trajectory deduplication (using TF-IDF + cosine similarity with a threshold of 0.85).
4. Apply reward-based selection (using a simple heuristic: reward = 1 if the final answer is correct in the trajectory, else 0 – but I also used the GLM-4.7-Flash API to rate trajectories for quality).
5. Select top 2000 for training and next 500 for validation.

```python
# Simplified pseudocode
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("datamind_12k.json") as f:
    data = json.load(f)

# 1. Complexity: keep trajectories with >3 reasoning steps
def complexity(traj):
    return len(traj.get("reasoning_steps", []))

data = [d for d in data if complexity(d) >= 3]

# 2. Deduplication
vectorizer = TfidfVectorizer()
texts = [d["question"] + " " + d["code"] for d in data]
tfidf = vectorizer.fit_transform(texts)
similarities = cosine_similarity(tfidf)
# remove pairs with similarity > 0.85
# ... (omitted for brevity)

# 3. Reward: use API to score
# ... (calling GLM-4.7-Flash)

# Select top 2000 and 500
train_data = data[:2000]
val_data = data[2000:2500]

# Save as Qwen format: each sample has "instruction", "output"
the processed data was saved as data_train.json and data_val.json in the format:
json 
[
  {
    "instruction": "User question...",
    "output": "Assistant's response... (including code and reasoning)"
  }
]
 
 

Note: I used the free API of GLM-4.7-Flash for reward scoring, which helped evaluate trajectory quality without manual labeling.

## Task 3: Model Fine-Tuning with Ray-Train

Requirement:  

    Use Ray-train Python code to fine-tune Qwen3.5-0.8B on the selected data.  
    Since no GPU is available on the local server, use pytorch-cpu for debugging, train for a few hours, and save a checkpoint.  
    Alternatively, use AutoDL or Google Colab for GPU.

Approach

I used Ray Train with the Hugging Face Transformers library. The training script:
```python
 
import ray
from ray.train.huggingface import TransformersTrainer
from ray.air.config import ScalingConfig

# Define training function
def train_func(config):
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    import torch
    
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    
    # Load data
    # ... (tokenization and dataset creation)
    
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=500,
        # ... other args
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    trainer.save_model("./final_checkpoint")

# Run with Ray
scaling_config = ScalingConfig(num_workers=1, use_gpu=False)  # CPU mode
trainer = TransformersTrainer(
    train_loop_per_worker=train_func,
    scaling_config=scaling_config,
)
trainer.fit()
 
 

Since my laptop had no GPU, I ran a small debugging loop on CPU (only 100 steps) to verify correctness, then submitted the full training to Google Colab with a free T4 GPU. The training took approximately 2 hours for 3 epochs.
## Task 4: Deployment and Demo Website

Requirement:  

    Deploy the agent model on your laptop.  
    Prepare a website for the demo (use the official Qwen3.5-0.8B checkpoint if needed).  
    Modify the web UI to better support data analysis tasks (e.g., add a code editor, data upload, result visualization).

Implementation

I used Gradio to build a simple web interface:
python
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

def generate_response(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=1024)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Enter your data analysis question"),
    outputs=gr.Textbox(label="Agent Output"),
    title="Data Agent Demo",
    description="Ask a natural language question about data analysis.",
)
iface.launch()
 
 

To better support data analysis, I extended the UI with:

    A file upload button to allow users to upload CSV datasets.
    A code display panel that shows the generated Python code.
    A results preview (like a table or plot placeholder).

The demo is now running on my laptop at http://localhost:7860.
Conclusion

This project gave me hands-on experience with:

    Data selection strategies for instruction tuning.
    End-to-end pipeline: data processing → training → deployment.
    Using Ray for distributed training (even on CPU).
    Building a web demo with Gradio.

The complete code is available on GitHub. Feel free to reach out if you have any questions!

Last updated: May 2025