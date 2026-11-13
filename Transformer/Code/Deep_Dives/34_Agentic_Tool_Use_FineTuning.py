"""
34. Agentic Tool Use (Function Calling) Fine-Tuning
Deep Dive Implementation

This file demonstrates how models like GPT-4 or Claude are trained to use tools
(like calculators or web search). It is fundamentally a Supervised Fine-Tuning (SFT)
task where the model learns a specific syntax (e.g., XML tags or JSON) to halt
generation, trigger an external API, and then resume generation once the 
Observation is appended to the context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. THE DATA FORMAT (The Secret to Tool Use)
# ---------------------------------------------------------------------------
# Tool use is entirely a formatting trick. We train the model to output a specific
# string, like `<tool_call>{"name": "calc", "args": "5*5"}</tool_call>`.
# The environment (Python) intercepts this string, runs the function, and appends
# `<observation>25</observation>` back into the prompt for the model to continue.

training_data = [
    {
        "prompt": "User: What is 534 * 89?\nModel:",
        "target_completion": "<tool_call>{\"name\": \"calculator\", \"args\": \"534 * 89\"}</tool_call>\n<observation>47526</observation>\nOutput: The answer is 47526."
    },
    {
        "prompt": "User: Who won the 2024 Super Bowl?\nModel:",
        "target_completion": "<tool_call>{\"name\": \"web_search\", \"args\": \"2024 Super Bowl winner\"}</tool_call>\n<observation>Kansas City Chiefs</observation>\nOutput: The Kansas City Chiefs won the 2024 Super Bowl."
    }
]

# ---------------------------------------------------------------------------
# 2. MOCK MODEL & TOKENIZER
# ---------------------------------------------------------------------------

class MockTokenizer:
    """A dummy tokenizer that simply assigns an ID to every character."""
    def encode(self, text):
        return [ord(c) for c in text]
    def decode(self, tokens):
        return "".join([chr(t) for t in tokens])
    @property
    def vocab_size(self):
        return 256

class ToolUseLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # RNN for simplicity of this toy example (Transformers would use Attention here)
        self.rnn = nn.GRU(embed_dim, 64, batch_first=True) 
        self.lm_head = nn.Linear(64, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        return self.lm_head(out)

# ---------------------------------------------------------------------------
# 3. THE SFT TRAINING LOOP
# ---------------------------------------------------------------------------

def train_tool_use():
    tokenizer = MockTokenizer()
    model = ToolUseLLM(vocab_size=tokenizer.vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("--- 34. Training Agentic Tool Use (SFT) ---")
    print("Teaching the model the <tool_call> syntax...")
    
    epochs = 100
    for epoch in range(epochs):
        total_loss = 0
        for data in training_data:
            # Combine prompt and target
            full_text = data["prompt"] + data["target_completion"]
            tokens = torch.tensor(tokenizer.encode(full_text)).unsqueeze(0)
            
            # Autoregressive shifting (Predict next token)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            
            logits = model(inputs)
            
            # Calculate Cross-Entropy Loss
            loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 25 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss/len(training_data):.4f}")
            
    print("Training Complete. Model has learned the tool syntax.")
    return model, tokenizer

# ---------------------------------------------------------------------------
# 4. INFERENCE ENVIRONMENT (The actual "Agent" loop)
# ---------------------------------------------------------------------------

def execute_tool(tool_string):
    """Parses the model's output and runs the actual Python code."""
    import json
    
    # Very naive parser for demonstration
    start = tool_string.find("<tool_call>") + len("<tool_call>")
    end = tool_string.find("</tool_call>")
    json_str = tool_string[start:end]
    
    try:
        command = json.loads(json_str)
        if command["name"] == "calculator":
            # WARNING: Never use eval() in production. This is just for the toy demo!
            result = str(eval(command["args"])) 
            return f"\n<observation>{result}</observation>\nOutput:"
        elif command["name"] == "web_search":
            return f"\n<observation>Simulated Search Result for {command['args']}</observation>\nOutput:"
    except Exception as e:
        return f"\n<observation>Error: {str(e)}</observation>\nOutput:"

if __name__ == "__main__":
    model, tokenizer = train_tool_use()
    
    print("\n--- Inference: The Agent Loop ---")
    print("In a real system, the inference script watches the generated tokens.")
    print("If it sees '</tool_call>', it pauses generation, executes the Python function,")
    print("appends the <observation> string to the context, and resumes generation.")
