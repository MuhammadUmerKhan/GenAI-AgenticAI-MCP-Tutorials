# DSPy Framework: An In-Depth Guide

Welcome to the ultimate guide to **DSPy**! DSPy is a declarative framework that shifts the paradigm of interacting with language models from *prompt engineering* to *programming*.

Instead of hacking together brittle string prompts, you define tasks as structured signatures and let DSPy figure out the most optimized prompt for the specific model you're using.

---

## 1. The Core Philosophy: "Program, don't prompt"

When building complex applications with LLMs, creating prompts by hand is tedious and hard to maintain. A prompt that works perfectly on `gpt-4o` might fail completely on a smaller model or a model from another provider (like Claude 3.5 Sonnet). 

DSPy solves this by breaking down an LLM application into modular, optimizable components:
- **Signatures**: Specify *what* the task is (Inputs & Outputs).
- **Modules**: Specify *how* to solve the task (e.g., Chain of Thought, ReAct loop).
- **Tools**: Normal Python functions the LLM can use to interact with the world.
- **Metrics & Optimizers**: Automatically tune and rewrite your prompts (or fine-tune weights) to maximize performance.

---

## 2. Installation and Setup

First, install the `dspy` library:

```bash
pip install dspy
```

DSPy uses **LiteLLM** under the hood, meaning it supports almost every major model provider (OpenAI, Anthropic, Google, OpenRouter, etc.). 

Here is how you connect to a language model and set it globally:

```python
import dspy

# Initialize an LM using the "provider/model" syntax
# This allows you to easily swap out models without changing application logic
lm = dspy.LM("openai/gpt-4o-mini", api_key="YOUR_OPENAI_API_KEY")

# Configure DSPy to use this language model globally for all subsequent operations
dspy.configure(lm=lm)
```

---

## 3. Signatures: Defining "What" You Want

A **Signature** defines the inputs and outputs of a task. It does not dictate how the LLM should process the data.

### 3.1 String-Based Signatures
The easiest way to define a signature is with a shorthand string `inputs -> outputs`.

```python
import dspy

# Define a signature that takes a 'subject' as input and returns a 'haiku' as output
haiku_signature = "subject -> haiku"

# Initialize the dspy.Predict module, the foundational block that executes signatures
haiku_generator = dspy.Predict(haiku_signature)

# Run the generator with our input arguments
result = haiku_generator(subject="computer science")

# The output object will have properties matching the outputs defined in the signature
print(result.haiku)
```

You can pass multiple inputs, return multiple outputs, and even enforce **Types**:

```python
# A typed signature: expects multiple inputs including a boolean flag, and returns a list of strings
haiku_bot = dspy.Predict("location, mood, contains_pun: bool -> haikus: list[str]")

# Execute the predictor with the strongly-typed inputs
result = haiku_bot(location="a quiet library", mood="mysterious", contains_pun=True)

# Access the generated list
print(result.haikus[0])
```

### 3.2 Class-Based Signatures
For more complex tasks, you can define your signature as a class. This gives you the ability to add detailed **docstrings** (used as system instructions) and **field descriptions**.

```python
from typing import Literal

class HaikuBot(dspy.Signature):
    """Write a classical haiku given the provided inputs."""
    
    # InputField with 'desc' gives the LLM explicit instructions on what this parameter means
    location: str = dspy.InputField(desc="The setting of the poem")
    mood: str = dspy.InputField()
    
    # You can restrict inputs using type constraints like Literal
    season: Literal["spring", "summer", "autumn", "winter"] = dspy.InputField()
    
    # OutputField declares what the LLM should generate
    haiku: str = dspy.OutputField()

# Using the class-based signature to instantiate a Predictor
bot = dspy.Predict(HaikuBot)
res = bot(location="Bodega Bay", mood="mysterious", season="autumn")
```

---

## 4. Modules: Defining "How" To Do It

If Signatures define the *what*, **Modules** define the *how*. 

- `dspy.Predict`: The foundational module. It directly asks the LLM to complete the signature.
- `dspy.ChainOfThought`: Instructs the LLM to think step-by-step before producing the output. This usually yields significantly better reasoning!
- `dspy.ReAct`: Enables the LLM to use a loop of reasoning and acting (via tools) until the task is complete.

### Composing Custom Modules
Just like PyTorch models, you can compose DSPy modules to build multi-step pipelines. 

```python
class HaikuEnsemble(dspy.Module):
    def __init__(self, num_candidates: int = 3):
        super().__init__()
        self.num_candidates = num_candidates
        
        # Module 1: Predictor that drafts multiple candidate haikus
        self.writer = dspy.Predict("location, season -> haikus: list[str]")
        
        # Module 2: ChainOfThought Judge to evaluate candidates and pick the best one
        # ChainOfThought inherently adds a 'reasoning' field to the output
        self.judge = dspy.ChainOfThought("location, season, candidates: list[str] -> best_index: int")

    def forward(self, location: str, season: str):
        # 1. Draft multiple candidates using the writer module
        candidates = self.writer(location=location, season=season).haikus
        
        # 2. Pick the best candidate using the judge module
        verdict = self.judge(location=location, season=season, candidates=candidates)
        
        # Return a custom Prediction object containing the final haiku and the judge's reasoning
        return dspy.Prediction(
            haiku=candidates[verdict.best_index],
            reasoning=verdict.reasoning
        )

# Run the composite module
ensemble = HaikuEnsemble()
result = ensemble(location="Tokyo", season="spring")

# The reasoning shows the LLM's thought process for why it picked that specific haiku
print("Reasoning:\n", result.reasoning)
print("\nSelected Haiku:\n", result.haiku)
```

---

## 5. Tools & Agents

You can turn your program into an agent by giving it **Tools** (ordinary Python functions) and using the `dspy.ReAct` module. DSPy automatically reads your tool's name, parameters, and docstring to teach the LLM how to use it!

```python
import wikipedia
import dspy

# Define a tool with descriptive type hints and a docstring
def wikipedia_search(query: str) -> list[str]:
    """Search Wikipedia for the given query and return a list of page titles."""
    return wikipedia.search(query)

def get_wikipedia_page(title: str) -> str:
    """Get the content of a Wikipedia page given its title."""
    return wikipedia.page(title).content

# Agent creation: give it a signature and a list of tools it can use
agent = dspy.ReAct(
    "location, topic -> poem", 
    tools=[wikipedia_search, get_wikipedia_page], 
    max_iters=5 # Limit the number of reasoning/acting loops
)

# The agent will search Wikipedia, read pages, and then write the poem!
result = agent(location="San Francisco", topic="Golden Gate Bridge history")
```

---

## 6. Datasets & Evaluation Metrics

To optimize a prompt, DSPy needs two things: Example Data and a Metric.

### Defining Examples
DSPy uses `dspy.Example` to handle data. You use `.with_inputs()` to specify which fields represent inputs to the program (the rest are treated as expected labels/targets).

```python
# Create training examples. The 'location' and 'season' are inputs,
# and if we had a 'haiku' field, it would be treated as the target label.
examples = [
    dspy.Example(location="Paris", season="spring").with_inputs("location", "season"),
    dspy.Example(location="New York", season="winter").with_inputs("location", "season")
]
```

### Defining a Metric
A metric is a function that scores a prediction. It returns a float `[0.0, 1.0]`, or a `dspy.Prediction` with a score and a feedback string.

```python
# The GEPA optimizer requires the metric to accept exactly five arguments
def haiku_score(example, prediction, trace=None, pred_name=None, pred_trace=None) -> float:
    text = prediction.haiku.lower()
    
    # Penalize (score = 0.0) if the haiku mentions the season word explicitly
    if example.season.lower() in text:
        return 0.0
        
    # Reward (score = 1.0) if it successfully implies the season without using the word
    return 1.0
```

---

## 7. Optimizers (e.g. GEPA)

This is DSPy's superpower. The Optimizer takes your unoptimized program, your training examples, and your metric, and iteratively **rewrites the prompts** until it maximizes the score.

```python
# 1. Choose an optimizer
# GEPA (Generate, Evaluate, Propose, Accept) uses an LLM to reflect and rewrite prompts
optimizer = dspy.GEPA(
    metric=haiku_score,
    reflection_lm=dspy.LM("openai/gpt-4o"), # It's best to use a smart model as a teacher here
    auto="light"
)

# 2. Compile! 
# This is where DSPy simulates runs, evaluates the metric, and rewrites the underlying prompt instructions.
optimized_bot = optimizer.compile(bot, trainset=examples)

# 3. Save your optimized prompt program for future use
optimized_bot.save("optimized_haiku_bot.json")
```

After compilation, even a small model like `gpt-4o-mini` can often outperform a base `gpt-4o` because DSPy discovered the perfect prompt syntax and edge-cases specific to your task!

---
*Happy Programming (not Prompting)!*
