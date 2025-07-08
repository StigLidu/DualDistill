## Training Data Preparation

Our training problems are selected from [DeepMath](https://huggingface.co/datasets/zwhe99/DeepMath-103K).

### For the Two Teachers

* **Text-based Reasoning Teacher:** [DeepMath](https://huggingface.co/datasets/zwhe99/DeepMath-103K) provides solution trajectories from Deepseek-R1.
* **Agentic Tool Use Teacher:** Generated from [Openhands](https://github.com/All-Hands-AI/OpenHands) using Claude-3.5-Sonnet as the underlying model.

The distilled trajectory composition data (teacher distillation) is available at [DualDistill](https://huggingface.co/datasets/VanishD/DualDistill).

### For Self-Distillation

The whole training problem list is `whole_training_set.jsonl`.