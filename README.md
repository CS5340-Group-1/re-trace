# Re-TRACE: HMM-Guided Controllable Generation with Higher-Order and Cloned Variants

Re-TRACE extends the [TRACE](https://github.com/yidouweng/trace) methodology by conducting precise experiments on its foundational backbone: the Hidden Markov Model (HMM). We explore whether increasing the complexity of the underlying probabilistic model can lead to more effective control over Large Language Model (LLM) generation.

## Overview

This project investigates two primary variants of the HMM architecture for controllable text generation:
- **Second-Order HMMs (SOHMMs):** Capturing deeper temporal dependencies in state transitions.
- **Cloned HMMs (CHMMs):** Leveraging state cloning to represent complex, non-Markovian patterns in the latent space.

Our experiments focus on documenting improvements across two key metrics:
1.  **Model Toxicity:** Reducing the likelihood of generating harmful or biased content.
2.  **Fluency:** Ensuring that controlled generation maintains the linguistic quality and coherence of the base model.

## Acknowledgments

This codebase is largely inspired by and builds upon the following research and implementations:
- **Training Logic:** Adapted from [Ctrl-G](https://github.com/joshuacnf/Ctrl-G) (Conditional Generation with Hidden Markov Models).
- **Text Generation:** Based on the [TRACE](https://github.com/yidouweng/trace) (Tractable Control for Autoregressive Generation) framework.

## Project Structure

- `src/re_trace/ctrlg/`: Core logic for training and distillation of HMM variants.
- `src/re_trace/trace/`: Integration with LLMs for guided generation using the trained HMMs.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
