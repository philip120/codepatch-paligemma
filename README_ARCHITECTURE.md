# CodePatch: A Text-Text Multimodal Architecture for Code Comprehension

This document outlines the architecture of our novel "CodePatch" model, a multimodal system designed to understand and describe MATLAB code without ever seeing the visual output.

## Acknowledgements and Core Components

Our work stands on the shoulders of giants. This project would not be possible without leveraging these fantastic open-source repositories:

-   **Base PaliGemma Architecture**: The overall structure of our model, including the implementation of the Gemma language model and the multi-modal projector, is based on the work in Hari P. Paudel's `hkproj/pytorch-paligemma` repository.
    -   **Source**: [https://github.com/hkproj/pytorch-paligemma](https://github.com/hkproj/pytorch-paligemma)

-   **MATLAB AST Parser**: For our advanced semantic patching strategy, we use the pure Python MATLAB parser created by `jol-jol`. This was a critical component for moving beyond simple token chunking.
    -   **Source**: [https://github.com/jol-jol/pymatlabparser](https://github.com/jol-jol/pymatlabparser)

## The Core Innovation: Replacing Vision with Code

The foundational idea of this project is to adapt a vision-language model for a code-language task. We treat snippets of code as if they were patches of an image.

To achieve this, we made one central modification to the original PaliGemma architecture:

-   **The SigLIP vision encoder was completely removed.**
-   **It was replaced with a `CodeBERT` model (`microsoft/codebert-base`)**, which acts as our "Code Encoder."

The goal is to teach the model to "see" the semantics of code in the same way the original model was taught to see the content of an image.

### The Patching Mechanism: Summarizing Code for the LLM

To efficiently process long source code files, we don't feed the raw code to the Gemma LLM. Instead, we use a patching strategy to create a short, high-level summary. We have explored two methods for this:

#### 1. Initial Approach: Fixed-Size Token Patches

Our first implementation was a simple and direct approach:

1.  The input MATLAB code is tokenized.
2.  The sequence of tokens is split into arbitrary, fixed-size chunks (e.g., 20 tokens each).
3.  Each chunk becomes a "patch."

While effective for summarization, this method lacks semantic awareness, as it can split logical constructs (like a function call) across multiple patches.

#### 2. Advanced Approach: AST-Based Semantic Patches

Our current, more sophisticated approach uses an Abstract Syntax Tree (AST) to create semantically meaningful patches.

1.  The MATLAB code string is first parsed into a full AST using `pymatlabparser`.
2.  We then traverse this tree and extract nodes that represent complete thoughts or actions (e.g., `statement` nodes).
3.  Each of these nodes is rebuilt into a clean, human-readable line of code (e.g., `plot(t, y)` or `A = -2`).
4.  This list of rebuilt statements becomes our sequence of "semantic patches."

This method is far superior as it provides the model with a list of complete, logical operations, which is a much higher-quality summary of the code's intent.

### Model Data Flow

The end-to-end data flow for training is as follows:

1.  A MATLAB script is parsed into a sequence of **semantic patches** using the AST.
2.  Each patch string is passed to the frozen **CodeBERT encoder** to produce a summary vector.
3.  The sequence of summary vectors is passed through a trainable **Multi-Modal Projector**, which aligns the vectors with Gemma's embedding space.
4.  These "code-word" embeddings are prepended to the text prompt embeddings and fed into the frozen **Gemma LLM**.
5.  The LLM generates a textual description, and the loss is calculated.
6.  Critically, backpropagation **only** updates the weights of the Multi-Modal Projector and the patch position embeddings, leaving the expert CodeBERT and Gemma models untouched. 