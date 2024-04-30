# Training a multimodal projector for vision, between SigLIP and Phi-3-mini.

The approach will be to freeze the weights of the encoder and decoder models,
and use the MLP class from this [release](https://huggingface.co/qresearch/llama-3-vision-alpha/).

The data that can be used includes:
- [Moondream dataset](https://huggingface.co/datasets/vikhyatk/lnqa) - 303k
- [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)

I'm hoping this will be enough. This is my first semi-serious project of this size.