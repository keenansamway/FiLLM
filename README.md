# FiLLM (Forgetting in Large Language Models)

This is the working repository for my thesis [Towards Localized Methods of Forgetting Knowledge in Language Models](https://drive.google.com/file/d/1arPF_SS2Ur12GK6pEfk6eANzSUeVPS6I/view?usp=sharing) for the MSc in Computational Statistics and Machine Learning at UCL during the 2023/2024 academic year.


## Abstract
> As large language models (LLMs) become more widespread in their use, the need for effective ma- chine unlearning methods has grown in response to privacy regulations and ethical considerations. In this thesis, we critically examine what we term the finetune-then-unlearn paradigm, the setting many researchers uses to evaluate unlearning methods, where models are finetuned on additional data before undergoing unlearning to forget a portion of that.
We introduce a novel approach called Second-Order Influence Ranking (SOIR), which allows us to rank model parameters based on their influence on knowledge retention and forgetting.1 This method enables localized unlearning, targeting specific parameters that are most impactful for balancing forgetfulness and retention. Our experiments reveal that finetuning induces non-uniform parameter shifts, particularly in the multi-layer perceptron (MLP) layers, and that certain layers are more robust to unlearning than others. Additionally, we show that forget performance degrades uniformly across model layers, while retain performance varies significantly, emphasizing the need to focus on retention. We evaluate the impact of gradient compression on SOIR, demonstrating that it retains ranking accuracy while improving computational efficiency.
Finally, we pose a re-framing of the localized unlearning problem in the finetune-then-unlearning setting: instead of focusing solely on maximizing forgetting, we argue that targeted unlearning might be thought of as a retention-maximization problem, aiming to identify parts of the model which best preserve essential knowledge while removing unwanted information. This highlights the need for future research to develop more robust evaluation frameworks that can more accurately assess the effectiveness of machine unlearning methods in LLMs.


## Installation
```bash
conda create -n fillm python=3.10
codna activate fillm
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Experiments
Scripts for the experiments can be found in the `src/srcipts/experiments` folder.

Notebooks to visualize the results can be found in the `src/notebooks` folder.


## Acknowledgement
Our implementation is based on following repo:
* [https://github.com/locuslab/tofu](https://github.com/locuslab/tofu)