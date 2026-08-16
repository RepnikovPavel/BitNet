arXiv:2504.12285v2 [cs.CL] 25 Apr 2025

BitNet b1.58 2B4T Technical Report

Shuming Ma* Hongyu Wang* Shaohan Huang Xingxing Zhang
Ying Hu Ting Song Yan Xia Furu Wei

https://aka.ms/GeneralAI

Abstract

We introduce BitNet b1.58 2B4T, the first open-source, native 1-bit Large Language Model (LLM) at the 2-billion parameter scale. Trained on a corpus of 4 trillion tokens, the model has been rigorously evaluated across benchmarks covering language understanding, mathematical reasoning, coding proficiency, and conversational ability. Our results demonstrate that BitNet b1.58 2B4T achieves performance on par with leading open-weight, full-precision LLMs of similar size, while offering significant advantages in computational efficiency, including substantially reduced memory footprint, energy consumption, and decoding latency. To facilitate further research and adoption, the model weights are released via Hugging Face along with open-source inference implementations for both GPU and CPU architectures.

![](images/task_0d497f856e4e_page_0_pic_0.png)

![](images/task_0d497f856e4e_page_0_pic_1.png)

Figure 1: BitNet b1.58 2B4T advances the Pareto frontier defined by leading open-weight LLMs under 3B parameters in terms of performance versus memory, demonstrating superior efficiency.

* Equal contribution. ◊ Corresponding author. S. Ma, S. Huang, X. Zhang, T. Song, Y. Xia and F. Wei are with Microsoft Research. H. Wang is with University of Chinese Academy of Sciences. Y. Hu is with Tsinghua University.

# 1 Introduction

Open-source large language models (LLMs) have become pivotal in democratizing access to advanced AI capabilities, fostering innovation, and enabling research across diverse fields such as natural language processing, code generation, and vision computing (Dubey et al., 2024; Yang et al., 2024; Bai et al., 2025). Their public availability allows for widespread experimentation and adaptation. However, a significant barrier hinders their broader adoption: the substantial computational resources required for deployment and inference. State-of-the-art open LLMs typically require large memory footprints, consume considerable energy, and exhibit notable inference latency, rendering them impractical for many edge devices, resource-constrained environments, and real-time applications.

1-bit LLMs, representing an extreme yet promising form of model quantization where weights and potentially activations are constrained to binary {-1, +1} or ternary {-1, 0, +1}, offer a compelling solution to the efficiency challenges. By drastically reducing the memory required to store weights and enabling highly efficient bitwise computations, they have the potential to significantly lower deployment costs, reduce energy consumption, and accelerate inference speeds. While prior work has explored 1-bit models, existing open efforts often fall into two categories: 1) post-training quantization (PTQ) methods applied to pre-trained full-precision models, which can lead to significant performance degradation (Xu et al., 2024b; Team, 2024), or 2) native 1-bit models (trained from scratch with 1-bit weights) that have been developed at relatively smaller scales (e.g., OLMo-Bitnet-1B²) and may not yet match the capabilities of larger, full-precision counterparts. This performance gap has limited the practical impact of 1-bit LLMs thus far.

To bridge this gap between efficiency and performance, we introduce BitNet b1.58 2B4T, the first open-source, native 1-bit LLM trained at scale. This model, comprising 2 billion parameters, was trained from scratch on a substantial dataset of 4 trillion tokens, leveraging architectural and training innovations specific to the 1-bit paradigm. **The core contribution of this work is to demonstrate that a native 1-bit LLM, when trained effectively at scale, can achieve performance comparable to leading open-weight, full-precision models of similar size across a wide range of tasks.**

This technical report details the development and evaluation of BitNet b1.58 2B4T. We describe the architecture and training methodology, and then present comprehensive evaluation results on standard benchmarks assessing language understanding, mathematical reasoning, coding proficiency, and multi-turn conversational abilities. Our findings confirm its strong performance relative to established full-precision baselines, coupled with significant advantages in efficiency. Finally, we announce the public release of the BitNet b1.58 2B4T model weights via Hugging Face and provide open-source inference code optimized for both GPU and CPU execution, aiming to facilitate further research and the practical deployment of highly efficient LLMs.

# 2 Architecture

The architecture of BitNet b1.58 2B4T is derived from the standard Transformer model (Vaswani et al., 2017), incorporating significant modifications based on the BitNet framework (Wang et al., 2023a; Ma et al., 2024). The model is trained entirely from scratch.

The core architectural innovation lies in replacing the standard full-precision linear layers (torch.nn.Linear) with custom BitLinear layers. This constitutes the foundation of the BitNet approach. Within these BitLinear layers:

* **Weight Quantization:** Model weights are quantized to 1.58 bits during the forward pass. This is achieved using an absolute mean (absmean) quantization scheme, which maps weights to ternary values $\{-1, 0, +1\}$. This drastically reduces the model size and enables efficient mathematical operations.
* **Activation Quantization:** Activations flowing through the linear projection are quantized to 8-bit integers. This employs an absolute maximum (absmax) quantization strategy, applied per-token.
* **Normalization:** We incorporate subln normalization (Wang et al., 2022) to further enhance training stability, which can be particularly beneficial in quantized training regimes.

²https://huggingface.co/NousResearch/OLMo-Bitnet-1B

Beyond the *BitLinear* layers, several established LLM techniques are integrated to enhance performance and stability:

* **Activation Function (FFN):** Within the feed-forward network (FFN) sub-layers, instead of the commonly used SwiGLU activation (Shazeer, 2020), BitNet b1.58 2B4T employs squared ReLU (ReLU²). This choice is motivated by its potential to improve model sparsity and computational characteristics within the 1-bit context (Wang et al., 2024b,a).
* **Positional Embeddings:** Rotary Position Embeddings (RoPE) (Su et al., 2024) are used to inject positional information, a standard practice in modern high-performance LLMs.
* **Bias Removal:** Consistent with architectures like LLaMA, all bias terms are removed from the linear layers and normalization layers throughout the network, reducing parameter count and potentially simplifying quantization.

For tokenization, we adopt the tokenizer developed for LLaMA 3 (Dubey et al., 2024). This tokenizer implements a byte-level Byte-Pair Encoding (BPE) scheme with a vocabulary size of 128,256 tokens. This choice ensures robust handling of diverse text and code, and its widespread adoption facilitates straightforward integration with existing open-source tooling and ecosystems.

## 3 Training

The training process for BitNet b1.58 2B4T involved three distinct phases: large-scale pre-training followed by supervised fine-tuning (SFT) and direct preference optimization (DPO). While advanced techniques like Proximal Policy Optimization (PPO) or Group Relative Policy Optimization (GRPO) can further enhance capabilities such as mathematics and chain-of-thought reasoning (Schulman et al., 2017; Shao et al., 2024), the current version of BitNet b1.58 2B4T relies solely on pre-training, SFT, and DPO. The exploration of reinforcement learning methods remains a direction for future work.

### 3.1 Pre-training

The pre-training phase aimed to imbue the model with broad world knowledge and foundational language capabilities. We adapted general training strategies from established LLM practices (Dubey et al., 2024), with specific adjustments tailored for the 1-bit architecture.

#### 3.1.1 Learning Rate Schedule

A two-stage learning rate schedule was employed.

1. **Stage 1 (High Learning Rate):** The initial phase utilized a standard cosine decay schedule but commenced with a relatively high peak learning rate. This decision was informed by the observation that 1-bit models often exhibit greater training stability compared to their full-precision counterparts, allowing for more aggressive initial learning steps.
2. **Stage 2 (Cooldown):** Approximately midway through the planned training token count, the learning rate was abruptly decayed and subsequently maintained via a cosine schedule with a significantly lower peak value. This "cooldown" phase allows the model to refine its representations on higher-quality data (see Section 3.1.3).

#### 3.1.2 Weight Decay Schedule

Complementing the learning rate adjustments, a two-stage weight decay strategy was implemented.

1. **Stage 1:** During the first training stage, weight decay followed a cosine schedule, reaching a peak value of 0.1. This regularization helps prevent overfitting during the initial high-learning-rate phase.
2. **Stage 2:** In the second stage, weight decay was effectively disabled (set to zero). This allows the model parameters to settle into finer-grained optima guided by the lower learning rate and curated data.

3.1.3 Pre-training Data

The pre-training corpus comprised a mixture of publicly available text and code datasets, including large web crawls like DCLM (Li et al., 2024b) and educational web pages like FineWeb-EDU (Penedo et al., 2024). To enhance mathematical reasoning abilities, we also incorporated synthetically generated mathematical data. The data presentation strategy aligned with the two-stage training: the bulk of general web data was processed during Stage 1, while higher-quality curated datasets were emphasized during the Stage 2 cooldown phase, coinciding with the reduced learning rate.

3.2 Supervised Fine-tuning (SFT)

Following pre-training, the model underwent supervised fine-tuning (SFT) to enhance its instruction-
following capabilities and improve its performance in conversational interaction formats.

3.2.1 SFT Data

The SFT phase utilized a diverse collection of publicly available instruction-following and conversa-
tional datasets. These included, but were not limited to, WildChat (Zhao et al., 2024), LMSYS-Chat-
1M (Zheng et al., 2024), WizardLM Evol-Instruct (Xu et al., 2024a), and SlimOrca (Lian et al., 2023).
To further bolster specific capabilities, particularly in reasoning and complex instruction adherence,
we supplemented these with synthetic datasets generated using methodologies like GLAN (Li et al.,
2024a) and MathScale (Tang et al., 2024).

3.2.2 Chat Template

For conversational tasks during SFT and inference, the following chat template structure was employed:

<|begin_of_text|>System: {system_message}<|eot_id|>
User: {user_message_1}<|eot_id|>
Assistant: {assistant_message_1}<|eot_id|>
User: {user_message_2}<|eot_id|>
Assistant: {assistant_message_2}<|eot_id|>...

3.2.3 Optimization Details

Several optimization choices were key during SFT:

*   **Loss Aggregation:** Instead of averaging the cross-entropy loss across tokens within a batch (mean reduction), we employed summation. Empirically, we observed that summing the losses led to improved convergence and better final performance for this model.
*   **Hyperparameter Tuning:** Careful tuning of the learning rate and the number of training epochs was performed. Consistent with our pre-training findings, the 1-bit model benefited from a relatively larger learning rate during SFT compared to typical full-precision model fine-tuning. Furthermore, achieving optimal convergence required extending the fine-tuning duration over a larger number of epochs than full-precision models of similar size.

3.3 Direct Preference Optimization (DPO)

To further align the model's behavior with human preferences regarding helpfulness and safety, we applied Direct Preference Optimization (DPO) (Rafailov et al., 2023) following the SFT phase. DPO offers an efficient alternative to traditional RLHF by directly optimizing the language model using preference data, thereby circumventing the need to train a separate reward model. This DPO stage served to refine the model's conversational prowess and overall alignment with desired interaction patterns in practical use cases.

3.3.1 Training Data

The preference dataset used for DPO training was constructed from a combination of publicly
available resources recognized for capturing diverse human judgments on model outputs. Specifically,

we utilized UltraFeedback (Cui et al., 2024) and MagPie (Xu et al., 2024c). The aggregation of these
datasets provided a robust and multifaceted preference signal, guiding the model towards generating
responses more aligned with human expectations.

3.3.2 Training Details

The DPO training phase was conducted for 2 epochs. We employed a learning rate of $2 \times 10^{-7}$ and set the DPO beta parameter, which controls the divergence from the reference policy, to 0.1. To enhance training efficiency during this phase, we integrated optimized kernels from the *Liger Kernel* library (Hsu et al., 2024). Qualitatively, our observations indicate that the DPO process effectively steered the model towards preferred response styles without inducing significant degradation in the core capabilities established during pre-training and SFT.

4 Evaluation

We measure performance on a wide variety of benchmarks classified as follows:

*   **Language understanding and reasoning:** ARC-Easy (Yadav et al., 2019), ARC-Challenge (Yadav et al., 2019), HellaSwag (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2020), PIQA (Bisk et al., 2019), OpenbookQA (Mihaylov et al., 2018), and CommonsenseQA (Talmor et al., 2019)
*   **World knowledge:** TruthfulQA (Lin et al., 2022) and MMLU (Hendrycks et al., 2021a)
*   **Reading comprehension:** TriviaQA (Joshi et al., 2017) and BoolQ (Clark et al., 2019)
*   **Math and code:** GSM8K (Cobbe et al., 2021), MATH-500 (Hendrycks et al., 2021b) and HumanEval+ (Liu et al., 2023)
*   **Instruction following and conversation:** IFEval (Zhou et al., 2023) and MT-bench (Zheng et al., 2023)

We compare BitNet b1.58 2B4T with leading open-weight full precision LLMs of similar size, including LLaMA 3.2 1B (Dubey et al., 2024), Gemma-3 1B (Team et al., 2025), Qwen2.5 1.5B (Yang et al., 2024), SmolLM2 1.7B (Allal et al., 2025) and MiniCPM 2B (Hu et al., 2024). All models are instruction-tuned versions. We re-run all benchmarks with a public evaluation pipeline for a fair comparison. More evaluation details are available at the appendix. The main results are presented in Table 1.

4.1 Main Results

As shown in Table 1, BitNet b1.58 2B4T demonstrates remarkable resource efficiency. Its non-embedding memory footprint and estimated energy consumption (Horowitz, 2014; Zhang et al., 2022) during decoding are substantially lower compared to all the full-precision models evaluated, highlighting a significant advantage in operational cost and deployability on resource-constrained devices.

In terms of task performance, BitNet b1.58 2B4T proves highly competitive. It achieves the best results among the compared models on several benchmarks spanning reasoning, knowledge, and math capabilities. On other benchmarks, its performance is closely comparable to the top-performing full-precision models. While some full-precision models show slight advantages on specific tasks or the overall average, BitNet b1.58 2B4T delivers strong performance across the board. The results indicate that BitNet b1.58 2B4T achieves capabilities nearly on par with leading models in its size class while offering dramatically improved efficiency.

4.2 Comparison with Post-training Quantized Models

We further investigate the efficiency-performance trade-off by comparing BitNet b1.58 2B4T against post-training quantized (PTQ) versions of a leading competitor, Qwen2.5 1.5B, using standard INT4 methods (GPTQ and AWQ). The results are summarized in Table 2.

While INT4 quantization successfully reduces the memory footprint of the full-precision model, BitNet b1.58 2B4T achieves an even lower memory requirement due to its native 1-bit architecture.

<table><thead><tr><th>Benchmark (Metric)</th><th>LLaMA 3.2<br>1B</th><th>Gemma-3<br>1B</th><th>Qwen2.5<br>1.5B</th><th>SmolLM2<br>1.7B</th><th>MiniCPM<br>2B</th><th>BitNet b1.58<br>2B</th></tr></thead><tbody><tr><td>Memory<br>(Non-emb)</td><td>2GB</td><td>1.4GB</td><td>2.6GB</td><td>3.2GB</td><td>4.8GB</td><td><strong>0.4GB</strong></td></tr><tr><td>Latency<br>(CPU; TPOT)</td><td>48ms</td><td>41ms</td><td>65ms</td><td>67ms</td><td>124ms</td><td><strong>29ms</strong></td></tr><tr><td>Energy<br>(Estimated)</td><td>0.258J</td><td>0.186J</td><td>0.347J</td><td>0.425J</td><td>0.649J</td><td><strong>0.028J</strong></td></tr><tr><td>Training Tokens<br>(Pre-training)</td><td>9T<br>(pruning &amp; distillation)</td><td>2T<br>(distillation)</td><td>18T</td><td>11T</td><td>1.1T</td><td>4T</td></tr><tr><td>ARC-Challange<br>(0-shot; Acc.norm)</td><td>37.80</td><td>38.40</td><td>46.67</td><td>43.52</td><td>44.80</td><td><strong>49.91</strong></td></tr><tr><td>ARC-Easy<br>(0-shot; Acc.norm)</td><td>63.17</td><td>63.13</td><td><strong>76.01</strong></td><td>62.92</td><td>72.14</td><td>74.79</td></tr><tr><td>OpenbookQA<br>(0-shot; Acc.norm)</td><td>34.80</td><td>38.80</td><td>40.80</td><td><strong>46.00</strong></td><td>40.20</td><td>41.60</td></tr><tr><td>BoolQ<br>(0-shot; Acc)</td><td>64.65</td><td>74.22</td><td>78.04</td><td>75.78</td><td><strong>80.67</strong></td><td>80.18</td></tr><tr><td>HellaSwag<br>(0-shot; Acc.norm)</td><td>60.80</td><td>57.69</td><td>68.28</td><td><strong>71.71</strong></td><td>70.81</td><td>68.44</td></tr><tr><td>PIQA<br>(0-shot; Acc.norm)</td><td>74.21</td><td>71.93</td><td>76.12</td><td>76.12</td><td>76.66</td><td><strong>77.09</strong></td></tr><tr><td>WinoGrande<br>(0-shot; Acc)</td><td>59.51</td><td>58.48</td><td>62.83</td><td>68.98</td><td>61.80</td><td><strong>71.90</strong></td></tr><tr><td>CommonsenseQA<br>(10-shot; Acc)</td><td>58.48</td><td>42.10</td><td><strong>76.41</strong></td><td>63.55</td><td>71.74</td><td>71.58</td></tr><tr><td>TruthfulQA<br>(10-shot; MC2)</td><td>43.80</td><td>38.66</td><td><strong>46.67</strong></td><td>39.90</td><td>41.41</td><td>45.31</td></tr><tr><td>TriviaQA<br>(5-shot; EM)</td><td>37.60</td><td>23.49</td><td>38.37</td><td><strong>45.97</strong></td><td>34.13</td><td>33.57</td></tr><tr><td>MMLU<br>(5-shot; Acc)</td><td>45.58</td><td>39.91</td><td><strong>60.25</strong></td><td>49.24</td><td>51.82</td><td>53.17</td></tr><tr><td>HumanEval+<br>(0-shot; Pass@1)</td><td>31.10</td><td>37.20</td><td><strong>50.60</strong></td><td>28.00</td><td>43.90</td><td>38.40</td></tr><tr><td>GSM8K<br>(4-shot; EM)</td><td>38.21</td><td>31.16</td><td>56.79</td><td>45.11</td><td>4.40</td><td><strong>58.38</strong></td></tr><tr><td>MATH-500<br>(0-shot; EM)</td><td>23.00</td><td>42.00</td><td><strong>53.00</strong></td><td>17.60</td><td>14.80</td><td>43.40</td></tr><tr><td>IFEval<br>(0-shot; Instruct-Strict)</td><td>62.71</td><td><strong>66.67</strong></td><td>50.12</td><td>57.91</td><td>36.81</td><td>53.48</td></tr><tr><td>MT-bench<br>(0-shot; Average)</td><td>5.43</td><td>6.40</td><td>6.12</td><td>5.50</td><td><strong>6.57</strong></td><td>5.85</td></tr><tr><td>Average</td><td>44.90</td><td>43.74</td><td><strong>55.23</strong></td><td>48.70</td><td>42.05</td><td>54.19</td></tr></tbody></table>

Table 1: Comparison of BitNet b1.58 2B4T with leading open-weight full-precision LLMs of similar size (1B-2B parameters) on efficiency metrics and performance across a wide range of benchmarks. All models compared are instruction-tuned versions.

More importantly, this superior memory efficiency does not compromise performance relative to the quantized models. Standard PTQ techniques lead to a noticeable degradation in performance compared to the original full-precision model. In contrast, BitNet b1.58 2B4T maintains stronger overall performance than the INT4 quantized versions of Qwen2.5-1.5B on the evaluated benchmarks. This comparison suggests that BitNet b1.58 2B4T represents a more favorable point on the efficiency-performance curve than applying conventional INT4 PTQ to existing architectures, offering better performance with lower resource usage.

<table><thead><tr><th rowspan="2">Benchmark (Metric)</th><th colspan="3">Qwen2.5</th><th>BitNet b1.58</th></tr><tr><th>1.5B-bf16</th><th>1.5B-GPTQ-int4</th><th>1.5B-AWQ-int4</th><th>2B</th></tr></thead><tbody><tr><td>Memory<br>(Non-emb)</td><td>2.6GB</td><td>0.7GB</td><td>0.7GB</td><td>0.4GB</td></tr><tr><td>Activation</td><td>bf16</td><td>bf16</td><td>bf16</td><td>int8</td></tr><tr><td>MMLU<br>(5-shot; Acc)</td><td><strong>60.25</strong></td><td>58.06</td><td>57.43</td><td>53.17</td></tr><tr><td>GSM8K<br>(4-shot; EM)</td><td>56.79</td><td>50.57</td><td>50.64</td><td><strong>58.38</strong></td></tr><tr><td>IFEval<br>(0-shot; Instruct-Strict)</td><td>50.12</td><td>47.84</td><td>45.44</td><td><strong>53.48</strong></td></tr><tr><td>Average</td><td><strong>55.72</strong></td><td>52.15</td><td>51.17</td><td>55.01</td></tr></tbody></table>

Table 2: Comparison of BitNet b1.58 (2B) against Qwen2.5 1.5B in its original bf16 precision and after INT4 post-training quantization (GPTQ and AWQ). All models shown are based on instruction-tuned checkpoints.

<table><thead><tr><th>Benchmark (Metric)</th><th>Bonsai 0.5B</th><th>OLMo-Bitnet 1B</th><th>Falcon3-1.58bit 7B</th><th>Llama3-8B-1.58 8B</th><th>BitNet b1.58 2B</th></tr></thead><tbody><tr><td>Native 1-bit</td><td>✓</td><td>✓</td><td>✗</td><td>✗</td><td>✓</td></tr><tr><td>ARC-Challange<br>(0-shot; Acc, norm)</td><td>33.19</td><td>26.54</td><td>37.80</td><td>43.69</td><td><strong>49.91</strong></td></tr><tr><td>ARC-Easy<br>(0-shot; Acc, norm)</td><td>58.25</td><td>25.38</td><td>65.03</td><td>70.71</td><td><strong>74.79</strong></td></tr><tr><td>OpenbookQA<br>(0-shot; Acc, norm)</td><td>33.60</td><td>28.20</td><td>38.20</td><td>37.20</td><td><strong>41.60</strong></td></tr><tr><td>BoolQ<br>(0-shot; Acc)</td><td>58.44</td><td>52.48</td><td>72.14</td><td>68.38</td><td><strong>80.18</strong></td></tr><tr><td>HellaSwag<br>(0-shot; Acc, norm)</td><td>48.01</td><td>25.88</td><td>59.46</td><td><strong>68.56</strong></td><td>68.44</td></tr><tr><td>PIQA<br>(0-shot; Acc, norm)</td><td>70.02</td><td>50.49</td><td>72.36</td><td>75.30</td><td><strong>77.09</strong></td></tr><tr><td>WinoGrande<br>(0-shot; Acc)</td><td>54.46</td><td>51.54</td><td>60.14</td><td>60.93</td><td><strong>71.90</strong></td></tr><tr><td>CommonsenseQA<br>(10-shot; Acc)</td><td>18.43</td><td>19.49</td><td>67.08</td><td>28.50</td><td><strong>71.58</strong></td></tr><tr><td>TruthfulQA<br>(10-shot; MC2)</td><td>40.65</td><td><strong>49.05</strong></td><td>43.29</td><td>39.13</td><td>45.31</td></tr><tr><td>TriviaQA<br>(5-shot; EM)</td><td>10.84</td><td>0.00</td><td>0.00</td><td>19.82</td><td><strong>33.57</strong></td></tr><tr><td>MMLU<br>(5-shot; Acc)</td><td>25.74</td><td>25.47</td><td>42.79</td><td>35.04</td><td><strong>53.17</strong></td></tr><tr><td>Average</td><td>41.06</td><td>32.22</td><td>50.76</td><td>49.75</td><td><strong>60.68</strong></td></tr></tbody></table>

Table 3: Performance comparison of BitNet b1.58 2B4T against other open-weight 1-bit models. This includes natively trained 1-bit models (Bonsai-0.5B, OLMo-Bitnet-1B) and larger models post-training quantized to 1.58-bit (Falcon3-1.58bit-7B, Llama3-8B-1.58).

4.3 Comparison with Open-weight 1-bit Models

Finally, we situate BitNet b1.58 2B4T within the landscape of other models designed for or quantized to near 1-bit precision. We compare it against natively trained 1-bit models of smaller scale and significantly larger models post-training quantized to 1.58-bit precision. The comparative results are presented in Table 3.

The evaluation clearly positions BitNet b1.58 2B4T as the leading model in this category. It demonstrates significantly stronger overall performance than all other compared 1-bit models, achieving

the highest scores on the vast majority of benchmarks. Notably, BitNet b1.58 2B4T substantially outperforms not only the smaller, natively trained 1-bit models but also the much larger models (in terms of parameter count) that were quantized to 1-bit. This highlights the effectiveness of the native training approach employed by BitNet b1.58 2B4T, allowing it to set a new state-of-the-art performance level for models operating at this extreme level of quantization, even surpassing larger models subjected to post-training quantization.

# 5 Inference Implementation

Efficient inference is crucial for deploying Large Language Models, particularly for resource-constrained environments. The unique quantization scheme of BitNet b1.58 2B4T, employing 1.58-bit weights and 8-bit activations (W1.58A8), necessitates specialized implementations, as standard deep learning libraries often lack optimized kernels for such mixed-precision, low-bit formats. To address this, we developed and open-sourced dedicated inference libraries for both GPU and CPU platforms. The code is publicly available at https://aka.ms/bitnet.

## 5.1 GPU Inference

Current GPU architectures and their associated software libraries (e.g., cuBLAS, PyTorch kernels) are primarily optimized for operations involving standard data types like FP16, BF16, and INT8/INT4. Native, high-performance support for the specific W1.58A8 matrix multiplication required by BitNet b1.58 2B4T is generally unavailable. This limitation can hinder the realization of the theoretical efficiency gains offered by 1-bit models on existing hardware.

To enable efficient GPU inference, we developed a custom CUDA kernel specifically designed for the W1.58A8 matrix multiplication. Since ternary weights ({-1, 0, +1}, representing 1.58 bits) cannot be stored efficiently using standard data types, we pack multiple weight values into a single 8-bit integer ('int8') for storage in High Bandwidth Memory (HBM). Specifically, four ternary values are encoded into one 'int8' value. During computation, the CUDA kernel loads the packed 'int8' weights from HBM into the GPU's faster on-chip Shared Memory (SRAM). It then unpacks these values back into a representation suitable for efficient ternary computation (e.g., reconstructing the -1, 0, +1 values) immediately before performing the matrix multiplication with the 8-bit activations. This 'pack-store-load-unpack-compute' strategy minimizes memory bandwidth usage while leveraging custom compute instructions. Further implementation details and optimization strategies are elaborated in the Ladder framework (Wang et al., 2023b).

While our custom kernel significantly improves performance compared to naive implementations, we note that current commodity GPU architectures are not optimally designed for the 1-bit models. We believe that future hardware innovations, potentially incorporating dedicated logic for low-bit operations, will be essential to fully unlock the performance and energy efficiency potential of models like BitNet b1.58.

## 5.2 CPU Inference

To ensure broad accessibility and enable deployment on devices lacking powerful GPUs (e.g., edge devices, laptops, standard servers), we developed bitnet.cpp. This C++ library serves as an official reference implementation for CPU inference of 1-bit LLMs, including BitNet b1.58.

bitnet.cpp provides optimized kernels tailored for efficient execution on standard CPU architectures. The kernels are designed to operate efficiently with the model's specific quantization scheme, avoiding the overhead of generic quantization libraries or intricate low-level bit manipulation where possible. It processes the weight elements in a manner consistent with the BitNet b1.58 training methodology, ensuring numerical accuracy (lossless inference relative to the training procedure).

This approach delivers fast and accurate inference of 1.58-bit models directly on CPUs. More technical details and usage instructions can be found in the bitnet.cpp repository and associated technical report (Wang et al., 2025).

6 Conclusion

This technical report introduced BitNet b1.58 2B4T, a significant step towards highly efficient yet capable Large Language Models. As the first open-source, native 1-bit LLM trained at the 2-billion parameter scale on 4 trillion tokens, our work demonstrates the viability of extreme quantization directly within the training process.

Comprehensive evaluations across benchmarks assessing language understanding, reasoning, mathematics, coding, and dialogue revealed that BitNet b1.58 2B4T achieves performance comparable to state-of-the-art open-weight, full-precision models of similar size. Crucially, this performance parity is achieved with dramatically reduced computational requirements, offering substantial savings in memory footprint, energy consumption, and inference latency. To facilitate practical use and further research, we developed and released optimized inference implementations for both GPU (via custom CUDA kernels) and CPU (via the 'bitnet.cpp' library), alongside the model weights available on Hugging Face.

BitNet b1.58 2B4T represents a compelling proof-of-concept that challenges the necessity of full-precision weights for achieving high performance in LLMs at scale. It opens avenues for deploying powerful language models in resource-constrained environments where previous models were prohibitive, potentially democratizing access to advanced AI capabilities.

7 Future Directions

While BitNet b1.58 2B4T demonstrates promising results, several exciting research directions remain:

* **Scaling Laws and Larger Models:** Investigating the scaling properties of native 1-bit LLMs is crucial. Future work will explore training larger models (e.g., 7B, 13B parameters and beyond) and training on even larger datasets to understand if the performance parity with full-precision models holds.
* **Hardware Co-Design and Optimization:** The full potential of 1-bit models is likely hindered by current hardware limitations. Continued development of highly optimized kernels for existing hardware (GPUs, CPUs, NPUs) is needed. Furthermore, co-designing future hardware accelerators specifically optimized for 1-bit computations and data movement could unlock orders-of-magnitude improvements in speed and energy efficiency.
* **Extended Sequence Length:** Extending the maximum sequence length of BitNet b1.58 2B4T can process is crucial. This enhancement is vital for tasks demanding long-context understanding, such as summarizing lengthy documents or engaging in complex problem-solving, and is particularly critical for improving performance on **long chain-of-thought reasoning** tasks. Investigating efficient attention mechanisms suitable for low-bit models at longer sequence lengths will be key.
* **Multilingual Capabilities:** The current model is primarily trained on English-centric data. Extending the pre-training corpus and potentially adapting the architecture to effectively support **multiple languages** is a key direction for broader applicability.
* **Multimodal Integration:** Exploring the integration of 1-bit principles into **multimodal architectures** is another promising frontier. Developing efficient ways to process and fuse information from different modalities (e.g., text and images) within a low-bit framework could enable new applications.
* **Theoretical Understanding:** Delving deeper into the theoretical underpinnings of why 1-bit training at scale is effective remains an open area. Analyzing the learning dynamics, loss landscapes, and representational properties of these models could yield valuable insights for future development.

By pursuing these directions, we aim to further advance the capability and efficiency of 1-bit LLMs, paving the way for more sustainable and accessible artificial intelligence. The open-source release of BitNet b1.58 2B4T and its associated tools provides a foundation for the community to build upon these efforts.

References

Allal, L. B., Lozhkov, A., Bakouch, E., Blázquez, G. M., Penedo, G., Tunstall, L., Marafioti, A., Kydlíček, H., Lajarín, A. P., Srivastav, V., Lochner, J., Fahlgren, C., Nguyen, X.-S., Fourrier, C., Burtenshaw, B., Larcher, H., Zhao, H., Zakka, C., Morlon, M., Raffel, C., von Werra, L., and Wolf, T. (2025). Smollm2: When smol goes big - data-centric training of a small language model. CoRR, abs/2502.02737.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M.-H., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., and Lin, J. (2025). Qwen2.5-vl technical report. CoRR, abs/2502.13923.

Bisk, Y., Zellers, R., Bras, R. L., Gao, J., and Choi, Y. (2019). PIQA: reasoning about physical commonsense in natural language. CoRR, abs/1911.11641.

Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., and Toutanova, K. (2019). Boolq: Exploring the surprising difficulty of natural yes/no questions. CoRR, abs/1905.10044.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. (2021). Training verifiers to solve math word problems. CoRR, abs/2110.14168.

Cui, G., Yuan, L., Ding, N., Yao, G., He, B., Zhu, W., Ni, Y., Xie, G., Xie, R., Lin, Y., Liu, Z., and Sun, M. (2024). ULTRAFEEDBACK: boosting language models with scaled AI feedback. In ICML. OpenReview.net.

Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., Goyal, A., Hartshorn, A., Yang, A., Mitra, A., Sravankumar, A., Korenev, A., Hinsvark, A., Rao, A., Zhang, A., Rodriguez, A., Gregerson, A., Spataru, A., Rozière, B., Biron, B., Tang, B., Chern, B., Caucheteux, C., Nayak, C., Bi, C., Marra, C., McConnell, C., Keller, C., Touret, C., Wu, C., Wong, C., Ferrer, C. C., Nikolaidis, C., Allonsius, D., Song, D., Pintz, D., Livshits, D., Esiobu, D., Choudhary, D., Mahajan, D., Garcia-Olano, D., Perino, D., Hupkes, D., Lakomkin, E., AlBadawy, E., Lobanova, E., Dinan, E., Smith, E. M., Radenovic, F., Zhang, F., Synnaeve, G., Lee, G., Anderson, G. L., Nail, G., Mialon, G., Pang, G., Cucurell, G., Nguyen, H., Korevaar, H., Xu, H., Touvron, H., Zarov, I., Ibarra, I. A., Kloumann, I. M., Misra, I., Evtimov, I., Copet, J., Lee, J., Geffert, J., Vranes, J., Park, J., Mahadeokar, J., Shah, J., van der Linde, J., Billock, J., Hong, J., Lee, J., Fu, J., Chi, J., Huang, J., Liu, J., Wang, J., Yu, J., Bitton, J., Spisak, J., Park, J., Rocca, J., Johnstun, J., Saxe, J., Jia, J., Alwala, K. V., Upasani, K., Plawiak, K., Li, K., Heafield, K., Stone, K., and et al. (2024). The llama 3 herd of models. CoRR, abs/2407.21783.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. (2021a). Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. (2021b). Measuring mathematical problem solving with the MATH dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual.

Horowitz, M. (2014). 1.1 computing's energy problem (and what we can do about it). In 2014 IEEE International Conference on Solid-State Circuits Conference, ISSCC 2014, Digest of Technical Papers, San Francisco, CA, USA, February 9-13, 2014, pages 10-14.

Hsu, P.-L., Dai, Y., Kothapalli, V., Song, Q., Tang, S., Zhu, S., Shimizu, S., Sahni, S., Ning, H., and Chen, Y. (2024). Liger kernel: Efficient triton kernels for LLM training. CoRR, abs/2410.10989.

Hu, S., Tu, Y., Han, X., He, C., Cui, G., Long, X., Zheng, Z., Fang, Y., Huang, Y., Zhao, W., Zhang, X., Thai, Z. L., Zhang, K., Wang, C., Yao, Y., Zhao, C., Zhou, J., Cai, J., Zhai, Z., Ding, N., Jia, C., Zeng, G., Li, D., Liu, Z., and Sun, M. (2024). Minicpm: Unveiling the potential of small language models with scalable training strategies. CoRR, abs/2404.06395.

Joshi, M., Choi, E., Weld, D. S., and Zettlemoyer, L. (2017). Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers*, pages 1601–1611.

Li, H., Dong, Q., Tang, Z., Wang, C., Zhang, X., Huang, H., Huang, S., Huang, X., Huang, Z., Zhang, D., Gu, Y., Cheng, X., Wang, X., Chen, S.-Q., Dong, L., Lu, W., Sui, Z., Wang, B., Lam, W., and Wei, F. (2024a). Synthetic data (almost) from scratch: Generalized instruction tuning for language models. *CoRR*, abs/2402.13064.

Li, J., Fang, A., Smyrnis, G., Ivgi, M., Jordan, M., Gadre, S. Y., Bansal, H., Guha, E., Keh, S. S., Arora, K., Garg, S., Xin, R., Muennighoff, N., Heckel, R., Mercat, J., Chen, M. F., Gururangan, S., Wortsman, M., Albalak, A., Bitton, Y., Nezhurina, M., Abbas, A., Hsieh, C.-Y., Ghosh, D., Gardner, J., Kilian, M., Zhang, H., Shao, R., Pratt, S. M., Sanyal, S., Ilharco, G., Daras, G., Marathe, K., Gokaslan, A., Zhang, J., Chandu, K. R., Nguyen, T., Vasiljevic, I., Kakade, S. M., Song, S., Sanghavi, S., Faghri, F., Oh, S., Zettlemoyer, L., Lo, K., El-Nouby, A., Pouransari, H., Toshev, A., Wang, S., Groeneveld, D., Soldaini, L., Koh, P. W., Jitsev, J., Kollar, T., Dimakis, A., Carmon, Y., Dave, A., Schmidt, L., and Shankar, V. (2024b). Datacomp-lm: In search of the next generation of training sets for language models. In *Globersons, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J. M., and Zhang, C.*, editors, *Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.*

Lian, W., Wang, G., Goodson, B., Pentland, E., Cook, A., Vong, C., and "Teknium" (2023). Slimorca: An open dataset of gpt-4 augmented flan reasoning traces, with verification.

Lin, S., Hilton, J., and Evans, O. (2022). Truthfulqa: Measuring how models mimic human falsehoods. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022*, pages 3214–3252.

Liu, J., Xia, C. S., Wang, Y., and Zhang, L. (2023). Is your code generated by chatgpt really correct? Rigorous evaluation of large language models for code generation. *Advances in Neural Information Processing Systems*, 36:21558–21572.

Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S., Dong, L., Wang, R., Xue, J., and Wei, F. (2024). The era of 1-bit llms: All large language models are in 1.58 bits. *CoRR*, abs/2402.17764.

Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A. (2018). Can a suit of armor conduct electricity? A new dataset for open book question answering. *CoRR*, abs/1809.02789.

Penedo, G., Kydlícek, H., Allal, L. B., Lozhkov, A., Mitchell, M., Raffel, C. A., von Werra, L., and Wolf, T. (2024). The fineweb datasets: Decanting the web for the finest text data at scale. In *Globersons, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J. M., and Zhang, C.*, editors, *Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.*

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. In *Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S.*, editors, *Advances in Neural Information Processing Systems 36*.

Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y. (2020). WinoGrande: an adversarial winograd schema challenge at scale. In *AAAI*, pages 8732–8740.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). Proximal policy optimization algorithms. *CoRR*, abs/1707.06347.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M., Li, Y. K., Wu, Y., and Guo, D. (2024). Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *CoRR*, abs/2402.03300.

Shazeer, N. (2020). GLU variants improve transformer. *CoRR*, abs/2002.05202.

Su, J., Ahmed, M. H. M., Lu, Y., Pan, S., Bo, W., and Liu, Y. (2024). Roformer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568:127063.

Talmor, A., Herzig, J., Lourie, N., and Berant, J. (2019). Commonsensea: A question answering challenge targeting commonsense knowledge. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers)*, pages 4149–4158.

Tang, Z., Zhang, X., Wang, B., and Wei, F. (2024). Mathscale: Scaling instruction tuning for mathematical reasoning. In *Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024*. OpenReview.net.

Team, F.-L. (2024). The falcon 3 family of open models.

Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Ram'e, A., Rivi'ere, M., et al. (2025). Gemma 3 technical report. *arXiv preprint arXiv:2503.19786*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention is all you need. In Guyon, I., von Luxburg, U., Bengio, S., Wallach, H. M., Fergus, R., Vishwanathan, S. V. N., and Garnett, R., editors, *Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA*, pages 5998–6008.

Wang, H., Ma, S., Dong, L., Huang, S., Wang, H., Ma, L., Yang, F., Wang, R., Wu, Y., and Wei, F. (2023a). Bitnet: Scaling 1-bit transformers for large language models. *CoRR*, abs/2310.11453.

Wang, H., Ma, S., Huang, S., Dong, L., Wang, W., Peng, Z., Wu, Y., Bajaj, P., Singhal, S., Benhaim, A., Patra, B., Liu, Z., Chaudhary, V., Song, X., and Wei, F. (2022). Foundation transformers. *CoRR*.

Wang, H., Ma, S., Wang, R., and Wei, F. (2024a). Q-sparse: All large language models can be fully sparsely-activated. *CoRR*, abs/2407.10969.

Wang, H., Ma, S., and Wei, F. (2024b). Bitnet a4.8: 4-bit activations for 1-bit llms. *CoRR*, abs/2411.04965.

Wang, J., Zhou, H., Song, T., Cao, S., Xia, Y., Cao, T., Wei, J., Ma, S., Wang, H., and Wei, F. (2025). Bitnet.cpp: Efficient edge inference for ternary llms. *CoRR*, abs/2502.11880.

Wang, L., Ma, L., Cao, S., Zheng, N., Zhang, Q., Xue, J., Miao, Z., Cao, T., and Yang, Y. (2023b). Ladder: Efficient tensor compilation on customized data format. In *OSDI*.

Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J., Tao, C., Lin, Q., and Jiang, D. (2024a). Wizardlm: Empowering large pre-trained language models to follow complex instructions. In *The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024*. OpenReview.net.

Xu, Y., Han, X., Yang, Z., Wang, S., Zhu, Q., Liu, Z., Liu, W., and Che, W. (2024b). Onebit: Towards extremely low-bit large language models. In Globersons, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J. M., and Zhang, C., editors, *Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024*.

Xu, Z., Jiang, F., Niu, L., Deng, Y., Poovendran, R., Choi, Y., and Lin, B. Y. (2024c). Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. *CoRR*, abs/2406.08464.

Yadav, V., Bethard, S., and Surdeanu, M. (2019). Quick and (not so) dirty: Unsupervised selection of justification sentences for multi-hop question answering. In *EMNLP-IJCNLP*.

Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., and Qiu, Z. (2024). Qwen2.5 technical report. *CoRR*, abs/2412.15115.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. (2019). HellaSwag: can a machine really finish your sentence? In *Proceedings of the 57th Conference of the Association for Computational Linguistics*, pages 4791–4800.

Zhang, Y., Zhang, Z., and Lew, L. (2022). PokeBNN: A binary pursuit of lightweight accuracy. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 12465–12475. IEEE.

Zhao, W., Ren, X., Hessel, J., Cardie, C., Choi, Y., and Deng, Y. (2024). Wildchat: 1m chatgpt interaction logs in the wild. In *The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024*. OpenReview.net.

Zheng, L., Chiang, W.-L., Sheng, Y., Li, T., Zhuang, S., Wu, Z., Zhuang, Y., Li, Z., Lin, Z., Xing, E. P., Gonzalez, J. E., Stoica, I., and Zhang, H. (2024). Lmsys-chat-1m: A large-scale real-world LLM conversation dataset. In *The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024*. OpenReview.net.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., and Stoica, I. (2023). Judging llm-as-a-judge with mt-bench and chatbot arena. In *Advances in Neural Information Processing Systems 36*.

Zhou, J., Lu, T., Mishra, S., Brahma, S., Basu, S., Luan, Y., Zhou, D., and Hou, L. (2023). Instruction-following evaluation for large language models. CoRR, abs/2311.07911.

A Open-weight Baselines

We summarize the links to the open-weight LLMs evaluated in this work as below:

* LLaMA 3.2 1B: meta-llama/Llama-3.2-1B-Instruct
* Gemma-3 1B: google/gemma-3-1b-it
* Qwen2.5 0.5B: Qwen/Qwen2.5-0.5B-Instruct
* Qwen2.5 1.5B: Qwen/Qwen2.5-1.5B-Instruct
* Qwen2.5 3B: Qwen/Qwen2.5-3B-Instruct
* SmolLM2 1.7B: HuggingFaceTB/SmolLM2-1.7B-Instruct
* MiniCPM 2B: openbmb/MiniCPM-2B-dpo-bf16
* Qwen2.5 1.5B-GPTQ-int4: Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4
* Qwen2.5 1.5B-AWQ-int4: Qwen/Qwen2.5-1.5B-Instruct-AWQ
* Bonsai 0.5B: deepgrove/Bonsai
* OLMo-Bitnet 1B: NousResearch/OLMo-Bitnet-1B
* Falcon3-1.58bit 7B: tiiuae/Falcon3-7B-Instruct-1.58bit
* Llama3-8B-1.58 8B: HF1BitLLM/Llama3-8B-1.58-100B-tokens

B Evaluation Pipeline Details

To ensure standardized evaluation, we employed established toolkits for different benchmark categories. Specifically:

* For the HumanEval+ coding benchmark, we utilized the evalplus toolkit.
* For the MATH-500 mathematical reasoning benchmark, we used a customized version of the math-evaluation-harness toolkit.
* For the MT-Bench conversational benchmark, evaluation was performed using the official LLM Judge open-source codebase.
* For all other benchmarks assessing language understanding, reasoning, knowledge, and comprehension, we used the standard lm-evaluation-harness framework.

<table><thead><tr><td>Bits</td><td>ADD Energy</td><td>MUL Energy</td></tr></thead><tbody><tr><td>FP16</td><td>0.16</td><td>0.34</td></tr><tr><td>INT8</td><td>0.007</td><td>0.07</td></tr></tbody></table>

Table 4: ADD and MUL energy consumption (in pJ) of different precision at 7nm process nodes.

Models were prompted using a chat format for generative tasks (e.g., GSM8K, IFEval, and MT-Bench), while default settings from the respective toolkits were used for other tasks.

For energy consumption, we utilize the energy model in (Horowitz, 2014; Zhang et al., 2022) to estimate the arithmetic operations energy (AOE) of matrix multiplication. The sequence length is set as 512 tokens. We present the energy consumption for ADD and MUL operation at 7nm process nodes in Table 4.

To assess CPU decoding performance, latency measurements were conducted on a Surface Laptop Studio 2 system powered by a 13th Gen Intel Core i7-13800H processor. The benchmarking process utilized 8 CPU threads. Specifically, the BitNet b1.58 2B4T model was tested using its *bitnet.cpp* implementation, whereas other models were evaluated using the *llama.cpp* framework. For each model, we generated 128 tokens and report the average latency per token for this task.