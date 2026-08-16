BitNet a4.8: 4-bit Activations for 1-bit LLMs

Hongyu Wang* Shuming Ma* Furu Wei◇

https://aka.ms/GeneralAI

Abstract

Recent research on the 1-bit Large Language Models (LLMs), such as BitNet b1.58 [MWM$^{+24}$], presents a promising direction for reducing the inference cost of LLMs while maintaining their performance. In this work, we introduce **BitNet a4.8**, enabling 4-bit activations for 1-bit LLMs. BitNet a4.8 employs a hybrid quantization and sparsification strategy to mitigate the quantization errors introduced by the outlier channels. Specifically, we utilize 4-bit activations for inputs to the attention and feed-forward network layers, while sparsifying intermediate states followed with 8-bit quantization. Extensive experiments demonstrate that BitNet a4.8 achieves performance comparable to BitNet b1.58 with equivalent training costs, while being faster in inference with enabling 4-bit (INT4/FP4) kernels. Additionally, BitNet a4.8 activates only 55% of parameters and supports 3-bit KV cache, further enhancing the efficiency of large-scale LLM deployment and inference.

![](images/task_43082fd44e32_page_0_pic_0.png)

Figure 1: The overview of BitNet a4.8 with both weight and activation quantization. All the parameters are ternary (i.e., 1.58-bit as in BitNet b1.58 [MWM$^{+24}$]). We use a hybrid quantization and sparsification strategy to deal with outlier activations in certain Transformer sub-layers.

* Equal contribution. ◇ Corresponding author. S. Ma and F. Wei are with Microsoft Research. H. Wang is with University of Chinese Academy of Sciences.

arXiv:2411.04965v1 [cs.CL] 7 Nov 2024

# 1 Introduction

Recent works [MWM+24] have shown that 1-bit LLMs can match the full-precision models given the same amount of parameters and training tokens while being significantly cost-effective in terms of latency, memory, throughput, and energy consumption. With model weights represented in 1.58-bit (i.e., {-1, 0, 1}), the bottleneck of inference has shifted from the limited memory bandwidth to high computational cost. Low-bit or sparse activations in LLMs served as a promising approach to further reduce the computational budget while maintaining performance on downstream tasks.

One common approach is to utilize activation sparsity [LWD+23, SXZ+24, LPC+24], which reduces the inference FLOPs and the I/O of weight by pruning the activation entries with smaller magnitudes. Sparsification is particularly well-suited for handling activations that exhibit highly imbalanced long-tailed distributions. Recent works [WMWW24] have demonstrated that LLMs with fully sparsely-activated activations can achieve results comparable to dense models while having much less active parameters.

In addition to sparsification, activation quantization is another approach to accelerate the matrix multiplication. However, the optimization of neural networks with low-bit activations is challenging due to the emergence of outlier dimensions as the training progresses and the model size grows. Despite these outliers only account for a very small portion of the activations [DLBZ22, XLS+23], they have much larger magnitude, which leads to significant quantization errors and performance degradation on downstream tasks. Previous works [XLCZ23, AMC+24, LZF+24, LXW+24] mostly utilize Hadamard or learnable rotary transformation to amortize the outlier features into other entries. However, they are mostly designed for the LLMs of higher precision (e.g., 4-bit). For 1-bit LLMs, the extremely low bit-width of the weights makes it challenging to absorb these transformation matrices directly into the weights, while leaving them as online transformations introduces additional computational overhead and limits overall inference performance.

In this work, we introduce **BitNet a4.8**, a hybrid quantization and sparsification strategy that enables 4-bit activations for 1-bit LLMs. By carefully analyzing the activation distribution of 1-bit LLMs, we selectively apply 4-bit quantization or sparsification based on the distribution patterns of these activations. Specifically, as shown in Figure 1, BitNet a4.8 employs 4-bit activations for the inputs to attention and FFN, while utilizing sparsification with 8 bits for intermediate states. To improve the training efficiency, BitNet a4.8 is trained from 8-bit to 4-bit activations with a two-stage recipe, which requires only a few training tokens to adapt BitNet b1.58 to the low-bit activations at the end of training. Extensive experiments demonstrate that BitNet a4.8 achieves competitive performance to BitNet b1.58 with the same training cost while being significantly more efficient at inference time. Furthermore, BitNet a4.8 has only 55% activated parameters and supports 3-bit KV cache, which further enhances the efficiency of LLM deployment.

# 2 BitNet a4.8

## 2.1 Architecture

As shown in Figure 1, BitNet a4.8 adopts the same layout as BitNet b1.58. Following [WMD+23, MWM+24], we replace the linear projections in both attention and feed-forward network (FFN) with BitLinear to learn 1.58-bit weights from the scratch. For activations, we adopt a hybrid quantization and sparsification strategy to mitigate the error introduced by outlier dimensions.

Figure 2 illustrates the distribution of each component's inputs of a BitNet b1.58 model with 7B model size. Inputs to the attention and FFN layers typically follow a Gaussian-like distribution, while activations before the FFN down projections and the output projections in attention have more outlier channels and massive amount of entries around zero. [LPC+24] also reported similar observations for full-precision LLMs. As shown in Figure 3, directly applying low-bit quantization to these intermediate states introduces substantial quantization errors.

Therefore, we use the sparsification method from Q-Sparse [WMWW24] to retain these intermediate states at 8 bits while removing the computation bottleneck. For output projection of self-attention layers, we use a sparsify-then-quantize function:

$$
\mathbf{Y} = (\mathrm{Q}_{\mathrm{INT8}}(\mathbf{X}) \odot \mathbf{M}) \cdot \mathrm{Q}_{\mathrm{w}}(\mathbf{W})^{\mathrm{T}}, \mathbf{M} = \mathrm{Top}_{k}(|\mathbf{X}|) \quad (1)
$$

![](images/task_43082fd44e32_page_2_pic_0.png)

Figure 2: The distribution of the inputs to each projection. The visualization is conducted with a 7B BitNet b1.58 model on a subset of the valid set of C4. For the layers that exhibit Gaussian-like distributions, we employ 4-bit activation quantization. For the layers which distributions are sharp, we adopt Q-Sparse [WMWW24] to perform sparsification on the activations.

where $Q_w(\cdot)$ and $Q_{\text{INT8}}(\cdot)$ denote the quantization function for weight $\mathbf{W}$ and activations $\mathbf{X}$, respectively. $\mathbf{M}$ is the mask tensor that indicates the maximum top-K elements in terms of the absolute values of the activations $\mathbf{X}$, $\odot$ is the element-wise multiplication operation.

Specifically, the functions of weight quantization and activation quantization can be formulated as:

$$
Q_w(\mathbf{W}) = \alpha \text{RoundClip}\left(\frac{\mathbf{W}}{\alpha + \epsilon}, -1, 1\right), \quad \alpha = \text{mean}(|\mathbf{W}|) \qquad (2)
$$

$$
Q_{\text{INT8}}(\mathbf{X}) = \frac{\gamma}{127} \text{RoundClip}\left(\frac{127}{\gamma + \epsilon} \mathbf{X}, -128, 127\right), \quad \gamma = \text{max}(|\mathbf{X}|) \qquad (3)
$$

$$
\text{RoundClip}(X, a, b) = \min(\max(\text{round}(X), a), b) \qquad (4)
$$

For FFN, we adopt squared ReLU [SML$^{+21}$, WMWW24] and gated linear unit (GLU) to further boost the activation sparsity. It is defined as follows:

$$
\text{ReLU}^2\text{GLU}(\mathbf{X}) = \mathbf{X}\mathbf{W}_{\text{up}}^{T} \odot \text{ReLU}^2(\mathbf{X}\mathbf{W}_{\text{gate}}^{T}) \qquad (5)
$$

According to our preliminary experiments, with squared ReLU, the inputs to the down projection achieve over 80% sparsity with minimal impact on performance. Additionally, we observe that the outputs of gate projection $\text{ReLU}^2(\mathbf{X}\mathbf{W}_{\text{gate}}^T)$ exhibit high activation sparsity as well (e.g., 67.5% for 7B models). This characteristic enables further reduction in inference FLOPs for the up projection by first computing the gate projection and then performing the up projection only on the non-zero channels of the gates.

For the input to attention and FFN, since they have much less outlier features, we use absmean function to quantize the activations to 4-bit integers:

$$
\mathbf{Y} = Q_{\text{INT4}}(\mathbf{X}) \cdot Q_w(\mathbf{W})^T \qquad (6)
$$

$$
Q_{\text{INT4}}(\mathbf{X}) = \frac{\beta}{\sqrt{7}} \text{RoundClip}\left(\frac{\sqrt{7}}{\beta + \epsilon} \mathbf{X}, -8, 7\right), \quad \beta = \text{mean}(|\mathbf{X}|) \qquad (7)
$$

![](images/task_43082fd44e32_page_3_pic_0.png)

Figure 3: The distribution of the inputs to the output projection of attention with different quantization and sparsification. The visualization is conducted with a 7B BitNet b1.58 model on a subset of the valid set of C4.

## 2.2 Training

Continue-training from BitNet b1.58. BitNet a4.8 is trained with a two-stage recipe from W1.58A8 to W1.58A4. For the first stage, we train the model with 8-bit activations and ReLU²GLU. For the second stage, we adopt the hybrid quantization and sparsification as shown in Section 2.1. BitNet a4.8 quickly adapts to 4-bit and sparse activations with only a few training tokens while having negligible loss on performance.

Gradient approximation. Following [WMD⁺23, WMWW24], we use straight-through estimator (STE) [BLC13] to conduct the gradient approximation for BitNet a4.8, as well as mixed precision training to update the parameters. We directly bypass the non-differentiable functions, including the quantization function and top-K sparsification function during the backward propagation. For mixed precision training, we maintain a full-precision latent weight to accumulate parameter updates. During the forward, we quantize the latent weight into 1.58-bit on the fly.

## 2.3 Floating-point quantization

Floating-point quantization offers a broader dynamic range than the integer-based quantization, which is crucial for handling the long-tailed distribution of the activations. For floating-point precision, we only leave the inputs to down projection of FFN at 8-bit integers, and quantize the other activations to FP4 using MinMax quantizer [LLH⁺23]. It is defined as follows:

$$
Q_{\text{FP4}}(\mathbf{X}) = \frac{\gamma}{2^{M+b}} \text{Round}\left(\frac{2^{M+b}}{\gamma} \mathbf{X}\right), \quad \gamma = 2^{\max(\lfloor \log_2 |\mathbf{X}| \rfloor + b, 1)} \quad (8)
$$

$$
b = \log_2\left(\frac{2 - 2^{-M}}{|\mathbf{X}|_{\max}}\right) + 2^E - 1 \quad (9)
$$

where *E* and *M* denote the bit-width of the exponent and mantissa component, respectively. We adopt the E2M1 format due to its larger dynamic range. As shown in Table 1, BitNet a4.8 with FP4

<table><thead><tr><th>Models</th><th>Size</th><th>PPL↓</th><th>ARCe↑</th><th>ARCe↑</th><th>HS↑</th><th>PQ↑</th><th>WGe↑</th><th>Avg↑</th></tr></thead><tbody><tr><td>LLaMA LLM</td><td rowspan="4">700M</td><td>11.44</td><td>27.13</td><td>43.27</td><td>44.70</td><td>68.12</td><td>53.99</td><td>47.44</td></tr><tr><td>BitNet b1.58</td><td>12.32</td><td>25.00</td><td>42.68</td><td>42.08</td><td>66.97</td><td>54.14</td><td>46.17</td></tr><tr><td><b>BitNet a4.8 (FP4)</b></td><td>12.40</td><td>25.17</td><td>42.68</td><td>42.36</td><td>66.27</td><td>52.96</td><td>45.89</td></tr><tr><td><b>BitNet a4.8</b></td><td>12.40</td><td>25.17</td><td>41.58</td><td>42.44</td><td>66.38</td><td>53.04</td><td>45.72</td></tr><tr><td>LLaMA LLM</td><td rowspan="4">1.3B</td><td>10.82</td><td>27.90</td><td>45.16</td><td>47.65</td><td>69.91</td><td>53.35</td><td>48.79</td></tr><tr><td>BitNet b1.58</td><td>11.27</td><td>27.65</td><td>45.33</td><td>46.86</td><td>68.39</td><td>54.06</td><td>48.46</td></tr><tr><td><b>BitNet a4.8 (FP4)</b></td><td>11.38</td><td>28.50</td><td>44.36</td><td>47.03</td><td>68.61</td><td>54.06</td><td>48.51</td></tr><tr><td><b>BitNet a4.8</b></td><td>11.35</td><td>28.50</td><td>44.15</td><td>46.98</td><td>68.34</td><td>54.14</td><td>48.42</td></tr><tr><td>LLaMA LLM</td><td rowspan="4">3B</td><td>9.61</td><td>29.95</td><td>48.11</td><td>55.25</td><td>71.76</td><td>57.46</td><td>52.51</td></tr><tr><td>BitNet b1.58</td><td>9.97</td><td>29.27</td><td>49.41</td><td>54.42</td><td>70.89</td><td>57.54</td><td>52.30</td></tr><tr><td><b>BitNet a4.8 (FP4)</b></td><td>9.99</td><td>29.10</td><td>49.24</td><td>54.60</td><td>71.38</td><td>56.12</td><td>52.08</td></tr><tr><td><b>BitNet a4.8</b></td><td>9.97</td><td>28.33</td><td>49.58</td><td>54.62</td><td>71.16</td><td>54.38</td><td>51.61</td></tr><tr><td>LLaMA LLM</td><td rowspan="4">7B</td><td>9.20</td><td>33.36</td><td>51.22</td><td>58.33</td><td>73.34</td><td>58.41</td><td>54.93</td></tr><tr><td>BitNet b1.58</td><td>9.24</td><td>32.00</td><td>50.88</td><td>59.79</td><td>72.96</td><td>59.83</td><td>55.09</td></tr><tr><td><b>BitNet a4.8 (FP4)</b></td><td>9.42</td><td>31.57</td><td>51.22</td><td>58.20</td><td>72.47</td><td>59.59</td><td>54.61</td></tr><tr><td><b>BitNet a4.8</b></td><td>9.37</td><td>31.66</td><td>50.88</td><td>58.78</td><td>73.01</td><td>59.35</td><td>54.74</td></tr></tbody></table>

Table 1: Perplexity and results of BitNet a4.8, BitNet b1.58 and LLaMA LLM on the end tasks. The standard variance of error for average scores is 1.06%.

quantization has the similar performance as it with the hybrid quantization and sparsification strategy based on integers.

## 3 Experiments

We compared BitNet a4.8 to BitNet b1.58 and our reproduced FP16 LLaMA LLM of various sizes. For 1.58-bit models, we adopted the two-stage weight decay and learning rate scheduling following the training recipe of BitNet b1.58 [MWM<sup>+24</sup>]. More details can be found in the Appendix A. All models were trained with 100B tokens from the RedPajama dataset [Com23] to ensure a fair comparison. For BitNet a4.8, we first train the model with 8-bit activations for 95B tokens. Then we reuse the optimizer states and continue-train the model with the proposed hybrid quantization and sparsification for 5B tokens. We set topK as 50% for the output projection of attention.

We evaluated the zero-shot accuracy for these models on a range of language tasks using the *lm-evaluation-harness* toolkit [GTA<sup>+24</sup>], including ARC-Easy (ARCe) [YBS19], ARC-Challenge (ARCc) [YBS19], Hellaswag (HS) [ZHB<sup>+19</sup>], Winogrande (WGe) [SBBC20] and PIQA (PQ) [BZB<sup>+19</sup>]. We also reported the perplexity on the validation set of C4 [RSR<sup>+19</sup>] dataset.

### 3.1 Main Results

Table 1 summarizes the detailed results of BitNet a4.8, BitNet b1.58 and FP16 LLaMA LLM. The performance gap between full-precision (i.e., FP16) LLaMA LLM and BitNet b1.58 narrows as the model size grows. For 7B models, BitNet b1.58 matches LLaMA LLM in terms of both language model perplexity and average accuracy on the end tasks. Furthermore, BitNet a4.8 achieves performance comparable to BitNet b1.58, with almost no loss in average accuracy.

**Sparsity.** Table 2 demonstrates detailed sparsity of each component for BitNet a4.8, BitNet b1.58 and FP16 LLaMA LLM across various sizes. The sparsity is calculated with non-embedding parameters on the valid set of C4. Notably, BitNet a4.8 achieves significantly higher sparsity than both BitNet b1.58 and LLaMA LLM. For example, in the 7B model, BitNet a4.8 reaches an overall sparsity of 44.5%, with only 3.4B active parameters. The inputs to the down projection demonstrate particularly high sparsity, consistent with our observation that intermediate state distributions are sharply centered around zero. Additionally, we also observe that the outputs of gate projection are very sparse. It leads to a high sparsity for up projection, since we only need to perform projection on the non-zero channels selected from the gates. Specifically, for the 7B BitNet a4.8, the sparsity of the

<table><thead><tr><th>Models</th><th>Activated</th><th>QKV</th><th>Out</th><th>Up</th><th>Gate</th><th>Down</th><th>Overall</th></tr></thead><tbody><tr><td>LLaMA LLM</td><td>679M</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>BitNet b1.58</td><td>638M</td><td>1.2</td><td>5.9</td><td>1.2</td><td>1.2</td><td>21.8</td><td>6.2</td></tr><tr><td><b>BitNet a4.8</b></td><td>390M</td><td>12.1</td><td>50.0</td><td>66.2</td><td>12.1</td><td>80.9</td><td>42.5</td></tr><tr><td>LLaMA LLM</td><td>1.2B</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>BitNet b1.58</td><td>1.1B</td><td>1.3</td><td>5.8</td><td>1.2</td><td>1.2</td><td>22.8</td><td>6.4</td></tr><tr><td><b>BitNet a4.8</b></td><td>0.7B</td><td>12.0</td><td>50.0</td><td>65.9</td><td>12.1</td><td>81.8</td><td>42.7</td></tr><tr><td>LLaMA LLM</td><td>3.2B</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>BitNet b1.58</td><td>3.0B</td><td>1.4</td><td>7.1</td><td>1.3</td><td>1.3</td><td>30.0</td><td>8.2</td></tr><tr><td><b>BitNet a4.8</b></td><td>1.8B</td><td>12.1</td><td>50.0</td><td>70.7</td><td>12.1</td><td>85.6</td><td>44.7</td></tr><tr><td>LLaMA LLM</td><td>6.5B</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>BitNet b1.58</td><td>6.0B</td><td>1.7</td><td>11.2</td><td>1.4</td><td>1.4</td><td>24.2</td><td>7.3</td></tr><tr><td><b>BitNet a4.8</b></td><td>3.4B</td><td>12.1</td><td>50.0</td><td>71.4</td><td>12.0</td><td>84.2</td><td>44.5</td></tr></tbody></table>

Table 2: Detailed sparsity of BitNet a4.8, BitNet b1.58 and LLaMA LLM on the valid set of C4.

<table><thead><tr><th>Models</th><th>Size</th><th>ARCc↑</th><th>ARCe↑</th><th>HS↑</th><th>PQ↑</th><th>WGe↑</th><th>Avg↑</th></tr></thead><tbody><tr><td>BitNet a4.8</td><td rowspan="4">3B</td><td>28.33</td><td>49.58</td><td>54.62</td><td>71.16</td><td>54.38</td><td>51.61</td></tr><tr><td>w/ 4-bit KV</td><td>28.24</td><td>48.86</td><td>54.41</td><td>71.87</td><td>55.49</td><td>51.77</td></tr><tr><td>w/ 4-bit QKV</td><td>27.30</td><td>48.91</td><td>54.32</td><td>71.98</td><td>56.75</td><td>51.85</td></tr><tr><td>w/ 4-bit Q, 3-bit KV</td><td>28.84</td><td>48.91</td><td>53.87</td><td>70.95</td><td>56.35</td><td>51.78</td></tr><tr><td>BitNet a4.8</td><td rowspan="4">7B</td><td>31.66</td><td>50.88</td><td>58.78</td><td>73.01</td><td>59.35</td><td>54.74</td></tr><tr><td>w/ 4-bit KV</td><td>31.40</td><td>50.93</td><td>58.68</td><td>73.12</td><td>60.85</td><td>55.00</td></tr><tr><td>w/ 4-bit QKV</td><td>30.63</td><td>51.30</td><td>58.45</td><td>72.52</td><td>59.83</td><td>54.55</td></tr><tr><td>w/ 4-bit Q, 3-bit KV</td><td>31.14</td><td>50.93</td><td>58.07</td><td>72.96</td><td>59.04</td><td>54.43</td></tr></tbody></table>

Table 3: Detailed results of BitNet a4.8 with QKV states varying bit-widths on the end tasks. We reported the zero-shot accuracy of all models.

gates and the inputs to up projection is 67.5% and 12.0%, respectively. Consequently, the sparsity of up projection can be estimated as $1 - (1 - 12.0\%) \times (1 - 67.5\%)$, that is 71.4%.

**Low-bit Attention.** Table 3 presented detailed results of BitNet a4.8 with low-bit attention in 3B and 7B model size. Low-bit attention is essential for efficient long sequence modeling, as it reduces the memory footprint and IO of KV cache and accelerates the attention computation. In our experiments, we adopted post-RoPE quantization. The QKV heads were directly quantized to unsigned integers using the absmax function, without the need of any calibration dataset. For 3-bit KV quantization, we retain the heads of the bos token at 4-bit, as it contains more outlier features. As shown in Table 3, BitNet a4.8 achieves negligible accuracy loss with 4-bit KV or QKV heads in 3B and 7B models. Furthermore, the KV cache of BitNet a4.8 can be quantized to 3-bit integers, resulting in almost no degradation on average accuracy.

## 3.2 Ablation Study

**Hybrid architecture.** Figure 4 presented the training loss curve of 700M BitNet a4.8 with the full INT4/FP4 quantization, and the hybrid quantization and sparsification. We train these models with the first-stage scheduling for 25B tokens from the RedPajama dataset. We adopt absmean and MinMax quantizer for full INT4 and FP4 quantization, respectively. Besides, for full INT4 quantization, we use absmean quantizer with $\beta = 2\text{mean}(|X|)$ for down projection in FFN, as its inputs have larger outliers. As shown in Figure 4, the full INT4 quantization leads to divergence. Furthermore, the hybrid architecture significantly outperforms the full FP4 architecture in terms of training perplexity.

**Down projection of FFN.** We compared 1.3B BitNet a4.8 with different quantization or activation function for the down projection of FFN. All models were trained with the first-stage scheduling for 50B tokens from the RedPajama dataset. To ensure a fair comparison, we leave the other activations

![](images/task_43082fd44e32_page_6_pic_0.png)

Figure 4: Ablation study on the hybrid quantization and sparsification.

![](images/task_43082fd44e32_page_6_pic_1.png)

Figure 5: Ablation study on different quantization or activation function for the inputs to down projection of FFN.

at 8-bits. We adopt the absmax quantizer for INT8 quantization and MinMax quantizer for FP4 quantization. The $\beta$ of absmean quantizer is set as 2mean($|X|$). Figure 5 shows the training loss curves of these models. Squared ReLU achieves slightly better training perplexity than Swish while enabling higher sparsity. Furthermore, applying FP4 quantization for the inputs to the down projection leads to a significant performance degradation, while using INT4 activations with STE causes divergence.

**Output projection of attention.** Table 4 demonstrates detailed results of 3B BitNet a4.8 with and without Top-K sparsification for the inputs to the output projection of attention. Both models were trained with the same two stage recipe from 8-bit to 4-bit activations. We set $K$ as 50% for sparsification. The baseline utilized the INT8 absmax quantizer for the output projection's inputs. The results show that TopK sparsification brings negligible perplexity and accuracy loss.

**4-bit quantization.** We presented the loss curves of 3B BitNet a4.8 with different 4-bit quantizers for the inputs to the attention and FFN. We compared the performance of BitNet a4.8 with floating-point quantization with E2M1 and E1M2 formats using MinMax quantizer, integer quantization with absmax and absmean quantizer. As shown in Figure 6, FP4 with E2M1 format and INT4 with absmean quantizer achieve slightly better training perplexity, as they are well-suited for handling small-magnitude activation entries.

![](images/task_43082fd44e32_page_6_pic_2.png)

Figure 6: Ablations on 4-bit quantizers for the inputs to attention and FFN.

<table><thead><tr><th>Quantization</th><th>Sparsification</th><th>PPL↓</th><th>ARCc↑</th><th>ARCe↑</th><th>HS↑</th><th>PQ↑</th><th>WGe↑</th><th>Avg↑</th></tr></thead><tbody><tr><td>INT8</td><td>-</td><td>9.95</td><td>28.33</td><td>48.53</td><td>54.90</td><td>72.31</td><td>56.51</td><td>52.11</td></tr><tr><td>INT8</td><td>TopK 50%</td><td>9.97</td><td>28.33</td><td>49.58</td><td>54.62</td><td>71.16</td><td>54.38</td><td>51.61</td></tr></tbody></table>

Table 4: Ablations on the TopK sparsification for the inputs to the output projection of attention.

<table><thead><tr><th>Models</th><th>HS</th><th>PQ</th><th>WGe</th><th>OBQA</th><th>Lambada</th><th>MMLU</th><th>ARCc</th><th>ARCe</th><th>Avg</th></tr></thead><tbody><tr><td>BitNet b1.58 2B</td><td>68.66</td><td>77.09</td><td>62.58</td><td>41.40</td><td>63.36</td><td>50.29</td><td>47.61</td><td>70.74</td><td>60.22</td></tr><tr><td>BitNet a4.8 2B</td><td>68.21</td><td>76.55</td><td>64.40</td><td>40.60</td><td>63.75</td><td>50.30</td><td>46.59</td><td>70.00</td><td>60.05</td></tr></tbody></table>

Table 5: Results of BitNet a4.8 and BitNet b1.58 with 2B parameters and 2T training tokens.

3.3 More Training Tokens

Prior research [DLBZ22] demonstrates a positive correlation between the number of training tokens and the prevalence of activation outliers in language models. To rigorously evaluate the scalability characteristics of BitNet a4.8, we conducted extensive experiments using a model configuration with 2 billion parameters trained on 2 trillion tokens. We performed a controlled comparison against BitNet b1.58 using identical training data and configurations. The empirical results, presented in Table 5, demonstrate that BitNet a4.8 maintains performance parity with negligible degradation in accuracy metrics while achieving 4-bit activation compression. These findings provide strong evidence for the efficacy of our proposed approach at scale.

4 Conclusion

In this paper, we present BitNet a4.8 which enables 4-bit activations for 1-bit LLMs. BitNet a4.8 uses a novel hybrid quantization and sparsification architecture to reduce the quantization errors introduced by outlier channels of activations. Specifically, we employ 4-bit quantization for inputs to the attention and FFN layers, while sparsifying the intermediate states with 8-bit integers. BitNet a4.8 is continue-trained from W1.58A8 to W1.58A4. Experimental results demonstrate that BitNet a4.8 achieves results comparable to BitNet b1.58 with the same training cost, while significantly enhancing inference efficiency.

5 Acknowledgements

We would like to acknowledge Lei Wang for the discussion on the inference efficiency.

References

[AMC+24] Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. Quarot: Outlier-free 4-bit inference in rotated llms. CoRR, abs/2404.00456, 2024.

[BLC13] Yoshua Bengio, Nicholas Léonard, and Aaron C. Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. CoRR, abs/1308.3432, 2013.

[BZB+19] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: reasoning about physical commonsense in natural language. CoRR, abs/1911.11641, 2019.

[Com23] Together Computer. Redpajama: an open dataset for training large language models, 2023.

[DLBZ22] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. LLM.int8(): 8-bit matrix multiplication for transformers at scale. CoRR, 2022.

[GTA$^{+24}$] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 07 2024.

[LLH$^{+23}$] Shih-Yang Liu, Zechun Liu, Xijie Huang, Pingcheng Dong, and Kwang-Ting Cheng. LLM-FP4: 4-bit floating-point quantized transformers. In EMNLP 2023, pages 592-605. Association for Computational Linguistics, 2023.

[LPC$^{+24}$] James Liu, Pragaash Ponnusamy, Tianle Cai, Han Guo, Yoon Kim, and Ben Athiwaratkun. Training-free activation sparsity in large language models. CoRR, abs/2408.14690, 2024.

[LWD$^{+23}$] Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher Ré, and Beidi Chen. Deja vu: Contextual sparsity for efficient llms at inference time. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 22137-22176. PMLR, 2023.

[LXW$^{+24}$] Haokun Lin, Haobo Xu, Yichen Wu, Jingzhi Cui, Yingtao Zhang, Linzhan Mou, Linqi Song, Zhenan Sun, and Ying Wei. Duquant: Distributing outliers via dual transformation makes stronger quantized llms. arXiv preprint arXiv:2406.01721, 2024.

[LZF$^{+24}$] Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. Spinquant: LLM quantization with learned rotations. CoRR, abs/2405.16406, 2024.

[MWM$^{+24}$] Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, and Furu Wei. The era of 1-bit llms: All large language models are in 1.58 bits. CoRR, abs/2402.17764, 2024.

[RSR$^{+19}$] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. CoRR, abs/1910.10683, 2019.

[SBBC20] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Wino-Grande: an adversarial winograd schema challenge at scale. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, pages 8732-8740, 2020.

[SML$^{+21}$] David R. So, Wojciech Manke, Hanxiao Liu, Zihang Dai, Noam Shazeer, and Quoc V. Le. Primer: Searching for efficient transformers for language modeling. CoRR, abs/2109.08668, 2021.

[SXZ$^{+24}$] Yixin Song, Haotong Xie, Zhengyan Zhang, Bo Wen, Li Ma, Zeyu Mi, and Haibo Chen. Turbo sparse: Achieving llm sota performance with minimal activated parameters. arXiv preprint arXiv:2406.05955, 2024.

[WMD$^{+23}$] Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, and Furu Wei. Bitnet: Scaling 1-bit transformers for large language models. CoRR, abs/2310.11453, 2023.

[WMWW24] Hongyu Wang, Shuming Ma, Ruiping Wang, and Furu Wei. Q-sparse: All large language models can be fully sparsely-activated. CoRR, abs/2407.10969, 2024.

[XLCZ23] Haocheng Xi, Changhao Li, Jianfei Chen, and Jun Zhu. Training transformers with 4-bit integers. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems, 2023.

[XLS$^{+23}$] Guangxuan Xiao, Ji Lin, Mickaël Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant: accurate and efficient post-training quantization for large language models. In *International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA*, 2023.

[YBS19] Vikas Yadav, Steven Bethard, and Mihai Surdeanu. Quick and (not so) dirty: Unsupervised selection of justification sentences for multi-hop question answering. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, *EMNLP-IJCNLP*, 2019.

[ZHB$^{+19}$] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: can a machine really finish your sentence? In *Proceedings of the 57th Conference of the Association for Computational Linguistics*, pages 4791-4800, 2019.

A Hyper-parameters

<table><thead><tr><th>Size</th><th>Hidden Size</th><th>GLU Size</th><th>#Heads</th><th>#Layers</th><th>Batch Size</th><th># Tokens</th><th>Seq Length</th></tr></thead><tbody><tr><td>700M</td><td>1536</td><td>4096</td><td>24</td><td>24</td><td>1M</td><td>100B</td><td>2048</td></tr><tr><td>1.3B</td><td>2048</td><td>5460</td><td>32</td><td>24</td><td>1M</td><td>100B</td><td>2048</td></tr><tr><td>3B</td><td>3200</td><td>8640</td><td>32</td><td>26</td><td>1M</td><td>100B</td><td>2048</td></tr><tr><td>7B</td><td>4096</td><td>11008</td><td>32</td><td>32</td><td>1M</td><td>100B</td><td>2048</td></tr></tbody></table>

Table 6: Model configurations for both BitNet a4.8, BitNet b1.58 and LLaMA LLM.

<table><thead><tr><th>Model</th><th>Size</th><th>Learning Rate</th><th>Weight Decay</th><th>Warm-up</th><th>Adam β</th></tr></thead><tbody><tr><td rowspan="4">BitNet a4.8</td><td>700M</td><td>1.5 × 10<sup>-3</sup> → 1 × 10<sup>-3</sup></td><td>0.1 → 0</td><td>375</td><td>(0.9, 0.95)</td></tr><tr><td>1.3B</td><td>1.2 × 10<sup>-3</sup> → 8 × 10<sup>-4</sup></td><td>0.1 → 0</td><td>375</td><td>(0.9, 0.95)</td></tr><tr><td>3B</td><td>1.2 × 10<sup>-3</sup> → 6.4 × 10<sup>-4</sup></td><td>0.1 → 0</td><td>375</td><td>(0.9, 0.95)</td></tr><tr><td>7B</td><td>1 × 10<sup>-3</sup> → 6 × 10<sup>-4</sup></td><td>0.1 → 0</td><td>375</td><td>(0.9, 0.95)</td></tr><tr><td rowspan="4">LLaMA LLM</td><td>700M</td><td>2.5 × 10<sup>-4</sup></td><td>0.1</td><td>375</td><td>(0.9, 0.95)</td></tr><tr><td>1.3B</td><td>2.0 × 10<sup>-4</sup></td><td>0.1</td><td>375</td><td>(0.9, 0.95)</td></tr><tr><td>3B</td><td>2.0 × 10<sup>-4</sup></td><td>0.1</td><td>375</td><td>(0.9, 0.95)</td></tr><tr><td>7B</td><td>1.5 × 10<sup>-4</sup></td><td>0.1</td><td>375</td><td>(0.9, 0.95)</td></tr></tbody></table>

Table 7: Hyper-parameters for both BitNet a4.8 and LLaMA LLM training.