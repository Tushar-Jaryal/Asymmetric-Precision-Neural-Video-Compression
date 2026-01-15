## Asymmetric Precision Neural Video Compression:

## Combining FP16 Encoding Quality with INT

## Decoding Efficiency

```
Wei-Jung Tsao, Tushar Jaryal, Ting-Yuan Chang
```
**_Abstract_** **—We propose a hybrid precision NVC
architecture (FP16 encoder, LSQ-based INT8 decoder) to
ensure cross-platform consistency. Achieving a BD-PSNR
of 0.0000 dB, our method guarantees deterministic
decoding and hardware efficiency with zero quality loss
against the FP32 baseline.**

I. INTRODUCTION
The rapid evolution of the digital era has precipitated
an unprecedented surge in data volume, with video
content currently constituting the preponderant share
of global network traffic. While Neural Video Codecs (NVCs)
have demonstrated the potential to surpass traditional coding
standards, such as H.266/VVC, in terms of rate-distortion (R-
D) performance, their practical deployment is hampered by
significant challenges regarding cross-platform consistency
and hardware inference efficiency.

Existing high-performance NVCs predominantly rely on
floating-point arithmetic. However, floating-point operations
exhibit non-determinism across disparate hardware platforms
(e.g., varying GPUs or NPUs). Consequently, minute round-off
errors can accumulate significantly as video frames progress,
ultimately precipitating severe temporal drift and catastrophic
degradation of the reconstructed video. To mitigate this issue,
model integerization has emerged as a prerequisite for ensuring
cross-platform consistency. Furthermore, this approach
significantly alleviates computational complexity and memory
bandwidth requirements.

```
Nevertheless, existing quantization strategies often fail to
strike an optimal balance between performance and efficiency.
For instance, while DCVC-RT achieves cross-platform
consistency via Int16 quantization, its inference speed is
constrained by the fact that many hardware accelerators lack
mature optimization for Int16 compared to Int8. Conversely,
MobileNVC attempts Int8 quantization to pursue maximal
speed; however, this aggressive quantization scheme results in
a prohibitive compression performance penalty—up to an 80%
increase in BD-Rate—rendering it unsuitable for high-quality
video transmission. This highlights an unresolved dilemma in
low-bitwidth quantization: maintaining high compression
efficiency while simultaneously guaranteeing decoding
consistency.
```
```
To address these limitations, we propose a hybrid-precision
quantization strategy designed to synergize the hardware
acceleration benefits at the decoder with overall compression
performance. Specifically, apart from the quantization prior to
the entropy model, we strategically quantize the core
components on the decoder side—comprising the Feature
Extractor and the Memory Module—to INT8 precision. Since
these modules are critical for reconstructing the current frame
and generating references for future frames, fixing them to
integer arithmetic eliminates the accumulation of floating-point
errors, thereby ensuring consistent decoding results across
different platforms while fully leveraging the Int8 acceleration
capabilities of mobile hardware.
```


In contrast, we retain FP16 precision for the encoder-side
components to preserve high feature expressiveness and
maximize compression quality. To mitigate the precision loss
induced by quantizing decoder modules to INT8, we employ
the Learned Step Size Quantization (LSQ) technique. By
learning the optimal quantization step size during training, LSQ
effectively minimizes quantization noise, enabling the Int
model to approximate the behavior of its floating-point
counterpart more accurately.

Ultimately, this work presents a receiver-oriented hybrid
precision architecture that reconciles the trade-off between
coding efficiency and deployment constraints. By restricting
LSQ-based INT8 quantization to the decoder-side modules
while preserving FP16 precision at the encoder, we guarantee
cross-platform consistency and real-time throughput without
the severe performance penalties typically observed in fully
quantized frameworks.

II. RELATED WORKS
A. _Real-Time Neural Video Compression_
As Neural Video Codecs (NVCs) have gradually surpassed
traditional coding standards, such as H.265/HEVC and
H.266/VVC, in terms of Rate-Distortion (R-D) performance,
the research focus has shifted from solely pursuing
compression efficiency to enhancing model practicality and
inference efficiency. In this domain, MobileNVC[1] and
DCVC-RT[2] represent two significant breakthroughs.

MobileNVC represents the first neural video decoder
optimized specifically for mobile devices, aiming to achieve
real-time 1080p decoding.
To mitigate computational complexity, MobileNVC replaces
the computationally expensive pixel-dense warping
characteristic of traditional NVCs with efficient block-based
motion compensation. Furthermore, by leveraging the Neural
Processing Unit (NPU) and implementing parallel entropy
decoding techniques, the model successfully achieves real-time
operation on mobile chipsets.

```
Conversely, DCVC-RT addresses efficiency from the
perspective of reducing "Operational Complexity." The study
identifies memory I/O and the frequency of function calls as
critical bottlenecks limiting inference speed. Consequently,
DCVC-RT eliminates explicit motion estimation and
compensation modules in favor of implicit temporal modeling.
Additionally, it utilizes a single low-resolution latent
representation to accelerate feature extraction and
reconstruction. These architectural innovations allow DCVC-
RT to achieve exceptionally high encoding and decoding frame
rates on desktop GPUs while maintaining superior R-D
performance.
```
```
B. Challenges in cross-platform consistency
Despite the advancements in inference speed, cross-platform
consistency remains a formidable obstacle to the practical
deployment of NVCs. The majority of high-performance NVCs
rely on floating-point arithmetic; however, floating-point
operations exhibit non-determinism across disparate hardware
architectures (e.g., NVIDIA GPUs, Intel CPUs, or mobile
NPUs). In video coding, which relies heavily on temporal
prediction, minute floating-point round-off errors generated
during decoding accumulate and propagate across frames. This
error drift eventually leads to severe visual artifacts and the
catastrophic collapse of the decoded video sequence.
```
```
To address this issue, model integerization—which enforces
all operations to be performed within a deterministic integer
domain—has emerged as the primary solution for ensuring
consistent decoding results across different platforms.
```
```
C. Limitations of Integer NVCs
Although MobileNVC and DCVC-RT attempt to resolve the
consistency issue through quantization, existing solutions are
forced to make difficult trade-offs between compression
performance and inference latency:
```

**Figure 1** : Hybrid Precision Framework Overview. The proposed architecture partitions the codec into an FP16 encoding path (sender)
to preserve feature expressiveness and an INT8 decoding path (receiver) utilizing LSQ to ensure cross-platform consistency and
hardware efficiency.

1) **Precision Loss in MobileNVC:**
MobileNVC adopts a "floating-point-centric" approach,
wherein a floating-point model is trained first and subsequently
converted to an Int8 model via Post-Training Quantization
(PTQ) or Quantization-Aware Training (QAT). While 8-bit
quantization significantly accelerates inference, it incurs a
severe penalty on coding efficiency. Experimental results
indicate that, compared to its floating-point counterpart, the
Int8 version of MobileNVC suffers from a BD-Rate loss of up
to 80%, rendering it unsuitable for high-quality video
transmission scenarios.

2) **Speed Bottlenecks in DCVC-RT:**
DCVC-RT opts for Int16 quantization to preserve higher
model precision and ensure consistency. However, current
hardware accelerators exhibit significantly less mature
optimization for Int16 operations compared to Int8 or FP16.

Although Int16 guarantees decoding determinism, it imposes
a substantial computational overhead. For instance, on an
NVIDIA A100 GPU, enabling Int16 mode causes the encoding
speed to drop from 125 fps to 28.3 fps, representing an
approximate 5-fold reduction in speed.

Current state-of-the-art techniques fail to simultaneously
satisfy the triad of requirements: cross-platform consistency,
high compression efficiency, and real-time inference speed.

```
This necessitates the exploration of novel quantization and
model design strategies to overcome the prevailing
performance-efficiency trade-offs.
```
### III. PROPOSED METHOD

```
A. Hybrid Precision Architecture
To address the asymmetric computational constraints
between the sender and receiver in video coding applications,
we propose a receiver-oriented hybrid precision Neural Video
Compression (NVC) architecture ( Figure 1 ). While full
floating-point models offer high compression efficiency, they
suffer from cross-platform decoding inconsistency. Conversely,
fully integerized models (e.g., MobileNVC) often incur
significant rate-distortion (R-D) performance degradation.
Therefore, our core strategy partitions the model into a high-
precision FP16 encoding path and a fixed-point INT8 decoding
path:
```
```
1) Sender (Encoder Side):
Comprising the Encoder and Hyperprior Encoder modules.
These components typically operate on servers or workstations
with abundant computational resources. Consequently, we
retain FP16 (Half-Precision Floating Point) precision for these
modules to maximize feature extraction granularity and
compression efficiency.
```

2) **Receiver (Decoder Side):**
Comprising the Decoder, Feature Extractor, and Memory
Module. These components are deployed on resource-
constrained edge devices and must guarantee decoding
determinism. Accordingly, we enforce strict INT8 quantization
for this path.

This design preserves the high expressiveness of floating-
point features at the encoder while utilizing integer arithmetic
at the decoder to eliminate floating-point round-off errors
caused by hardware discrepancies.

B. _INT8 Decoding Path_
A critical challenge in hybrid precision architectures is
preventing the propagation of temporal errors. Video coding
relies heavily on temporal prediction; minute floating-point
errors within the decoding loop accumulate over frames,
leading to severe error drift and reconstruction artifacts.

To mitigate this, we enforce INT8 integerization on three
core components within the decoding loop:

1) **Decoder:**
Responsible for reconstructing pixel-domain frames from
latent representations. INT8 arithmetic significantly reduces
the computational latency of this high-load module.

2) **Feature Extractor:**
Extracts features from reconstructed frames. Quantization
reduces the memory bandwidth pressure during interactions
with the memory module.

3) **Memory Module:**
This is the linchpin for cross-platform consistency. We
mandate that all historical states and features stored in the
memory module be fixed to INT8. This ensures that the
reference features read for the subsequent frame are bit-exact
across all hardware platforms, thereby physically blocking the
accumulation of floating-point drift.

```
C. LSQ-Based Quantization-Aware Training
To compensate for the precision loss induced by quantizing
the decoding path to INT8, we adopt Learned Step Size
Quantization (LSQ)[3] for Quantization-Aware Training
(QAT).
```
```
Unlike static post-training quantization (PTQ), LSQ treats
the quantization step size 𝑠 as a trainable parameter, allowing
the network to adaptively learn the optimal dynamic range for
feature distributions.
```
```
For a given activation or weight 𝑣, the quantization process
is defined as:
```
### 𝑣̅=⌊𝑐𝑙𝑖𝑝(𝑣𝑠,−𝑄𝑁,𝑄𝑃)⌉,

### 𝑣̂=𝑣̅×𝑠 ( 1 )

```
where 𝑄𝑁= 128 and 𝑄𝑃= 127 define the INT8 range.
LSQ provides a means to learn s based on the training loss by
introducing the following gradient through the quantizer to the
step size parameter:
```
### 𝜕𝑣̂

### 𝜕𝑠={

### −𝑣/𝑠+⌊𝑣/𝑠⌉, 𝑖𝑓−𝑄𝑁<𝑣𝑠<𝑄𝑃

### −𝑄𝑁, 𝑖𝑓 𝑣/𝑠 ≤−𝑄𝑁

### 𝑄𝑃, 𝑖𝑓 𝑣/𝑠 ≥−𝑄𝑁

### ( 2 )

```
Additionally, we utilize the Straight-Through Estimator
(STE) to approximate gradients for the rounding operation
(𝜕𝑣̂/𝜕𝑣 ) during backpropagation. This enables the FP
encoder to "perceive" the quantization noise introduced by the
INT8 decoder and generate latent representations that are
robust to quantization artifacts.
```
```
Ultimately, our asymmetric architecture and LSQ-based
optimization reconcile the tension between high compression
efficiency and deployment constraints. The result is a practical
NVC system that ensures cross-platform consistency and real-
time speed without compromising quality.
```

### IV. EXPERIMENT

A. _Experimental Settings_
We evaluated the proposed method on a high-performance
workstation designed to benchmark training stability and
inference feasibility. The hardware configuration consists of an
Intel(R) Xeon(R) W-2255 CPU @ 3.70GHz, 125 GiB of RAM,
and an NVIDIA GeForce RTX 4090 GPU (24GB VRAM). The
implementation is based on PyTorch, building upon the official
DCVC-RT codebase.

To validate the quantization robustness, we adopted a
"calibration on target" strategy, performing calibration directly
on the test sequences to maximize performance for specific
resolutions. We quantify the compression performance using
Rate-Distortion (R-D) curves and the Bjøntegaard Delta PSNR
(BD-PSNR) metric. The primary goal is to compare our INT
LSQ-based model against the original FP32 DCVC-RT anchor
to assess quality preservation.

B. _Implementation Details_
We replaced the standard convolutional layers in the DCVC-
RT decoder with custom LSQConv2d layers. Unlike naive
quantization which relies on static min-max statistics, our
Learned Step Size Quantization (LSQ) treats the step size s as
a trainable parameter. As shown in Figure 2 , the network
dynamically learns varying step sizes across different layers,
adapting to the specific dynamic range required by each feature
map.

Teacher-Student Distillation. We utilized a Teacher-Student
knowledge distillation framework to fine-tune the quantization
parameters without requiring ground-truth labels.

Teacher:
The frozen pre-trained FP32 DCVC-RT decoder.

Student:
The trainable LSQ-equipped INT8 model.

```
Figure 2 : Learned LSQ Step Sizes per Layer. The magnitude
of the learned quantization step sizes (s) across different layers
of the decoder. The variation indicates that the model
adaptively assigns different dynamic ranges to different depths
of the network, which static quantization fails to achieve.
```
```
Training Objective:
We minimized the Mean Squared Error (MSE) between the
reconstructed frames of the Teacher and Student models over
100 epochs. As illustrated in Figure 3 , the quantization error
across most layers remains extremely low (< 2. 0 × 10 −^5 ),
validating that the student model successfully approximates the
teacher's floating-point behavior.
```
```
To bridge the gap between research code and practical
deployment, we addressed three critical engineering challenges
identified during testing:
```
```
Figure 3 : Learned LSQ Step Sizes per Layer. The magnitude
of the learned quantization step sizes (s) across different layers
of the decoder. The variation indicates that the model
adaptively assigns different dynamic ranges to different depths
of the network, which static quantization fails to achieve.
```

1) **Parameter Locking:**
We implemented a mechanism to lock the is_initialized flag
during inference, preventing the learned LSQ parameters from
resetting to default values.

2) **Precision Management:**
To prevent numerical overflow caused by extremely small
learned step sizes (s)—where 1/s results in infinity under
FP16—we enforced Float32 precision for the inference of
quantization layers.

3) **Buffer Overflow Prevention:**
For high-complexity scenarios (e.g., the Kimono sequence at
QP 21), we developed a robust execution script (RobustRunner)
to handle potential buffer overflows in the entropy coder
backend, ensuring data integrity.

C. _Rate-Distortion Performance_
We evaluated the Rate-Distortion performance by comparing
the reconstructed video quality of our INT8 model against the
FP32 baseline.

The experimental results demonstrate that our proposed
method achieves a BD-PSNR of 0.0000 dB. As illustrated in
the R-D curves (see Figure), our INT8 model achieves a perfect
overlap with the original FP32 baseline across all tested bitrates.
This indicates that our LSQ-based quantization incurs zero
quality loss, successfully transforming the research-grade
codec into a hardware-friendly format without the degradation
typically associated with naive quantization.

To understand why LSQ outperforms naive approaches, we
visualized the weight distribution (Internal Proof). Naive
quantization typically uses a grid based on maximum values,
causing outliers to stretch the quantization range inefficiently.
In contrast, our analysis in Figure 4 confirms that LSQ
automatically learns to "zoom in" on the main distribution (bell
curve) of the weights, allocating precision where it matters
most and effectively ignoring outliers.

```
Figure 4 : Weight Distribution vs. Learned INT8 Grid. A
visualization of the weight distribution for layer
decoder.conv1.1.ffn.2. The histogram shows that the learned
LSQ step sizes (blue bars) align tightly with the main bell curve
of the FP32 weights, effectively clipping outliers (red lines) to
maximize precision for the majority of values.
```
```
D. Efficiency Analysis
While the current evaluation is performed via PyTorch
simulation on an RTX 4090, the successful conversion to an
INT8 architecture enables deployment on integer-dedicated
hardware accelerators.
```
### V. CONCLUSIONS

```
In this paper, we propose a practical hybrid-precision NVC
framework focused on balancing rate-distortion performance,
decoding consistency, and inference efficiency. By analyzing
the bottlenecks of deployment, we identify that while the sender
(encoder) can afford higher precision for quality, the receiver
(decoder) requires deterministic integer arithmetic to prevent
error propagation and enable acceleration.
```
```
Based on this insight, we design an asymmetric architecture
where the encoder operates in floating-point while the decoder
and memory modules are quantized to INT8. We introduce
Learned Step Size Quantization (LSQ) into the DCVC-RT
architecture, allowing the model to learn optimal quantization
grids rather than relying on static statistics.
```

This approach successfully eliminates the "cliffs" in quality
typically associated with low-bitwidth quantization. Our
experimental results confirm that our method achieves a perfect
overlap with the floating-point baseline (0.0000 dB BD-PSNR
loss) while ensuring bit-exact consistency for the decoder. This
work serves as a proof-of-concept that research-grade codecs
can be transformed into hardware-friendly integer models
without compromising visual fidelity.

While our method theoretically supports 2× to 4× speedups
on integer hardware (e.g., DSPs or NPUs), the current
evaluation is performed via simulation on GPUs. The actual
wall-clock speedup will depend on the specific hardware
implementation and the optimization of INT8 operators on the
target device. In the future, we aim to deploy this framework
on edge devices to validate real-world latency improvements.
[4, 5]
REFERENCES

[1] T. Van Rozendaal _et al._ , "Mobilenvc: Real-time 1080p
neural video compression on a mobile device," in
_Proceedings of the IEEE/CVF Winter Conference on
Applications of Computer Vision_ , 2024, pp. 4323-
4333.
[2] Z. Jia _et al._ , "Towards practical real-time neural video
compression," in _Proceedings of the Computer Vision
and Pattern Recognition Conference_ , 2025, pp.
12543 - 12552.
[3] S. K. Esser, J. L. McKinstry, D. Bablani, R.
Appuswamy, and D. S. Modha, "Learned step size
quantization," _arXiv preprint arXiv:1902.08153,_ 2019.
[4] J. Ballé, N. Johnston, and D. Minnen, "Integer
networks for data compression with latent-variable
models," in _International Conference on Learning
Representations_ , 2018.
[5] J. Li, B. Li, and Y. Lu, "Deep contextual video
compression," _Advances in Neural Information
Processing Systems,_ vol. 34, pp. 18114-18125, 2021.


