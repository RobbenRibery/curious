# End-to-End Formalization: Compute-Efficient Structural Clipping for CISPO

## 1. The Problem: The $O(N^2)$ Memory Wall in Structural RLHF
Working backwards from empirical failures in reasoning tasks reveals that standard CISPO is artificially bottlenecked by **token-agnostic truncation**. 

The standard objective applies a static scalar bound ($\epsilon_{high}$) to the importance weight $r_t(\theta)$ for every token:
$$L_{CISPO}(\theta) = - \mathbb{E} \left[ \text{detach}(\min(r_t(\theta), \epsilon_{high})) \cdot \hat{A}_t \cdot \log \pi_\theta(a_t|s_t) \right]$$

This mechanism clips the gradient of a critical reasoning token (e.g., a mathematical operator) exactly as it would a structural filler word, preventing the active policy ($\pi_\theta$) from rapidly updating critical logical hubs. 

To solve this, the clipping bound must be dynamic ($\epsilon_t$), scaled by the token's structural importance (attention in-degree). However, calculating exact attention saliency requires caching the full $N \times N$ attention matrix across multiple layers during the active RL rollout. In a standard PPO/CISPO training loop, backpropagation already consumes maximum VRAM. Forcing $\pi_\theta$ to maintain an $O(N^2)$ spatial complexity for attention matrices triggers immediate Out-of-Memory (OOM) errors and destroys training throughput.

---

## 2. The Solution: Decoupled & Sub-Linear Saliency
To resolve the bottleneck, we must extract structural saliency without caching $N \times N$ matrices on the active policy. We achieve this through a unified architecture that isolates computation and reduces spatial complexity.

### Component A: Reference Model Offloading (The Systems Shortcut)
Instead of forcing the active policy ($\pi_\theta$) to compute its own saliency, we offload the calculation entirely to the frozen reference model ($\pi_{ref}$). Because the foundational semantic routing of a prompt (the identification of logic gates and numbers) is structurally inherent to the sequence, the attention topology differs negligibly between $\pi_{ref}$ and $\pi_\theta$. We extract all structural metrics exclusively during the $\pi_{ref}$ forward pass (used for KL-divergence), adding zero FLOPs or memory overhead to the active policy.

### Component B: Sub-Linear Saliency Extraction
Within the offloaded $\pi_{ref}$ pass, we utilize one of two highly efficient metrics to quantify the raw structural importance score ($S_t$) for each token without materializing the full attention matrix.

**Path 1: The Architectural Proxy (Key-Vector $L_2$ Norms)**
Tokens that act as structural hubs naturally develop massive Key vectors so future queries can attend to them. We intercept the materialized KV-cache of $\pi_{ref}$ at the final reasoning layers ($L_{top}$) and compute the $L_2$ norm.
* **Spatial Complexity:** $O(1)$ overhead.
* **Metric:** $S_t = ||k_t^{(L_{top})}||_2$

**Path 2: The Exact Algorithmic Solution ($O(N)$ Streaming Accumulators)**
Instead of storing the full matrix, we dynamically accumulate the in-degree score. At each generation step $i$, $\pi_{ref}$ computes a $1 \times i$ attention vector $\mathbf{a}_i$. We initialize a 1D tensor $\mathbf{c}$ of size $N$ and update the running total at each step: $\mathbf{c}_{1:i} \leftarrow \mathbf{c}_{1:i} + \mathbf{a}_i$.
* **Spatial Complexity:** $O(N)$ per head.
* **Metric:** $S_t = \mathbf{c}[t]$

---

## 3. End-to-End Implementation Formalization

The complete pipeline operates in three sequential steps during the post-training rollout.

### Step 1: Forward Pass & Saliency Extraction
During the generation and KL-divergence phase, the reference model $\pi_{ref}$ processes the sequence. We extract the raw structural importance score $S_t$ for every token $t \in [1, T]$ using either the KV-norm proxy or the streaming accumulator.

### Step 2: Dynamic Bound Normalization
To ensure the active policy's variance remains theoretically bounded over the entire sequence, we cannot simply use $S_t$ as the clipping threshold. We must normalize the raw scores to create a multiplier $m_t$, ensuring the mean of the dynamic bounds strictly equals the standard scalar $\epsilon_{high}$.

We calculate the dynamic bound $\epsilon_t$ as follows:
$$m_t = \frac{S_t}{\frac{1}{T} \sum_{j=1}^T S_j}$$
$$\epsilon_t = \epsilon_{high} \cdot m_t$$

*Implementation Note: A lower bound $\delta$ can be applied ($m_t = \max(m_t, \delta)$) to prevent $\epsilon_t$ from collapsing entirely to zero on extreme filler tokens.*

### Step 3: The AD-CISPO Objective Update
With the frozen tensor $\epsilon_t$ successfully derived from $\pi_{ref}$, it is passed to the active policy $\pi_\theta$. The standard scalar is replaced, yielding the final Adaptive and Dense CISPO objective:

$$L_{AD-CISPO}(\theta) = - \mathbb{E} \left[ \text{detach}(\min(r_t(\theta), \epsilon_t)) \cdot \hat{A}_t \cdot \log \pi_\theta(a_t|s_t) \right]$$

**Result:** The active policy seamlessly executes gradient updates where critical tokens possess widened learning corridors, while filler tokens are aggressively clamped. The pipeline strictly maintains standard training throughput by eliminating all $O(N^2)$ tracking requirements.