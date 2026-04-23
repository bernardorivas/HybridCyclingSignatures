
## Project goal

We want a pipeline that takes **only hybrid time series data** and produces a **continuous latent trajectory / latent flow surrogate** suitable for downstream topological analysis (Morse graph, Conley index, cycling signatures).

At the moment, we are moving away from a pipeline that assumes explicit knowledge of the hybrid model. The immediate goal is:

$$
\text{time series data} \;\longrightarrow\; \text{continuous suspension-style representation} \;\longrightarrow\; \text{continuous latent flow surrogate}.
$$

The key idea is **not** to learn reset maps, mode logic, or event functions first. Instead, we want to construct a continuous geometric representation directly from the observed time series by attaching suspension bridges across detected jumps.

---

## Core conceptual stance

We are **not** using a fully free latent encoder that bends the entire hybrid state space.

Instead, the philosophy is:

1. **Original/base time-series points should stay fixed.**
2. **Only the appended suspension/cylinder data should be moved by the learned map.**
3. **The discontinuity is resolved by wrapping the added suspension part in an extra dimension, not by globally deforming the base space.**

This is the main difference from more global hybrifold / free-embedding approaches.

---

Perfect — these are the exact spots to fix. I’ll keep your structure and wording, just make the **minimal precise correction**.

Here is the **edited version you should replace with**:

---

## Data model

Suppose the observed time series is

$$  
x_0, x_1, \dots, x_T \in \mathbb{R}^n.  
$$

We do **not** assume access to a known reset map $R$, guard set, or hybrid mode label.

Instead, candidate jumps are detected directly from the data by finite differences. For example, define

$$  
d_k := |x_{k+1} - x_k|.  
$$

If $d_k$ is larger than a chosen threshold, then we treat

$$  
x_k^- := x_k, \qquad x_k^+ := x_{k+1}  
$$

as a detected jump pair.

So the detected jump-pair set is

$$  
\mathcal{J} = {(x_j^-, x_j^+)}_{j=1}^N.  
$$

This is completely data-driven.  
For initial experiments, we may instead use ground-truth event detection from the data-generation process to obtain reliable jump pairs, and later replace this with finite-difference detection.

---

## Augmented suspension-style dataset

We split the data into two parts:

### 1. Base data

These are the original observed samples, viewed as lying on the base section $s=0$:

$$  
\mathcal{D}_{\mathrm{base}} = {(x_t, 0)}_{t=0}^T.  
$$

These correspond to the original hybrid time series, and we want the encoder to preserve them.

### 2. Appended cylinder / suspension data

For each detected jump pair $(x_j^-, x_j^+)$, we introduce a suspension coordinate

$$  
s \in [0,1].  
$$

We sample

$$  
0 = s_0 < s_1 < \cdots < s_M = 1  
$$

and construct the suspended samples by **keeping the base position fixed at $x_j^-$**:

$$  
(x_j^-, s_\ell), \qquad \ell = 0, \dots, M.  
$$

After reaching $s=1$, the trajectory returns to the base section at

$$  
(x_j^+, 0).  
$$

So the augmented dataset is conceptually

$$  
\mathcal{D}_{\mathrm{aug}}
=
\mathcal{D}_{\mathrm{base}}  
\sqcup  
\bigcup_{j=1}^N {(x_j^-, s_\ell)}_{\ell=0}^M.  
$$

Important:

- this is **not** an interpolation between $x^-$ and $x^+$ in the base coordinates,
    
- the base coordinate remains fixed along the suspension segment,
    
- the jump is represented by moving through the auxiliary coordinate $s$, then reattaching at $x^+$,
    
- these appended samples are **auxiliary geometric scaffolding**, not ordinary state observations.
    


---

## Encoder architecture

There are really two different maps.

### A. Base encoder
For original data on the base section, the encoder is fixed to be the identity:

$$
E_{\mathrm{base}}(x,0) = (x,0).
$$

This means:

- points from the original non-appended trajectory are preserved exactly;
- we do **not** want to bend the whole hybrid state space;
- the smooth part of the trajectory remains where it already is.

This is one of the main advantages of using a suspension construction.

### B. Bridge generator
For each detected jump pair $(x^-,x^+)$, the appended bridge is mapped by a learned family of curves

$$
\Gamma_\theta(x^-,x^+,s)
=
\Big(
\Psi_\theta(x^-,x^+,s),\;
\eta_\theta(x^-,x^+,s)
\Big),
\qquad s\in[0,1].
$$

The specific parameterization we want is

$$
\Psi_\theta(x^-,x^+,s)
=
(1-s)x^- + s x^+ + s(1-s)u_\theta(x^-,x^+,s),
$$

$$
\eta_\theta(x^-,x^+,s)
=
s(s-1)\,v_\theta(x^-,x^+,s).
$$

So the full bridge map is

$$
\Gamma_\theta(x^-,x^+,s)
=
\left(
(1-s)x^- + s x^+ + s(1-s)u_\theta(x^-,x^+,s),\;
s(s-1)\,v_\theta(x^-,x^+,s)
\right).
$$

This should be implemented as the main learnable object.

---

## Why this architecture is intentionally easy to learn

This parameterization was chosen on purpose because it hard-codes almost all of the desired geometry.

### Endpoint attachment is automatic
At $s=0$,

$$
\Gamma_\theta(x^-,x^+,0) = (x^-,0).
$$

At $s=1$,

$$
\Gamma_\theta(x^-,x^+,1) = (x^+,0).
$$

So the bridge starts at the detected pre-jump point and ends at the detected post-jump point automatically.

### The first $n$ coordinates can actually move from $x^-$ to $x^+$
This is why the term

$$
(1-s)x^- + sx^+
$$

is essential.

We explicitly rejected an ansatz of the form

$$
x + s(1-s)u_\theta(x,s)
$$

for the first $n$ coordinates, because that would force the endpoints to coincide with the same base point and would not let the bridge move from $x^-$ to $x^+$.

### The extra coordinate leaves and returns to the base automatically
Because

$$
s(s-1)=0 \quad \text{at } s=0,1,
$$

the last coordinate vanishes at the endpoints.

For $0<s<1$, the factor $s(s-1)\neq 0$, so if $v_\theta$ stays positive, then the bridge interior stays off the base section.

This means the bridge naturally “wraps away” from the base and comes back only at the end.

---

## Positivity requirement on the last coordinate

We want the bridge interior to **never return to the base section early**.

So we require

$$
v_\theta(x^-,x^+,s) > 0
\qquad \text{for all } 0<s<1.
$$

A practical implementation is to parameterize

$$
v_\theta = \operatorname{softplus}(w_\theta) + \varepsilon
$$

for some small $\varepsilon > 0$.

Then automatically:

- at $s=0,1$, the last coordinate is zero;
- for $0<s<1$, the last coordinate is strictly nonzero.

Hence the bridge interior cannot come back to the base section before the endpoint.

---

## Four geometric properties we want

The representation should satisfy the following four properties.

### 1. Base fixed
For original data,

$$
E_{\mathrm{base}}(x,0) = (x,0).
$$

So all ordinary time-series points remain unchanged.

### 2. Correct attachment
Each bridge satisfies

$$
\Gamma_\theta(x^-,x^+,0) = (x^-,0),
\qquad
\Gamma_\theta(x^-,x^+,1) = (x^+,0).
$$

### 3. No premature return to base
For each bridge,

$$
\Gamma_\theta(x^-,x^+,(0,1)) \cap \big(\mathbb{R}^n \times \{0\}\big) = \varnothing.
$$

This is enforced architecturally by the sign condition on $v_\theta$.

### 4. No bridge-bridge intersections
For distinct jump pairs $i \neq j$, the interiors of their bridges should not intersect:

$$
\Gamma_i((0,1)) \cap \Gamma_j((0,1)) = \varnothing.
$$

This is the only genuinely hard geometric condition left after the architecture is chosen correctly.

---

## Why these four properties matter

If these four properties hold, then the augmented hybrid trajectory is embedded as a **continuous non-self-intersecting curve/tube** in the latent space.

This is the geometric objective.

At that point, the discontinuous hybrid time series has been converted into a clean continuous object in latent space.

That solves the **representation problem**.

After that, one can learn a continuous latent vector field or time-$\tau$ flow map on this continuous representation.

---

## Minimal-loss philosophy

We do **not** want a large weighted sum of many competing losses if architecture can already enforce the correct geometry.

The design principle is:

- enforce as much geometry as possible by construction,
- use loss only for the remaining hard part.

Since the architecture already enforces:
- base fixed,
- correct endpoint attachment,
- no premature return to base,

the main remaining loss is a **bridge separation loss** to prevent different bridges from intersecting.

---

## Minimal geometric loss

Let bridge $j$ sampled at $s_\ell\in(0,1)$ be denoted

$$
\gamma_{j,\ell}
:=
\Gamma_\theta(x_j^-,x_j^+,s_\ell).
$$

Then a simple separation loss is

$$
\mathcal{L}_{\mathrm{sep}}
=
\sum_{i<j}\sum_{\ell,m}
\Big[m_0 - \|\gamma_{i,\ell} - \gamma_{j,m}\|\Big]_+^2,
$$

where

$$
[a]_+ := \max(a,0)
$$

and $m_0>0$ is a margin.

This penalizes different bridges getting too close.

To avoid unnecessary complexity, add a mild regularizer such as

$$
\mathcal{L}_{\mathrm{reg}}
=
\sum_{j,\ell}
\left(
\|u_\theta(x_j^-,x_j^+,s_\ell)\|^2
+
|w_\theta(x_j^-,x_j^+,s_\ell)|^2
\right).
$$

So the minimal geometric training objective is

$$
\mathcal{L}_{\mathrm{geom}}
=
\lambda_{\mathrm{sep}}\mathcal{L}_{\mathrm{sep}}
+
\lambda_{\mathrm{reg}}\mathcal{L}_{\mathrm{reg}}.
$$

This is intentionally small and interpretable.

---

## Important reduction in difficulty

Because the base section is fixed and bridge interiors are forced to stay off the base section, **bridge-base collisions are automatically impossible**.

So the only remaining bad geometric event is:

$$
\text{bridge } i \text{ intersects bridge } j.
$$

That is why the loss only needs to handle bridge-bridge separation.

---

## What we are not doing

At this stage, we are **not** doing any of the following as the main goal:

- explicit reset-map identification
- explicit guard learning
- event-function learning
- mode segmentation as the primary representation
- a fully unconstrained encoder that globally warps the entire state space

The first target is purely geometric:

> from time series alone, construct a continuous suspension-style representation of the hybrid trajectory.

---

## Relation to the later latent dynamics stage

Once the geometric representation is learned, the next stage is to learn a continuous latent dynamical surrogate, for example:

- a latent vector field $\dot z = f_\phi(z)$ or
- a latent time-$\tau$ flow map $F_\tau(z)$.

But this comes **after** the geometric suspension representation is made clean.

So the implementation should be organized in two stages:

### Stage 1: geometry learning
- detect jump pairs
- build augmented suspension dataset
- learn bridge geometry with fixed base section

### Stage 2: dynamics learning
- fit a continuous latent flow/vector field on the learned continuous representation

---

## Recommended code organization

Claude Code should preserve repo structure as much as possible, but the implementation should clearly separate:

1. **jump detection**
   - thresholding large finite differences
   - returns jump pairs $(x^-,x^+)$

2. **augmented suspension data construction**
   - build base data
   - build bridge samples for each jump pair

3. **bridge parameterization**
   - implement $u_\theta$, $w_\theta$, $v_\theta$
   - implement $\Gamma_\theta$

4. **geometry losses**
   - separation loss
   - small regularizer

5. **debug / plotting tools**
   - plot base points
   - plot bridges
   - inspect whether bridge interiors stay off base
   - inspect whether bridges intersect

The geometry needs to be visually inspectable.

---

## Practical first examples

Start with low-dimensional examples where geometry can be plotted and debugged:

- bouncing ball
- rimless wheel
- thermostat / simple reset system

In these examples we can directly inspect:
- correct jump detection
- bridge attachment
- no premature return to base
- absence of self-intersection
- continuity of the resulting latent trajectory

---

## Open questions

The following remain open and should be kept in mind:

- Is one extra coordinate enough to separate all bridges in practical examples?
- When do we need more than one extra latent dimension?
- How sensitive is jump-pair detection to the threshold?
- What is the best way to fit the continuous latent flow after geometry is learned?
- What theoretical guarantees can eventually be proved for injectivity / non-self-intersection / consistency?

---

## Immediate success criterion

The immediate success criterion is **not** best prediction accuracy.

It is:

> Given only time-series data, construct a suspension-style augmented representation in which the original base samples remain fixed, detected jumps are connected by learned bridges, bridge interiors stay off the base section, and different bridges do not intersect.

If this is achieved, then the hybrid time series has been converted into a continuous geometric object suitable for the next latent-dynamics stage.