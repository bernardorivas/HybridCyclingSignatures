# Proposed Section 3 — unified view incorporating Teng et al. (2026)

**Status**: draft for review. Not yet promoted to `chyll.tex`. The current `chyll.tex` is untouched. The intent is to keep our commutative diagram and our six-term loss intact, and add the notational bridge that recasts Sangli's five-term loss as the multi-step / continuous-time parameterization of the same diagram.

**Note on notation in this file**: all custom LaTeX macros from `macros.tex` and `main.tex` (`\Enc`, `\Dec`, `\Dyn`, `\Lat`, `\R`, `\cH`, `\cX`, `\Xr`, `\flowr`, `\hybrifold`, `\susp`, `\setdef`, `\UTB`, `\Kaito`, `\Bernardo`) are expanded to standard LaTeX below so the .md is readable without `macros.tex`. The substitution table:

| Custom macro | Standard LaTeX | Meaning |
|---|---|---|
| `\R` | `\mathbb{R}` | reals |
| `\cH` | `\mathcal{H}` | hybrid system |
| `\cX` | `\mathcal{X}` | sample set |
| `\susp` | `\Sigma_{\mathcal{H}}` | suspension space functor |
| `\Xr` | `X'` | relaxed state space |
| `\flowr` | `\varphi'` | relaxed flow |
| `\hybrifold` | `M_{\mathcal{H}}` | hybrifold |
| `\Enc` | `E_\theta` | encoder |
| `\Dec` | `D_\theta` | decoder |
| `\Dyn` | `F_\theta` | latent dynamics |
| `\Lat` | `Z` | latent space |
| `\setdef{a}{b}` | `\left\{ a \,\middle|\, b \right\}` | set-builder |
| `\UTB` | `\operatorname{UT}` | unit tangent bundle |
| `\Kaito{...}` | `{\color{red}\textbf{Kaito: } ...}` | author comment |
| `\Bernardo{...}` | `{\color{red}\textbf{Bernardo: } ...}` | author comment |

When you promote this to `chyll.tex`, you can either keep the expanded form or substitute the macros back in — the meaning is identical.

---

## Summary of proposed changes

1. **Intro of §3** (after the diagram-implies-semiconjugacy paragraph): add one paragraph noting that $F_\theta$ admits two equivalent parameterizations — a direct MLP, or the time-$\tau$ flow $\exp(\tau V_\theta)$ of a learned vector field.
2. **§3.1 Latent dimension, `rem:teng_compare`**: add one closing sentence pointing forward to the loss-side comparison in §3.4.
3. **§3.2 Architecture**: append a paragraph on the Neural-ODE parameterization.
4. **§3.4 Loss**: append a new paragraph **Relationship to Teng et al. (2026)** with an alignment table. No existing equation removed; no weight changed.
5. **§3.5 Two-phase training, §3.6, §3.7**: unchanged.

Deltas are marked inline below with `% --- DELTA: ... ---` LaTeX comments at their boundaries.

---

## Full proposed §3 (LaTeX source)

```latex
\section{Learning Framework}
\label{sec:chyll}

{\color{red}\textbf{Bernardo: }Perhaps a brief literature review on section
might be useful. Just noticed that I know nothing about past work on
embeddings of quotient spaces / piecewise continuous data}

While the suspension semiflow restores continuity, the resulting quotient
space $\Sigma_{\mathcal{H}}(X)$ is not easily accessible. We solve this
problem by learning a continuous representation of the suspension space and
suspension semiflow via an \emph{autoencoder}, a pair of parametric maps
$(E_\theta, D_\theta)$ trained jointly so that $D_\theta \circ E_\theta$
approximates the identity on a chosen domain, together with a latent
dynamics map $F_\theta$ which approximates the semiflow in the codomain of
$E_\theta$.

There are three maps involved in this process: an encoder
$E_\theta : X' \to Z = \mathbb{R}^d$, a latent dynamics map
$F_\theta : Z \to Z$, and a decoder $D_\theta : Z \to X'$ such that, for a
fixed time step $\tau > 0$, writing $f_\tau(x') = \varphi'(\tau, x')$,
$\iota_0(x) = (x, 0)$, and $\iota_1(g) = (g, 1)$, the following diagram
commutes:
\begin{equation}
  \label{eq:cd}
  \small
  \begin{tikzcd}[column sep=3.0em, row sep=2.4em]
    G \arrow[r, "\iota_1"] \arrow[d, "\iota_0 \circ r"']
    & X' \arrow[r, "f_\tau"] \arrow[d, "E_\theta"']
    & X' \arrow[d, "E_\theta"] \\
    X' \arrow[r, "E_\theta"'] \arrow[rd, "\mathrm{id}_{X'}"']
    & Z \arrow[r, "F_\theta"'] \arrow[d, "D_\theta"']
    & Z \\
    & X'
    \arrow[phantom, from=2-1, to=3-2]
    &
  \end{tikzcd}
\end{equation}
The gluing condition implies that $E_\theta$ is constant on the fibers of
$\pi : X' \to \Sigma_{\mathcal{H}}(X)$, hence
$E_\theta = \bar E \circ \pi$ for a continuous map
$\bar E : \Sigma_{\mathcal{H}}(X) \to Z$. The semi-conjugacy condition
then becomes
\[
  F_\theta \circ \bar E = \bar E \circ \Phi_{\mathcal{H}}^\tau,
\]
where $\Phi_{\mathcal{H}}^\tau$ is the time-$\tau$ map of the suspension
semiflow. When $\bar E$ is injective, it allows one to identify
$\Sigma_{\mathcal{H}}(X)$ with $\bar E(\Sigma_{\mathcal{H}}(X)) \subset Z$.

% --- DELTA 1: NEW PARAGRAPH BELOW ---
The latent dynamics $F_\theta$ in~\eqref{eq:cd} is a map $Z \to Z$ at a
fixed time step $\tau$. Two parameterizations are useful in what follows:
the direct form $F_\theta = \mathrm{MLP}_\theta$ that we use as default
throughout, and the continuous-time form $F_\theta = \exp(\tau\, V_\theta)$,
i.e. the time-$\tau$ flow of an autonomous vector field
$V_\theta : Z \to TZ$ integrated by a Neural ODE solver \cite{Chen2018}. The
two are equivalent in the diagram and we comment on the second form in
Sections~\ref{subsec:architecture} and~\ref{subsec:training} where it
differs from the first.
% --- end DELTA 1 ---

\subsection{Latent dimension}
\label{subsec:embedding}

% {\color{red}\textbf{Bernardo: }Kaito, I don't mind elaborating on this,
% but could you write in a few words how did you choose the dimension of
% the embedding? ...}

The identification of $\Sigma_{\mathcal{H}}(X)$ with
$\bar E(\Sigma_{\mathcal{H}}(X)) \subset \mathbb{R}^d$ requires $\bar E$ to
be a $C^k$-embedding, that is, an injective $C^k$ immersion that is a
homeomorphism onto its image. Such an $\bar E$ realises
$\Sigma_{\mathcal{H}}(X)$ faithfully inside $\mathbb{R}^d$: topological
invariants of $\Sigma_{\mathcal{H}}(X)$ agree with those of
$\bar E(\Sigma_{\mathcal{H}}(X))$, including the first homology $H_1$ used
by the cycling-signature framework of Section~\ref{sec:signatures}. The
latent dimension $d$ must be large enough for such an embedding to exist
at all.

We strengthen the standing assumption on $\mathcal{H}$ accordingly: in
addition to the data of Definition~\ref{defn:HybridSystem}, we assume
that $X$ is a compact $C^k$ manifold with boundary of dimension $n$ for
some $k \geq 1$, that $G \subset \partial X$ and $r(G) \subset \partial X$
are codimension-one $C^k$ submanifolds of $X$ without boundary (otherwise
the cylinder side $\partial G \times [0, 1]$ also contributes to
$\partial X'$), and that the reset map $r : G \to r(G)$ is a $C^k$
diffeomorphism. Under these assumptions, the mapping-cylinder
construction equips the relaxed space $X'$ with the structure of a
compact $C^k$ manifold of dimension $n$ whose boundary is
$\partial X \setminus G$ together with the cylinder top
$G \times \{1\}$. The suspension space $\Sigma_{\mathcal{H}}(X)$, obtained
by further identifying $(g, 1) \sim r(g)$, is a compact $C^k$ manifold of
dimension $n$ whose boundary is $\partial X \setminus (G \cup r(G))$.
Under the Trapping Guard Condition, the suspension semiflow
$\Phi_{\mathcal{H}}$ is continuous on $\Sigma_{\mathcal{H}}(X)$
(Theorem~\ref{thm:SuspensionSemiflow}). Whether $\bar E$ can be made an
embedding is a purely topological question about $\Sigma_{\mathcal{H}}(X)$,
addressed by the Whitney embedding theorem (cf.\ \cite{TengICML2026} for
the parallel argument on the hybrifold).

\begin{thm}[Whitney embedding theorem,
            {\cite[Chapter~2, Theorems~2.13 and 2.14, with §3 for the boundary case]{Hirsch2012}}]
\label{thm:weak_whitney}
Let $M$ be a compact $C^k$ manifold of dimension $n$ (possibly with
boundary), $1 \leq k \leq \infty$. For every integer $d \geq 2n + 1$,
the set of $C^k$ embeddings $M \hookrightarrow \mathbb{R}^d$ is dense in
$C^k_S(M, \mathbb{R}^d)$. In particular, $M$ is $C^k$ diffeomorphic to a
closed submanifold of $\mathbb{R}^{2n+1}$.
\end{thm}

Applied to $\Sigma_{\mathcal{H}}(X)$, Theorem~\ref{thm:weak_whitney}
yields the sufficient bound
\begin{equation}
  \label{eq:whitney-bound}
  d \geq 2n + 1
\end{equation}
on the latent dimension, with $C^k$ embeddings
$\Sigma_{\mathcal{H}}(X) \hookrightarrow \mathbb{R}^d$ dense in
$C^k_S(\Sigma_{\mathcal{H}}(X), \mathbb{R}^d)$ at every such $d$. The bound
is uniform in the underlying $n$-manifold and not necessary in general,
since a particular $\Sigma_{\mathcal{H}}(X)$ may embed in lower dimension.
For our examples it reads $d \geq 5$ for the rimless wheel ($n = 2$) and
$d \geq 9$ for the compass-gait ($n = 4$).

Given an embedding $\bar E$, the pushforward
\[
  \widetilde\Phi_{\mathcal{H}}(\tau, \bar E(z))
    \coloneqq \bar E(\Phi_{\mathcal{H}}(\tau, z))
    \qquad z \in \Sigma_{\mathcal{H}}(X),\; \tau \geq 0
\]
is automatically a continuous semiflow on
$\bar E(\Sigma_{\mathcal{H}}(X)) \subset \mathbb{R}^d$, since
$\Phi_{\mathcal{H}}$ is continuous on $\Sigma_{\mathcal{H}}(X)$. With
$\Sigma_{\mathcal{H}}(X)$ given the metric pulled back from $\mathbb{R}^d$
via $\bar E$, the cycling signature computed on the embedded image
$\bar E(\Sigma_{\mathcal{H}}(X))$ coincides with the one of
Definition~\ref{defn:HybridCyclingSignature} on $\Sigma_{\mathcal{H}}(X)$
itself. The relationship between $\Sigma_{\mathcal{H}}(X)$ and the
hybrifold $X / {\sim}$ is discussed in Remark~\ref{rem:teng_compare} below.

In Section~\ref{sec:results} we exercise two settings: the minimal
$d = n + 1$, which keeps the encoder square on the relaxed coordinates
$(x, s)$ and sits strictly below \eqref{eq:whitney-bound}, and the
extended $d = 2n + 1$ at the bound. Both choices empirically recover the
$H_1$ cycling signature of $\Sigma_{\mathcal{H}}(X)$ on both example
systems. We do not prove that the trained encoders are embeddings at
$d = n + 1$, only that the cycling-signature output is consistent with
such an embedding on the trajectories tested. At $d = 2n + 1$ the bound
\eqref{eq:whitney-bound} is satisfied, and we observe additionally that
the round-trip $D_\theta \circ E_\theta \approx \mathrm{id}_{X'}$
tightens on the compass-gait example.
{\color{red}\textbf{Kaito: }point to the embed9 numbers in
Section~\ref{sec:results} once those are written.}

\begin{rem}[Comparison with \cite{TengICML2026}]
\label{rem:teng_compare}
A closely related embedding result for hybrid systems was obtained
recently in \cite{TengICML2026}, working on the hybrifold
$M_{\mathcal{H}} = X / {\sim}$ of \cite{Simic2000}: the quotient of $X$
that directly identifies each $g \in G$ with its post-reset image
$r(g) \in r(G)$. The hybrid flow descends to a continuous trajectory map
on $M_{\mathcal{H}}$ in the quotient topology, since the pre- and
post-reset states $g$ and $r(g)$ are collapsed to a single equivalence
class through which the trajectory passes continuously in time. The
corresponding hybrid vector field on $M_{\mathcal{H}}$ has a directional
discontinuity at the gluing stratum, where the left-derivative $V(g)$
and the right-derivative $V(r(g))$ generically differ. Beyond the
classical existence of an embedding, their main contribution is that for
$m > 2n$ the embedding $M_{\mathcal{H}} \hookrightarrow \mathbb{R}^m$
can be chosen so that its image admits a continuous vector field
extending the hybrid dynamics. This property is shown to be generic
among $C^k$ embeddings via a parametric transversality argument.

We follow \cite{Kvalheim2021} instead and use the suspension space
$\Sigma_{\mathcal{H}}(X)$, in which the gluing $g \sim r(g)$ is replaced
by a unit-time traversal of the cylinder $G \times [0, 1]$. The two
quotients are homotopy equivalent. The suspension construction provides
$\Phi_{\mathcal{H}}$ as a continuous semiflow on a compact metric space
under the Trapping Guard Condition, which is the working setting
introduced in \cite{Kvalheim2021} for the Conley-theoretic analysis of
hybrid systems and which we adopt here.

% --- DELTA 2: NEW SENTENCE BELOW ---
The relationship between our learning framework and theirs is addressed
in Section~\ref{subsec:training}: their continuous latent vector field
$V_\theta$ is a parameterization of our discrete
$F_\theta = \exp(\tau\, V_\theta)$, and their five-term loss is a
multi-step / continuous-time realisation of our diagram
relations~\eqref{eq:diagram-relations}
(Table~\ref{tab:teng-alignment}).
% --- end DELTA 2 ---

{\color{red}\textbf{Kaito: }The precise topological advantage of
$\Sigma_{\mathcal{H}}(X)$ over $M_{\mathcal{H}}$ for the cycling-signature
setting deserves a closer look. Kvalheim's primary motivation was Conley
index theory and we have not verified that the same dependence arises
here. A clean statement of why $\Sigma_{\mathcal{H}}(X)$ is preferred for
cycling signature (beyond consistency with \cite{Kvalheim2021}) is
currently missing.}
\end{rem}

\subsection{Architecture}
\label{subsec:architecture}

Each of $E_\theta$, $D_\theta$, $F_\theta$ is a fixed-depth multilayer
perceptron with GELU nonlinearities in residual form around the natural
skip between the relevant input and output spaces:
\[
  E_\theta(x') = \mathrm{pad}_d(x') + \mathrm{MLP}_\theta^{E_\theta}(x'),
  \quad
  D_\theta(z) = \pi_{n+1}(z) + \mathrm{MLP}_\theta^{D_\theta}(z),
  \quad
  F_\theta(z) = z + \mathrm{MLP}_\theta^{F_\theta}(z),
\]
where $\mathrm{pad}_d : \mathbb{R}^{n+1} \hookrightarrow \mathbb{R}^d$
zero-pads to the latent dimension and
$\pi_{n+1} : \mathbb{R}^d \to \mathbb{R}^{n+1}$ projects onto the first
$n + 1$ coordinates. The final linear layer of
$\mathrm{MLP}_\theta^{E_\theta}$ is initialised with small weights, so that
$E_\theta$ starts close to the (zero-padded) identity. Parameters are fit
by minibatched gradient descent on the loss defined below, using AdamW
with a cosine learning-rate schedule.

% --- DELTA 3: NEW PARAGRAPH BELOW ---
The continuous-time parameterization $F_\theta = \exp(\tau\, V_\theta)$
is realised by an MLP $V_\theta : Z \to Z$ representing the autonomous
vector field, integrated by an adjoint Neural ODE solver \cite{Chen2018};
this is the form adopted in \cite{TengICML2026}. In
Section~\ref{sec:results} we report the direct form as the primary
baseline and the continuous form as a head-to-head comparison; the two
parameterizations of $F_\theta$ are interchangeable in the diagram, but
the loss formulation in Section~\ref{subsec:training} takes a multi-step
form under the continuous parameterization
(Table~\ref{tab:teng-alignment}).
% --- end DELTA 3 ---

\subsection{Sampling}
\label{subsec:sampling}

% {\color{red}\textbf{Bernardo: }We should specify how we sample for the
% suspension space embedding and dynamics. ...}
% {\color{red}\textbf{Bernardo: }It might be useful to say that, for a
% fixed $\tau>0$, the sampled time-$\tau$ training pairs do not contain
% both $(g,1)$ and $(r(g),0)$ as consecutive samples.}

We optimise $(E_\theta, F_\theta, D_\theta)$ on a finite sampling
$\mathcal{X}' \subset X'$ of the relaxed space. The sampling is built
from a collection of initial conditions in $X'$, half drawn uniformly
from the base space at $s = 0$ and half from the cylinder $G \times [0, 1]$
by sampling $g \in G$ and $s \in (0, 1)$. From each initial condition
$x'_0$ we integrate the relaxed semiflow forward for $K$ steps of size
$\tau$, producing an orbit $(x'_0, x'_1, \ldots, x'_K)$ in $X'$, and
collect the $K$ consecutive pairs $(x'_i, x'_{i+1}) = (x'_i, f_\tau(x'_i))$
as supervised samples for the dynamics term. The guard sampling
$\mathcal{R} \subseteq G$ used by the gluing and seam terms is drawn from
the same guard distribution as the cylinder half. For a fixed $\tau$, the
sampled pairs do not contain both $(g, 1)$ and $(r(g), 0)$ as consecutive
elements: the cylinder ends and the next arc starts are identified by
the gluing condition rather than by step-$\tau$ integration.

\subsection{Loss}
\label{subsec:training}

% {\color{red}\textbf{Bernardo: }below a short paragraph regarding loss
% functions.}

The total loss is a weighted combination of six terms,
\begin{equation}
  \label{eq:loss_total}
  \mathcal{L}_{\mathrm{total}}
  = \underbrace{
      w_{\mathrm{dyn}}\mathcal{L}_{\mathrm{dyn}}
      + w_{\mathrm{glue}}\mathcal{L}_{\mathrm{glue}}
      + w_{\mathrm{rec}}\mathcal{L}_{\mathrm{rec}}
    }_{\text{diagram relations}}
  + \underbrace{
      w_{\mathrm{seam}}\mathcal{L}_{\mathrm{seam}}
      + w_{\mathrm{conf}}\mathcal{L}_{\mathrm{conf}}
      + w_{\mathrm{coll}}\mathcal{L}_{\mathrm{coll}}
    }_{\text{geometric regularizers}},
\end{equation}
with $w_i \geq 0$. The first three are empirical squared residuals of
the commutative diagram~\eqref{eq:cd},
\begin{equation}
  \label{eq:diagram-relations}
  F_\theta \circ E_\theta = E_\theta \circ f_\tau,
  \qquad
  E_\theta \circ \iota_1 = E_\theta \circ \iota_0 \circ r,
  \qquad
  D_\theta \circ E_\theta = \mathrm{id}_{X'},
\end{equation}
written out as
\[
  \begin{aligned}
    \mathcal{L}_{\mathrm{dyn}}
      &= \sum_{x' \in \mathcal{X}'}
         \|F_\theta(E_\theta(x')) - E_\theta(f_\tau(x'))\|^2, \\
    \mathcal{L}_{\mathrm{glue}}
      &= \sum_{g \in \mathcal{R}}
         \|E_\theta(g, 1) - E_\theta(r(g), 0)\|^2, \\
    \mathcal{L}_{\mathrm{rec}}
      &= \sum_{x' \in \mathcal{X}'_\varepsilon}
         \|D_\theta(E_\theta(x')) - x'\|^2,
  \end{aligned}
\]
where $\mathcal{X}'_\varepsilon
  = \left\{ x' \in \mathcal{X}' \,\middle|\,
            s(x') = 0 \text{ or }
            \varepsilon < s(x') < 1 - \varepsilon \right\}$
is a boundary-masked subset of $\mathcal{X}'$ excluding a small
$\varepsilon$-neighbourhood of the gluing identification
$(g, 1) \sim (r(g), 0)$. There the two pre-images $(g, 1)$ and
$(r(g), 0)$ are sent to the same latent by $\mathcal{L}_{\mathrm{glue}}$,
so requiring $D_\theta \circ E_\theta = \mathrm{id}_{X'}$ on those points
would force the decoder to fight the gluing.

The last three terms constrain geometric properties of $E_\theta$ on $Z$
that are not implied by~\eqref{eq:diagram-relations} but that proved
necessary in practice for the cycling-signature comparisons of
Section~\ref{sec:signatures}: $C^1$ tangent continuity of the lift at
every impact ($\mathcal{L}_{\mathrm{seam}}$), conformality of the
encoder Jacobian ($\mathcal{L}_{\mathrm{conf}}$), and a per-coordinate
variance floor preventing latent collapse ($\mathcal{L}_{\mathrm{coll}}$).

Unlike the hybrifold construction in \cite{TengICML2026}, this setup
imposes the quotient relation only at the level of the encoder via
$\mathcal{L}_{\mathrm{glue}}$.
% {\color{red}\textbf{Bernardo: }although we could}
We do not modify the architecture to be differentiable across the
gluing. Instead we enforce \emph{tangent} continuity along the lifted
trajectory in $Z$ at every impact, through the seam term below.

\paragraph{Seam-tangent matching.}
At each impact $g \in G$ the lifted trajectory in $Z$ transitions
between two regimes: arc evolution under the encoder Jacobian acting on
the base vector field, $\nabla_x E_\theta \cdot v_X$, and cylinder
evolution along the $s$-direction derivative $\partial_s E_\theta$. For
the lift to be $C^1$ across the seam, as required for the unit tangent
bundle lift of Section~\ref{sec:signatures}, these two latent tangent
directions must agree on each side of every impact. We enforce this
directly:
\begin{equation}
  \label{eq:loss_seam}
  \mathcal{L}_{\mathrm{seam}}
  = \sum_{g \in \mathcal{R}}
    \Big[
      \delta\big(\partial_s E_\theta(g, 0),\;
                 \nabla_x E_\theta(g, 0)\, v_X(g)\big)
      + \delta\big(\partial_s E_\theta(g, 1),\;
                   \nabla_x E_\theta(r(g), 0)\, v_X(r(g))\big)
    \Big],
\end{equation}
where $\delta(u, v) = 1 - \langle u, v \rangle / (\|u\|\, \|v\|)$ is the
cosine distance. The first summand controls the pre-impact seam at
$s = 0$ (arc end joining cylinder bottom at $g$), and the second
controls the post-impact seam at $s = 1$ (cylinder top joining the next
arc start at $r(g)$ via the gluing identification). The base vector
field $v_X$ is the continuous component of $\varphi'$. The
$\partial_s E_\theta$ derivatives are computed by finite differences
along the cylinder coordinate, and the
$\nabla_x E_\theta \cdot v_X$ directional derivatives by automatic
differentiation through $E_\theta$. Without this term the encoder
Jacobian on the base is unconstrained and the lift develops a tangent
discontinuity at every impact, empirically up to $145^\circ$, which
corrupts the unit tangent bundle in which cycling signatures are
compared.

\paragraph{Conformal regularizer.}
A soft constraint pushing the encoder Jacobian toward a similarity
transform at each sample,
\begin{equation}
  \label{eq:loss_conf}
  \mathcal{L}_{\mathrm{conf}}
  = \sum_{x' \in \mathcal{X}'}
    \big\| \lambda(x')\, I_{n+1} - J(x')^\top J(x') \big\|_F^2,
  \qquad
  \lambda(x') = \frac{1}{n+1}
                 \mathrm{tr}\big(J(x')^\top J(x')\big),
\end{equation}
with $J(x') = \nabla_{x'} E_\theta(x')$. This term keeps the encoder
close to a bi-Lipschitz embedding by penalising deviation of $J^\top J$
from a scalar multiple of the identity at each sample. We treat it as a
soft prior rather than a hard constraint, since strict conformality
conflicts with the cylinder geometry near the seam.

\paragraph{Anti-collapse.}
A per-coordinate variance floor preventing the encoder from contracting
onto a low-dimensional subset of $Z$,
\begin{equation}
  \label{eq:loss_coll}
  \mathcal{L}_{\mathrm{coll}}
  = \sum_{k=1}^{d}
    \mathrm{ReLU}\big(\Lambda -
                      \mathrm{Var}_{x' \in \mathcal{X}'} E_\theta(x')_k\big),
  \qquad \Lambda > 0.
\end{equation}
Without this floor the encoder can satisfy $\mathcal{L}_{\mathrm{dyn}}$
and $\mathcal{L}_{\mathrm{glue}}$ trivially by driving one or more
latent coordinates to near-constant, which defeats the purpose of $Z$.

% --- DELTA 4: NEW PARAGRAPH + TABLE BELOW ---
\paragraph{Relationship to Teng et al. (2026).}
\label{par:teng-loss-alignment}
The loss in \cite{TengICML2026} is a five-term composite that fits
inside our diagram~\eqref{eq:cd} under the parameterization
$F_\theta = \exp(\tau\, V_\theta)$ described in
Section~\ref{subsec:architecture}. With this identification, writing
$\hat z_k = F_\theta^k\, E_\theta(x'_0)$ for the $k$-th rolled-out latent
from an initial sample $x'_0$, their latent-consistency and
reconstruction terms are the multi-step accumulations
\begin{equation}
  \label{eq:teng-multistep}
  \mathcal{L}_z^{(T)}
    = \sum_{k=1}^{T-1}
      \big\|\hat z_k - E_\theta(f_\tau^k(x'_0))\big\|^2,
  \qquad
  \mathcal{L}_x^{(T)}
    = \sum_{k=0}^{T-1}
      \big\|D_\theta(\hat z_k) - f_\tau^k(x'_0)\big\|^2,
\end{equation}
which reduce to $\mathcal{L}_{\mathrm{dyn}}$ and
$\mathcal{L}_{\mathrm{rec}}$ at $T = 1$ (with a one-step rollout in the
first case and a pointwise reconstruction in the second). Their gluing
and variance-floor terms coincide with $\mathcal{L}_{\mathrm{glue}}$ and
$\mathcal{L}_{\mathrm{coll}}$ verbatim, and their velocity-compatibility
term plays the role of $\mathcal{L}_{\mathrm{seam}}$ with the cosine
distance $\delta$ replaced by a squared $L^2$ residual. The conformal
regularizer $\mathcal{L}_{\mathrm{conf}}$ has no counterpart in
\cite{TengICML2026}; we retain it for its role in the bi-Lipschitz
stability of the cycling signature (Section~\ref{sec:signatures}). The
rollout-horizon curriculum of \cite{TengICML2026}, which grows $T$
during training, is therefore a schedule on $\mathcal{L}_z^{(T)}$ and
$\mathcal{L}_x^{(T)}$ that interpolates between our single-step diagram
residuals and a fully integrated trajectory consistency.
Table~\ref{tab:teng-alignment} summarises the correspondence.

\begin{table}[h]
\centering
\caption{Loss-term correspondence between our framework and
\cite{TengICML2026} under the parameterization
$F_\theta = \exp(\tau\, V_\theta)$. $T$ denotes the rollout horizon.}
\label{tab:teng-alignment}
\small
\begin{tabular}{lll}
\toprule
Ours (Section~\ref{subsec:training}) & \cite{TengICML2026} & Relationship \\
\midrule
$\mathcal{L}_{\mathrm{dyn}}$
  & $\mathcal{L}_z$
  & multi-step accumulation, $T = 1$ recovers ours \\
$\mathcal{L}_{\mathrm{rec}}$
  & $\mathcal{L}_x$
  & evaluated along rollout, $T = 1$ recovers ours \\
$\mathcal{L}_{\mathrm{glue}}$
  & $\mathcal{L}_g$
  & identical on the sampled guard $\mathcal{R}$ \\
$\mathcal{L}_{\mathrm{seam}}$
  & $\mathcal{L}_v$
  & same tangent agreement; cosine vs squared $L^2$ \\
$\mathcal{L}_{\mathrm{coll}}$
  & $\mathcal{L}_c$
  & identical \\
$\mathcal{L}_{\mathrm{conf}}$
  & --- (dropped in 2026)
  & retained here; see Section~\ref{sec:signatures} \\
\bottomrule
\end{tabular}
\end{table}
% --- end DELTA 4 ---

\subsection{Two-phase training}
\label{subsec:training_phases}

Training proceeds in two phases.

\textbf{Phase I.} Train $E_\theta$ and $F_\theta$ jointly on the encoder
side of the diagram together with the geometry regularizers,
$\mathcal{L}_{\mathrm{dyn}}
 + \mathcal{L}_{\mathrm{glue}}
 + \mathcal{L}_{\mathrm{seam}}
 + \mathcal{L}_{\mathrm{conf}}
 + \mathcal{L}_{\mathrm{coll}}$,
holding $D_\theta$ free.

\textbf{Phase II.} Freeze $E_\theta$ and $F_\theta$. Train $D_\theta$
alone on the boundary-masked reconstruction term
$\mathcal{L}_{\mathrm{rec}}$.

This split isolates encoder geometry from decoder inversion: Phase I
shapes the encoder purely by the diagram and the latent-geometry
regularizers, and Phase II then learns $D_\theta$ as the best inverse
of the now-fixed $E_\theta$ on the boundary-masked support. We observed
that joint optimisation of all six terms makes the decoder fight
$\mathcal{L}_{\mathrm{glue}}$ near the gluing boundary, which the
boundary mask in $\mathcal{X}'_\varepsilon$ resolves.

\subsection{Cycling Signatures in the Latent Space}
\label{subsec:cycling_latent}

{\color{red}\textbf{Bernardo: }This is where we should make a choice
between computing cyclic signatures just from the enconding of points
or if we're using finite differences to compute in $\operatorname{UT}(Z)$}

\subsection{Open questions regarding approximations?}
\label{subsec:approximation}

{\color{red}\textbf{Bernardo: }As I already wrote in
Section~\ref{subsec:hybridcyc}, there is some theoretical context into
why all computations work and why it's ok to allow errors in
approximations. I think this section should outline the numerical
perspectives of those assertions (if they exist).}
```

---

## Rationale and review notes

**On the unification claim.** The substantive content is the identification $F_\theta = \exp(\tau V_\theta)$. Once granted, the rest is bookkeeping:

- $\mathcal L_z$ and $\mathcal L_x$ are sums over a rolled-out trajectory; setting $T = 1$ reduces them to our single-step relations.
- $\mathcal L_g$ and $\mathcal L_c$ are syntactically the same expressions in both papers.
- $\mathcal L_v$ and $\mathcal L_\text{seam}$ enforce the same geometric condition with different metrics. On the mapping cylinder both reduce to comparing $\partial_s E_\theta(g, 1)$ and $\nabla_x E_\theta(r(g), 0)\cdot v_X(r(g))$. We use cosine because cycling signatures are computed in the unit tangent bundle, where direction is the load-bearing structure.
- $\mathcal L_\text{conf}$ is ours alone, and we cite the bi-Lipschitz / cycling-signature motivation for keeping it.

**On the diagram.** Unchanged. The unification happens entirely at the loss-and-parameterization layer.

**On the code.** Unchanged. The `src/` pipeline uses the direct $F_\theta = \mathrm{MLP}$ form. The `chyll_v2/` pipeline uses the continuous $F_\theta = \exp(\tau V_\theta)$ form. Both fit the same diagram.

**Potential concerns**:

- *The multi-step relationship is approximate, not exact.* Strictly, $\mathcal L_z^{(T)}$ enforces commutativity at $T - 1$ specific timesteps; $\mathcal L_\text{dyn}$ enforces it on every sampled pair in $\mathcal X'$. If the sampling covers all $\tau$-step pairs along trajectories, they are equivalent in expectation; if it does not, $\mathcal L_z^{(T)}$ may emphasise some regions over others. Probably fine to gloss over for a first reading.
- *The metric choice in $\mathcal L_v$ vs $\mathcal L_\text{seam}$.* I say "cosine vs squared $L^2$". Should I expand on why cosine is preferred for cycling signatures, or leave it cited?
- *Whether to include the comparison table.* I think it earns its space — it's the cleanest way to make the unification visible at a glance — but the prose paragraph alone is enough if you'd rather drop the table.
- *Whether the new paragraph belongs at the end of §3.4 (current placement) or as a new §3.5 of its own.* End of §3.4 keeps the comparison adjacent to the loss it relates to; a separate subsection gives it more prominence. I went with the former since it stays in line with the existing "diagram relations + geometric regularizers" arc.
- *The forward pointer added to `rem:teng_compare`.* That remark currently focuses on topology (suspension vs hybrifold). I added one sentence at the end pointing readers to §3.4 for the learning-framework side. If you'd rather keep the remark purely topological, drop that sentence.

**Things I deliberately did not change**:

- The total-loss equation~\eqref{eq:loss_total}, the six-term expansion, or any weight $w_i$.
- The seam paragraph and the $145^\circ$ empirical claim.
- The two-phase training prose.
- The standing assumption block in §3.1 or the Whitney embedding theorem statement.

---

## What I'd like you to weigh

1. Is the unification narrative the one you want, or would you rather position Sangli as a contemporary parallel rather than a special case of our framework?
2. The cosine-vs-MSE point in $\mathcal L_v / \mathcal L_\text{seam}$ — keep it as a one-line aside or expand into a small paragraph with the cycling-signature motivation made explicit?
3. The alignment table — keep, drop, or move to an appendix?
4. The forward pointer added to `rem:teng_compare` — keep or drop?
5. Anything I should *remove* from the current §3 to make space?
