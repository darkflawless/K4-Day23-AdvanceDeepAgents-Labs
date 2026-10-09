# World Models in Artificial Intelligence: Foundations, Generative Simulators, Embodied Agents, and the Road to General Intelligence

## Introduction

A world model is a learned internal model of an environment that supports prediction of future observations, prediction of the consequences of one's own actions, and the use of those predictions for planning and control. In contemporary AI, that model is almost always a neural network trained on passive observation — video, gameplay, sensor streams — rather than a hand-authored physics engine. Since Ha and Schmidhuber's 2018 paper that gave the term its modern meaning [1][2], the field has grown from toy recurrent networks trained on Car Racing to 10-billion-parameter generative simulators of playable worlds [3] and families of "world foundation models" built for robotics and driving [4].

This report surveys the field along four axes. It begins with foundations and taxonomy — what a world model is, and the three architectural paradigms that dominate. It then examines the decisive 2023–2025 development: the use of large generative video and token models as interactive simulators, and the accompanying latency and memory problems. It moves to embodied applications — model-based reinforcement learning, real robots, autonomous driving — and the emerging recognition that evaluation, not generation, is the bottleneck. It closes with the AGI thesis, the JEPA-style latent-prediction program, and the increasingly quantitative critique that visual realism is not evidence of physical or causal understanding.

## TL;DR

- The canonical definition is Ha and Schmidhuber's: a compact, unsupervised learned model (a V–M–C pipeline of encoder, recurrent dynamics, and controller) that can be "dreamed in" to train a policy without interacting with the real environment [1][2]. Nearly all later work is a variation on which of those three components is scaled and in what space prediction happens.
- Three architectural paradigms now coexist: recurrent latent state-space models rolled out for imagination (PlaNet → Dreamer → DreamerV3 → Dreamer 4) [5][6][7][8], representation-space prediction without reconstruction (JEPA) [9][10], and video/token-based generative simulators (Genie, GameNGen, Cosmos, Matrix-Game) [3][11][4][12].
- 2024–2025 was dominated by the generative-video-as-world-simulator wave, plus a strong counter-current of benchmarks showing that physics adherence does not follow from visual fidelity: PhyGenBench [13], WorldModelBench [14], PhyWorldBench [15], World-in-World [16], and WorldGym [17] all point the same way.
- Evaluation has fractured into at least three incommensurable axes — closed-loop task success, action controllability/fidelity, and physical plausibility — with a repeated finding that perceptual metrics (FID/FVD) correlate poorly with functional utility [13][16][14][15][17].
- The "world models → AGI" thesis remains programmatic rather than demonstrated. Video generators achieve visual realism without reliable physical-law compliance [18][19][20][21], and even scaling-based "emergence" claims have been argued to be metric artifacts [22].

## Background: What Counts as a World Model?

### The 2018 anchor and its three components

Ha and Schmidhuber's "World Models" trains a model of an environment from raw pixels in an unsupervised fashion and then trains a controller inside the model's own generated "dream." The architecture separates a spatial encoder (V), a recurrent dynamics model (M) that predicts the next latent, and a controller (C) — so the agent's policy never sees the real environment during training. The companion paper "Recurrent World Models Facilitate Policy Evolution" shows the same learned model can host evolved policies, establishing the dream-training loop as a general recipe [1][2]. Two properties of this framing have proven durable and contested: the world model is *learned* rather than authored, and it is *generative* — it can roll forward without the environment.

### Three paradigms

Subsequent work diverges along two questions: should the world model reconstruct observations or only predict in a learned representation space, and should dynamics be recurrent-latent or token/video-generative?

The **latent state-space** family compresses observations into a low-dimensional latent and rolls out imagined trajectories there. PlaNet learned latent dynamics directly from images for planning [5]; Dreamer learned behaviors by latent imagination [6]; DreamerV3 reported mastering diverse domains with a single fixed hyperparameter configuration [7]. MuZero represents the planning-oriented extreme: a learned model of environment dynamics used purely for lookahead search, with strong results across Atari, Go, Chess, and Shogi [23].

The **reconstruction-free prediction** family, rooted in the Joint-Embedding Predictive Architecture (JEPA), argues that reconstruction wastes capacity on visual detail that is irrelevant — or actively harmful — to control. MC-JEPA jointly learns motion and content features in representation space as a concrete instantiation [9], and later work reframes visual representation learning itself as world modeling by predicting global photometric transformations [10]. Applications to driving with LiDAR data [24] and to policy representation learning [25] show the paradigm extending beyond ImageNet-style benchmarks.

The **video/token generative** family treats the world model as a learned simulator that emits future frames or tokens, conditioned on actions. Genie generates action-controllable, playable environments from unlabeled video using spatiotemporal tokenization, a latent action model, and autoregressive dynamics [3]; UniSim proposes learning a universal simulator of real-world interaction from heterogeneous video data with zero-shot transfer [26]. This is the paradigm that absorbed most of the field's attention and compute in 2024–2025.

A fourth substrate deserves separate mention: explicit **3D/4D scene representations** (neural fields, occupancy, factored scene graphs). The "3D and 4D World Modeling" survey establishes definitions, taxonomy, datasets, and metrics for this branch, positioning geometric structure as an alternative substrate to latents and pixels [27].

### Taxonomy: understanding versus predicting

Survey literature has converged on a functional taxonomy. "Understanding World or Predicting Future?" explicitly separates a world model's role in understanding dynamics from its role in predicting the future, and organizes applications across generative games, autonomous driving, robotics, and social simulation [28]. A companion survey for embodied AI structures the field by taxonomy, data resources, metrics, and challenges across simulation, prediction, and decision-making [29], and a further line frames world models as internal environmental representations for cognitive agents operating under compute and bandwidth constraints [30]. A 2025 roadmap argues video generation is evolving toward foundation models that unify world simulation and rendering into physically plausible, interactive video [31].

The practical consequence of these taxonomies is that "world model" is now an umbrella term covering systems as different as a 20-million-parameter RSSM and a diffusion transformer trained on internet video. Definitional scope is itself a live dispute: surveys disagree on whether every predictive model — including LLMs used as implicit text-based simulators — counts, and on how sharply "understanding" can be separated from "predicting" [28][32].

## Generative Video World Models and Interactive Simulators

### Sora and the rebranding of video generation

OpenAI's Sora (February 2024) popularized the claim that large-scale video generation models act as world simulators, using a diffusion-transformer backbone with variable resolution and duration. The research response was immediate and bifurcated. A technical review catalogued Sora's background, technology, limitations, and opportunities as a large vision model [33], while a dedicated survey asked directly "Is Sora a World Simulator?" and reviewed general world models across video generation, autonomous driving, and agents [34]. This pair of papers marks the moment the "video generation ≈ world model" hypothesis entered the mainstream literature as a claim to be tested rather than assumed.

### Diffusion as a real-time engine: GameNGen and its relatives

GameNGen (August 2024) demonstrated a diffusion model functioning as a real-time interactive game engine for DOOM: next-frame prediction is conditioned on the sequence of past frames and actions, the system runs at roughly 20 frames per second on a single TPU, and human raters are only slightly better than chance at distinguishing short clips of the simulation from the real game [11]. This established that diffusion models can serve as real-time neural engines — and immediately raised the latency question that has organized the subfield since: per-frame diffusion is expensive, and interactivity imposes a hard budget. Related work on Atari argued that visual detail matters for world-model quality [35].

### Autoregressive token models: iVideoGPT and MineWorld

An alternative to diffusion is autoregressive prediction over discrete tokens with explicit action and reward conditioning. iVideoGPT is a scalable multimodal autoregressive transformer that integrates visual observations, actions, and rewards, enabling video prediction, planning, and RL through an interactive tokenizer and compressive tokenization [36]. MineWorld applies the same recipe to Minecraft as a real-time, open-source interactive world model built on a visual–action autoregressive Transformer, reportedly outperforming diffusion baselines on that task [37].

### The push to real-time streaming: Matrix-Game and latent-increment simulators

The Matrix-Game line is the clearest expression of the latency problem. "The Matrix" (December 2024) targeted infinite-horizon world generation with real-time moving control [38]; Matrix-Game (June 2025) presented an interactive world foundation model trained in two stages to produce action-controllable, physically consistent Minecraft video [12]; Matrix-Game 2.0 (August 2025) reframed the goal explicitly as open-source, real-time, streaming interaction using few-step autoregressive diffusion to cut inference latency [39]. The trajectory here is instructive: the field did not abandon diffusion for quality reasons but engineered it toward few-step, causal, streaming generation for interactivity reasons.

### World foundation models for physical AI: NVIDIA Cosmos

NVIDIA's Cosmos platform is a family of world foundation models built explicitly for Physical AI — robotics and autonomous driving — supplied with a video curation pipeline, pretrained world foundation models, examples of post-training, and video tokenizers so that practitioners can build customized world models [4]. Cosmos-Transfer1 added conditional world generation with adaptive multimodal control from segmentation, depth, and edge signals for real-time simulation [40], and Cosmos-Drive-Dreams used Cosmos-derived models to synthesize high-fidelity driving data, improving rare-case coverage for 3D detection and policy learning [41]. Cosmos is significant less for a single benchmark result than for productizing the "world model as data engine" thesis: the primary use of the simulator is generating training data, not replacing the environment.

### Action-conditioned game generation and interactive environments

A dense cluster of 2024–2025 work extended action-conditioned interactive video using diffusion transformers and instruction tuning. GameGen-X generates and interactively controls open-world game video via two-stage training (text-to-video pretraining followed by instruction tuning) [42]. GameFactory leverages pretrained video diffusion with a multi-phase strategy for diverse, action-controllable, scene-generalizable game videos [43]. Hunyuan-GameCraft targets high-dynamic interactive game video with a unified input representation, hybrid history-conditioned training, and distillation [44]. Yume generates explorable interactive worlds from a single image using a masked video diffusion transformer with sampling advances and acceleration [45]. On the embodied side, work adapting pretrained video generators into controllable simulators with action-conditioned modules and motion-reinforced losses targets dynamic consistency and controllability directly [46], extending the UniSim lineage [26].

### The persistent failures: physics, drift, memory, latency

Long-horizon consistency and "drifting" recur as the field's central limitations. Explicit memory mechanisms are the emerging answer: work on out-of-sight dynamics maintains a persistent global state representation for entities beyond the field of view, and other systems add memory-augmented generation for long-horizon temporal consistency under bounded context windows. The shared diagnosis is that entity identities, dynamic states, and intervention-induced changes leave the active context window, so a purely Markovian frame-conditioned generator cannot maintain a coherent world.

The physics question is the most consequential. PhyGenBench tested physical commonsense in text-to-video generation and found failures that were "not fully addressed by scaling models or prompt engineering" [13]. This single negative result — robustness of physical failure to scale — is the strongest empirical argument that the diffusion-video paradigm does not automatically acquire physics from more data or compute, and it is the finding most often revisited by later benchmark work [15][19].

## World Models for Embodied AI, Robotics, and Driving

### Model-based reinforcement learning: the latent-state-space line

DreamerV3 is the flagship result of the latent paradigm: a model-based RL algorithm whose world model learns a categorical latent representation and trains actor and critic entirely inside imagination, reported to outperform prior approaches across diverse tasks with fixed hyperparameters [7]. Dreamer 4 ("Training Agents Inside of Scalable World Models") extends this to training an agent by RL inside a fast and accurate world model, learning to obtain diamonds in Minecraft from offline data alone [8]. The direction of travel is clear — from domain-specific world models to scalable, general world models that serve as the substrate for training.

### Real robots and continuous control

DayDreamer is the canonical demonstration that latent world models transfer from simulator to physical hardware: it learns locomotion and manipulation tasks directly from real-world interaction without a simulator [47]. TD-MPC2 provides the decoder-free alternative for continuous control, building model-predictive control on a learned latent world model [48]. The practical significance of these two papers is that world models are not only a game-playing curiosity; they are competitive methods for real, sample-constrained control problems where simulator access is the binding constraint.

### Autonomous driving: four distinct roles

Driving world models occupy four roles, sometimes simultaneously. As **generative simulators**, GAIA-1 frames driving as next-token prediction over video and actions [49], GAIA-2 adds multi-view controllability [50], Vista targets high fidelity with versatile controllability [51], and DriveDreamer works from real-world driving data [52]. As **structured predictors**, OccWorld forecasts future 3D occupancy rather than pixels, sidestepping the appearance/dynamics entanglement of video models [53]. As **data engines**, DriveDreamer4D uses world-model priors to improve 4D driving-scene representation explicitly as a "data machine" [54], and Cosmos-Drive-Dreams synthesizes rare-case driving data [41]. As **evaluators**, ACT-Bench evaluates action fidelity in driving world models [55], going beyond the FVD/FID perceptual metrics that dominate driving-simulator papers.

### Learning policies inside learned simulators

The newest frontier treats the world model as a training and evaluation environment for robot policies rather than as a generator to be admired. WorldGym uses a video-generation model as a surrogate environment for policy evaluation, reporting correlation with real-world performance plus generalization testing [17]. PolaRiS builds high-fidelity real-to-sim evaluations via neural reconstruction, improving correlation with real performance [56]. Genie Envisioner generalizes the Genie idea into a unified world-foundation platform for robotic manipulation, coupling a video diffusion world model with a neural simulator for instruction-driven control [57]. A survey of learning embodied intelligence from physical simulators and world models frames world models as the learned complement to classical physical simulators [58].

## Evaluating World Models: The Real Bottleneck

### Three incommensurable evaluation axes

Evaluation has split into at least three families that do not agree with each other. The first is **closed-loop task success** in standard agent benchmarks (Atari, DeepMind Control, Minecraft) and downstream policy performance [7][8]. The second is **perceptual and semantic fidelity** of generated rollouts: WorldSimBench judges world simulators by explicit perceptual evaluation plus implicit manipulative evaluation in embodied scenarios [59], and EWMBench measures scene, motion, and semantic quality with a dataset and toolkit [60]. The third is **action controllability and physical plausibility**: ACT-Bench for action fidelity [55], PhyGenBench for physical commonsense [13], and PhyWorldBench, which adds an "Anti-Physics" category [15].

### The realism–utility divergence

The most important empirical finding in the evaluation literature is that these axes diverge. World-in-World evaluates generative world models in closed-loop environments and stresses task success over visual quality, reporting insights on controllability, data scaling, and compute allocation that do not track perceptual scores [16]. WorldModelBench moves beyond perceptual similarity by scoring instruction-following and physics adherence with a human-labeled, fine-tuned automatic judger [14]. LikePhys evaluates intuitive-physics understanding in video diffusion models via likelihood preference rather than visual inspection [61]. PhysBench benchmarks and attempts to enhance vision-language models' physical-world understanding, showing substantial headroom [62]. WorldScore proposes a unified benchmark for world generation [63].

The pattern across these benchmarks is a consistent negative result: models that look best tend to be evaluated on the axis least predictive of whether a policy trained or tested inside them will succeed. Systems used as policy evaluators are also found to depend more on long-horizon rollout consistency and controllability than on short-term visual realism [17].

### Benchmarking the learning process, not just the output

A complement to output benchmarking is to test the world-model learning process itself. "Benchmarking World-Model Learning" separates reward-free interaction from a scored test phase conducted in a different environment, arguing that many evaluations fail to distinguish memorized dynamics from transferable ones [64]. This methodological move matters because it targets the construct — general dynamics understanding — rather than an output artifact.

### Evaluation validity as a threat to scaling claims

Finally, the evaluation literature intersects with a broader methodological critique: "Are Emergent Abilities of Large Language Models a Mirage?" argues that apparent emergence can be an artifact of the chosen metric rather than a real discontinuity with scale [22]. Applied to world models, this warns that claims of "physical understanding emerging with compute" may reflect benchmark design. Since almost all world-model benchmarks use a handful of custom metrics, this threat to validity is not hypothetical.

## The AGI Thesis, Latent Prediction, and the Skeptical Response

### The JEPA program as the technical core

The most articulate AGI-facing program is Yann LeCun's latent-prediction agenda: autonomous intelligence should be built around a world model that predicts in an abstract latent space rather than reconstructing pixels, trained with non-contrastive self-supervised objectives that avoid representation collapse. The Joint-Embedding Predictive Architecture instantiates this as an encoder, an action- or latent-conditioned predictor, and an energy-based compatibility objective. MC-JEPA is a concrete motion/content instantiation [9], and the "Image World Models" line extends JEPA to predicting global photometric transformations, framing representation learning as world modeling [10]. The program's extension to embodied settings — LiDAR-based spatial world models for driving [24] and policy representation learning [25] — shows it is not confined to static images.

### Genie as the concrete "foundation world model"

Where JEPA is a program, Genie is an artifact: an unsupervised generative model that creates action-controllable, playable virtual worlds from unlabeled video, inferring a discrete latent action space so that agents can be trained on behaviors never explicitly demonstrated [3]. This is the strongest concrete evidence that action-conditioned world models can be pretrained broadly and applied downstream, and it is the anchor for the "foundation world model" framing. Genie Envisioner generalizes the pattern to robotic manipulation [57].

### World-model-augmented LLM agents

A parallel and increasingly active thread asks whether LLMs can serve as — or be augmented with — world models. Web Agents with World Models augments a web navigation agent with a learned model that simulates action outcomes as free-form natural-language state differences, improving decisions where RL training is impractical [65]. Text2World is a PDDL-based benchmark for the symbolic world-model generation capabilities of LLMs [32]. R-WoM augments LLM agents with retrieval to combat long-horizon hallucination in world simulation [66]. The consistent finding is that LLM-as-world-model works in some regimes and fails in others, contingent on behavioral coverage and environment complexity — a bounded, not universal, result.

### The skeptical case: realism without physics

The strongest counter-evidence to the AGI thesis is empirical. Physics-IQ shows current AI video models achieve visual realism without understanding underlying physical principles [18]. "How Far is Video Generation from World Model: A Physical Law Perspective" finds diffusion-based video models show perfect in-distribution generalization and measurable scaling on combinatorial generalization but fail out-of-distribution, prioritizing color over other factors [19]. VideoPhy independently finds significant gaps in text-to-video models' adherence to physical commonsense [20]. PISA Experiments shows that fine-tuning large video models on simulated freefall videos improves some physical behavior while revealing generalization limits [21]. PhyGenBench's finding that physical failures are not resolved by scale or prompting closes the loop with the taxonomy literature [13].

Together these frame the central dispute of the field: visual fidelity is not evidence of a causal world model. A model can generate a physically impossible video that is perceptually indistinguishable from a valid one [15], and its errors may be robust to precisely the interventions — more data, more compute, better prompts — on which the scaling thesis depends [13][19].

### What is disputed

Four disagreements remain unresolved. First, **forecasting**: whether world-model capability follows predictable scaling laws. Relevant work on observational scaling laws and task-level scaling exists, but dedicated world-model scaling laws are immature — and they inherit the metric-artifact problem [22]. Second, **causality**: whether latent predictive models capture causal dynamics or only correlational structure; the JEPA thesis deliberately trades reconstruction for predictable structure but does not by itself establish causal grounding [10][19]. Third, **paradigm choice**: whether video-generative or latent-predictive world models are the more credible path to general intelligence — currently an argument by intuition on both sides [3][57]. Finally, **cost**: video world models are compute-expensive to train and slow to run, which is why few-step distillation and streaming architectures dominate the engineering literature [39][45].

## Trends and Open Problems

### Trends of the last two years

Three shifts define 2023–2025. **Rebranding and scaling**: video generation models were reinterpreted as world simulators, and the field moved from millions to billions of parameters with world foundation model platforms [3][34][4]. **From generation to interaction**: the frontier question changed from "does it look real?" to "can I steer it, in real time, persistently?" This produced few-step autoregressive streaming architectures [39], latent-increment simulators [38], and explicit memory mechanisms for long-horizon coherence. **From generation to evaluation and data**: the most consequential uses of world models are increasingly as data engines for training other systems [41][54] and as surrogate environments for policy evaluation [17][56], not as deliverables in themselves.

### Open problems

**Evaluation remains unsolved.** No consensus metric exists, and the realism–utility divergence means perceptual scores cannot substitute for closed-loop results [13][16][14][17]. Benchmarks are proliferating faster than they are being validated.

**Physical and causal grounding.** Whether scale produces physical understanding is empirically unresolved and currently points negative [13][18][19][20]. Structured alternatives — occupancy prediction [53], latent-increment dynamics, explicit 3D/4D representations [27] — are promising but immature.

**Long-horizon coherence and memory.** Drift, entity persistence, and out-of-sight dynamics are acknowledged as unsolved, and the memory mechanisms proposed so far are architectural patches rather than principled solutions.

**Latency versus quality.** Real-time interaction forces few-step distillation and streaming generation, which trades fidelity for interactivity; the optimal operating point is unestablished [39][45].

**Data quality and provenance.** Several sources surveyed here surfaced anomalous publication metadata from automated paper indexes, and world-model training corpora are largely uncurated internet video; both undermine the reliability of claims built on top of them. Systematic metadata hygiene is a practical prerequisite for the field's benchmark-driven narrative.

**Definitional scope.** As "world model" expands to cover RSSMs, video diffusion transformers, occupancy predictors, and LLM agents, the term risks losing analytic value. Clearer capability tiers — prediction, planning, simulation, counterfactual reasoning — would make claims comparable; current usage [28][29][30][58] is inconsistent.

### What would count as progress

The field's most defensible near-term targets are unglamorous: benchmarks that measure closed-loop utility and action fidelity rather than appearance [55][16][17], evaluations that distinguish memorized from transferable dynamics [64], reports of physical-plausibility results that survive scale and prompt engineering [13], and published scaling behavior specific to world models rather than borrowed from language models [22]. Until those exist, the AGI-through-world-models thesis remains a research program with strong artifacts — DreamerV3's cross-domain mastery [7], Dreamer 4's offline Minecraft diamonds [8], Genie's playable worlds from unlabeled video [3] — and an unproven central claim.

Additionally, recent agentic language world models explore interactive environment simulation from action traces [67].

## References
[1] World Models. arxiv. https://arxiv.org/abs/1803.10122 (2018-05-09)
[2] Recurrent World Models Facilitate Policy Evolution. arxiv. https://arxiv.org/abs/1809.01999 (2018-09-04)
[3] Genie: Generative Interactive Environments. arxiv. https://arxiv.org/abs/2402.15391 (2024-02-23)
[4] Cosmos World Foundation Model Platform for Physical AI. arxiv. https://arxiv.org/abs/2501.03575 (2025-07-09)
[5] Learning Latent Dynamics for Planning from Pixels. arxiv. https://arxiv.org/abs/1811.04551 (2019-06-04)
[6] Dream to Control: Learning Behaviors by Latent Imagination. arxiv. https://arxiv.org/abs/1912.01603 (2020-03-17)
[7] Mastering Diverse Domains through World Models. arxiv. https://arxiv.org/abs/2301.04104 (2024-04-17)
[8] Training Agents Inside of Scalable World Models (Dreamer 4). arxiv. https://arxiv.org/abs/2509.24527 (2025-09-29)
[9] MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features. arxiv. https://arxiv.org/abs/2307.12698 (2023-07-24)
[10] Learning and Leveraging World Models in Visual Representation Learning. hf-search. https://huggingface.co/papers/2403.00504 (2024-03-01)
[11] Diffusion Models Are Real-Time Game Engines (GameNGen). hf-search. https://huggingface.co/papers/2408.14837 (2024-08-27)
[12] Matrix-Game: Interactive World Foundation Model. arxiv. https://arxiv.org/abs/2506.18701 (2025-06-23)
[13] Towards World Simulator: Crafting Physical Commonsense-Based Benchmark for Video Generation (PhyGenBench). hf-search. https://huggingface.co/papers/2410.05363 (2024-10-07)
[14] WorldModelBench: Judging Video Generation Models As World Models. arxiv. https://arxiv.org/abs/2502.20694 (2025-02-27)
[15] PhyWorldBench: A Comprehensive Evaluation of Physical Realism in Text-to-Video Models. hf-search. https://huggingface.co/papers/2507.13428 (2025-07-17)
[16] World-in-World: World Models in a Closed-Loop World. hf-search. https://huggingface.co/papers/2510.18135 (2025-10-20)
[17] WorldGym: World Model as An Environment for Policy Evaluation. hf-search. https://huggingface.co/papers/2506.00613 (2025-09-30)
[18] Do generative video models learn physical principles from watching videos?. hf-search. https://huggingface.co/papers/2501.09038 (2025-01-14)
[19] How Far is Video Generation from World Model: A Physical Law Perspective. hf-search. https://huggingface.co/papers/2411.02385 (2024-11-04)
[20] VideoPhy: Evaluating Physical Commonsense for Video Generation. hf-search. https://huggingface.co/papers/2406.03520 (2024-06-05)
[21] PISA Experiments: Exploring Physics Post-Training for Video Diffusion Models by Watching Stuff Drop. hf-search. https://huggingface.co/papers/2503.09595 (2025-03-12)
[22] Are Emergent Abilities of Large Language Models a Mirage?. hf-search. https://huggingface.co/papers/2304.15004 (2023-04-28)
[23] Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. arxiv. https://arxiv.org/abs/1911.08265 (2020-02-21)
[24] AD-L-JEPA: Self-Supervised Spatial World Models with Joint Embedding Predictive Architecture for Autonomous Driving with LiDAR Data. hf-search. https://huggingface.co/papers/2501.04969 (2025-01-09)
[25] ACT-JEPA: Joint-Embedding Predictive Architecture Improves Policy Representation Learning. hf-search. https://huggingface.co/papers/2501.14622 (2025-01-24)
[26] Learning Interactive Real-World Simulators (UniSim). arxiv. https://arxiv.org/abs/2310.06114 (2024-09-26)
[27] 3D and 4D World Modeling: A Survey. hf-search. https://huggingface.co/papers/2509.07996 (2025-09-04)
[28] Understanding World or Predicting Future? A Comprehensive Survey of World Models. hf-search. https://huggingface.co/papers/2411.14499 (2024-11-21)
[29] A Comprehensive Survey on World Models for Embodied AI. hf-search. https://huggingface.co/papers/2510.16732 (2025-10-19)
[30] World Models for Cognitive Agents: Transforming Edge Intelligence in Future Networks. hf-search. https://huggingface.co/papers/2506.00417 (2025-05-31)
[31] Simulating the Visual World with Artificial Intelligence: A Roadmap. hf-search. https://huggingface.co/papers/2511.08585 (2025-11-11)
[32] Text2World: Benchmarking Large Language Models for Symbolic World Model Generation. hf-search. https://huggingface.co/papers/2502.13092 (2025-02-18)
[33] Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models. arxiv. https://arxiv.org/abs/2402.17177 (2024-04-17)
[34] Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond. hf-search. https://huggingface.co/papers/2405.03520 (2024-05-06)
[35] Diffusion for World Modeling: Visual Details Matter in Atari. arxiv. https://arxiv.org/abs/2405.12399 (2024-10-30)
[36] iVideoGPT: Interactive VideoGPTs are Scalable World Models. arxiv. https://arxiv.org/abs/2405.15223 (2024-10-31)
[37] MineWorld: a Real-Time and Open-Source Interactive World Model on Minecraft. hf-search. https://huggingface.co/papers/2504.08388 (2025-04-11)
[38] The Matrix: Infinite-Horizon World Generation with Real-Time Moving Control. arxiv. https://arxiv.org/abs/2412.03568 (2024-12-04)
[39] Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model. hf-search. https://huggingface.co/papers/2508.13009 (2025-08-18)
[40] Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control. hf-search. https://huggingface.co/papers/2503.14492 (2025-03-18)
[41] Cosmos-Drive-Dreams: Scalable Synthetic Driving Data Generation with World Foundation Models. hf-search. https://huggingface.co/papers/2506.09042 (2025-06-10)
[42] GameGen-X: Interactive Open-world Game Video Generation. hf-search. https://huggingface.co/papers/2411.00769 (2024-11-01)
[43] GameFactory: Creating New Games with Generative Interactive Videos. hf-search. https://huggingface.co/papers/2501.08325 (2025-01-14)
[44] Hunyuan-GameCraft: High-dynamic Interactive Game Video Generation with Hybrid History Condition. hf-search. https://huggingface.co/papers/2506.17201 (2025-06-20)
[45] Yume: An Interactive World Generation Model. hf-search. https://huggingface.co/papers/2507.17744 (2025-07-23)
[46] Pre-Trained Video Generative Models as World Simulators. hf-search. https://huggingface.co/papers/2502.07825 (2025-02-10)
[47] DayDreamer: World Models for Physical Robot Learning. arxiv. https://arxiv.org/abs/2206.14176 (2022-06-28)
[48] TD-MPC2: Scalable, Robust World Models for Continuous Control. arxiv. https://arxiv.org/abs/2310.16828 (2024-03-21)
[49] GAIA-1: A Generative World Model for Autonomous Driving. arxiv. https://arxiv.org/abs/2309.17080 (2023-09-29)
[50] GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving. arxiv. https://arxiv.org/abs/2503.20523 (2025-03-26)
[51] Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability. arxiv. https://arxiv.org/abs/2405.17398 (2024-10-28)
[52] DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving. arxiv. https://arxiv.org/abs/2309.09777 (2023-11-27)
[53] OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving. arxiv. https://arxiv.org/abs/2311.16038 (2023-11-27)
[54] DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation. hf-search. https://huggingface.co/papers/2410.13571 (2024-10-17)
[55] ACT-Bench: Towards Action Controllable World Models for Autonomous Driving. hf-search. https://huggingface.co/papers/2412.05337 (2024-12-06)
[56] PolaRiS: Scalable Real-to-Sim Evaluations for Generalist Robot Policies. hf-search. https://huggingface.co/papers/2512.16881 (2025-12-18)
[57] Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation. hf-search. https://huggingface.co/papers/2508.05635 (2025-08-07)
[58] A Survey: Learning Embodied Intelligence from Physical Simulators and World Models. arxiv. https://arxiv.org/abs/2507.00917 (2025-09-02)
[59] WorldSimBench: Towards Video Generation Models as World Simulators. hf-search. https://huggingface.co/papers/2410.18072 (2024-10-23)
[60] EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models. hf-search. https://huggingface.co/papers/2505.09694 (2025-05-14)
[61] LikePhys: Evaluating Intuitive Physics Understanding in Video Diffusion Models via Likelihood Preference. hf-search. https://huggingface.co/papers/2510.11512 (2025-10-13)
[62] PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding. arxiv. https://arxiv.org/abs/2501.16411 (2025-01-28)
[63] WorldScore: A Unified Evaluation Benchmark for World Generation. arxiv. https://arxiv.org/abs/2504.00983 (2025-11-28)
[64] Benchmarking World-Model Learning. hf-search. https://huggingface.co/papers/2510.19788 (2025-10-22)
[65] Web Agents with World Models: Learning and Leveraging Environment Dynamics in Web Navigation. hf-search. https://huggingface.co/papers/2410.13232 (2024-10-17)
[66] R-WoM: Retrieval-augmented World Model For Computer-use Agents. hf-search. https://huggingface.co/papers/2510.11892 (2025-10-13)
[67] From Traces to Agentic Worlds: Agentic Language World Models for Interactive Environment Simulation. hf-daily. https://huggingface.co/papers/2610.06100 (2026-10-05)
