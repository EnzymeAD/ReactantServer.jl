# Project Philosophy

This document describes who the project is for, what it optimizes for, and where it is headed. It exists to set expectations for users and contributors, and to serve as a reference when evaluating proposed changes.

## Mission

Make serving compiled models elegant: a hackable, Julia-first inference stack that maximizes the economic efficiency of GPU-based inference, built first for small and mid-size labs.

The efficiency half is concrete. GPU memory accounts for roughly two thirds of GPU cost; serving infrastructure that wastes memory wastes money. For the foreseeable future that means serving the largest number of models per GPU at a given quality of service, and serving models larger than a GPU's memory on the hardware a lab already has. The elegance half is what makes that power usable by a small team: the whole stack is plain Julia, legible end to end, adoptable off the shelf, and open to being bent toward a workload nobody anticipated. The project's purpose is to give the people who care about both arguments the tools to act on them.

## Who this is for

Small and mid-size labs and engineering organizations who need big-tech-level efficiency without big-tech scale. Concretely, this includes:

- Startups and small companies serving production ML where each GPU is a meaningful fraction of infrastructure spend.
- Scientific computing and research groups operating within bounded compute budgets.
- On-premise deployments where hardware is already purchased and underutilization is wasted capacity.
- Cloud deployments where GPU-hours are the dominant cost line.
- Teams running large language models locally, on hardware they own or rent, next to the rest of their model catalog.

The common thread is that these organizations measure their bottlenecks, optimize their stacks, and cannot solve serving problems by spending their way out. They benefit from compiler-grade optimization but lack the headcount to build it themselves.

The audience is engineers and operators who understand what is happening inside their systems. The project gives them tools; it does not abstract those tools beyond recognition or hide the system's behavior behind convenient defaults.

Larger organizations can use it too. There is no reason the project cannot run as a component of a large-scale system, and the control-plane seam exists for exactly that. The difference is in what ships in the box: for small and mid-size labs the project provides streamlined, off-the-shelf deployments; at large scale it provides the seams and leaves the surrounding tooling to the organizations that already build their own.

## One system, three deployment scales

The mission is served at three deployment scales, and the project treats each as a first-class use case with its own complete answer:

- **Small: one GPU, one process.** A single worker is the entire deployment. The fair scheduler shares the GPU across models by weighted deficit and learned cost, batch coalescing turns queue depth into throughput, and the on-demand weight cache serves a catalog larger than device memory. The worker speaks the full KServe V2 gRPC surface itself: one Julia process, one YAML file, nothing else to operate.
- **Medium: a few GPUs, one extra process.** One worker per GPU behind the gateway. In its LPT-packing mode the gateway derives model placement from live measurements, concentrating each model's traffic on few workers so batches actually fill while balancing compute load and weight memory across GPUs. The workers switch to the simpler FIFO discipline and the placement intelligence moves upstream. There is still no external infrastructure to stand up: the gateway is one more Julia process and one more YAML file, and there is no placement file to maintain.
- **Large: bring your own control plane.** Workers in explicit mode cede model lifecycle and residency to the operator's own control plane through a small gRPC control surface (status, residency, and policy RPCs), observable through the same KServe readiness APIs and Prometheus metrics the smaller tiers use. The project supplies the seam; the organization supplies the tooling that encodes needs only it can know. The first two tiers are off-the-shelf solutions; this one is deliberately a building block.

The principle that binds the tiers is non-interference: the project tries to serve these different use cases, but one use case must not crowd out, degrade, or overcomplicate the others. Each tier is strictly additive. A single-GPU deployment never pays for the gateway in dependencies, configuration, or runtime cost; a gateway deployment never requires a control plane; each step up is selected by a configuration value, not a different architecture. A feature that helps one tier is welcome in the form that leaves the other tiers untouched, and suspect in any form that does not.

## Local LLM serving

Serving large language models on local hardware is a goal of the project. The pressure that drives many models onto one GPU applies equally to one model that does not fit on it: device memory is the binding cost, and the answer in both cases is to treat device memory as a working set over a larger store rather than a permanent home for every weight.

The direction builds on mechanisms the server already has. The on-demand weight cache keeps weights in host RAM and moves them onto the GPU when a model runs, and a bundle can already carry several compiled programs that share one set of weights. From there, the project aims to:

- **Split one model into several compiled programs.** A model too large for device memory is partitioned into stages, each an ahead-of-time compiled executable holding its own slice of the weights, run in sequence.
- **Stream weights from storage.** Each stage's weights are loaded as the stage is about to run and released after, with the next transfer overlapping the current stage's compute, so the size of a servable model is bounded by storage rather than by host RAM or device memory.
- **Move weights directly from storage into device memory.** Where the hardware supports it, weights travel from local storage to the GPU without a copy through host memory.

The machinery specific to generation, such as the state carried between decoding steps, follows the non-interference principle above: it is added in the form that leaves the serving path for every other model untouched.

Local means the deployments the project already serves: one GPU or a few, operated by the team that uses them. Hosted LLM services with large numbers of concurrent users across many GPUs have different bottlenecks, and engines such as vLLM, TGI, and TensorRT-LLM are purpose-built for them.

## Where other tools fit better

The project is built for a specific design point. Naming where its edges are helps users choose quickly.

- **Hyperscale platform requirements.** Organizations operating at the scale of thousands of GPUs need multi-tenant isolation, complex traffic shaping, and deep integration with bespoke internal platforms. The project supports large deployments through the control-plane seam described above, which lets an organization bring exactly that machinery, and keeps it out of the core, where every smaller deployment would pay for it.
- **Hosted, high-concurrency LLM services.** As described above, the project's LLM focus is local serving.
- **A managed service.** The project is infrastructure for builders who run their own stack. Teams that would rather not operate one are better served by a managed inference service.
- **Serving models without converting them.** Every model is lowered by Reactant to a device executable (today via StableHLO/XLA). Teams that need to serve PyTorch, TensorFlow, and ONNX models side by side unconverted are better served by Triton or similar.
- **Eager-mode experimentation.** The project serves compiled programs in production. Interactive iteration with dynamic graphs belongs in the training framework.

## What the project optimizes for

In approximate priority order:

1. **Economic efficiency per GPU.** Models served per GPU, the largest model a GPU can serve, watts per inference, dollars per million requests. These are the metrics that matter for the target audience.
2. **Predictable, deterministic resource use.** Pre-allocated memory pools, static buffer assignment, no surprise allocations. Predictability matters for regulated deployments and for operators who need to size their hardware confidently.
3. **Compiler-grade optimization.** Whole-program compiler optimization (XLA today), kernel fusion across operations, layout planning. This is the leverage that lets small teams approach big-tech efficiency.
4. **A hackable, Julia-first stack.** The serving path, scheduler, gateway, client, and export tooling are plain Julia. There is no opaque core wrapped in scripting glue: the code a lab reads is the code that runs, so behavior can be inspected with the language's own tools and modified without a second toolchain. Elegance here is not aesthetics; it is the property that makes the system adoptable off the shelf and bendable when the shelf does not fit.
5. **Operator clarity.** Operators should be able to read configuration files, inspect bundles, understand scheduler behavior, and debug failures without reverse-engineering the system. The architecture is legible.
6. **Verification scope.** The codebase is small enough to audit. Dependencies are bounded and stable. The system is suitable for regulated environments without heroic compliance effort.

## What the project explicitly does not optimize for

- **Convenience over efficiency.** When a choice trades runtime cost for developer convenience, efficiency wins by default. Users who prefer the opposite tradeoff have other tools.
- **Maximum generality.** The project is opinionated about its design point. It does not try to be the inference server for every use case.
- **Compatibility with all upstream frameworks.** Models Reactant can trace and lower work (today via StableHLO/XLA); others do not. The conversion tooling makes this boundary as friendly as possible, but the boundary is real.
- **Featureful APIs.** The server's API surface is small on purpose. Each addition is weighed against the cost of the cognitive load it imposes on every user, not just the users who would benefit.

## What features will not be considered

Some categories of changes are off the table, regardless of who proposes them. Stating this explicitly is not standoffish; it is honest about the project's identity. Users and potential contributors deserve to know what to expect.

**Scale-specific features in the shared core.** A feature that serves one deployment scale belongs behind the seam that scale already uses (a configuration switch, the gateway, the control-plane RPCs), not in the core where every other deployment carries its weight. A feature useful at the scale of thousands of GPUs but with no benefit at the scale of one to fifty is better implemented in the control plane an organization brings; the project will extend that seam before it will absorb the feature.

**Features that abstract the project to death.** Every layer of abstraction has a cost: cognitive load on users, complexity in the codebase, surface area for bugs, longer onboarding for new contributors. The project will not add layers of abstraction that primarily exist to support uncommon use cases. The cost falls on everyone; the benefit accrues to a few.

**Features that add significant cognitive load for minimal benefit.** Configuration options that require deep understanding to set correctly, APIs that have many ways to express the same thing, behaviors that depend on subtle interactions between settings. If a feature's correct use requires extensive documentation or expertise that most users will not have, the feature is suspect even if it is technically sound.

**Features driven by users who have not otherwise contributed.** This is not a hard rule, and well-argued proposals from any source will be considered on their merits. But proposals from organizations that benefit from the project without contributing to it are weighed against proposals from contributors who have done the work. The project's direction is set by people who care about the same problem, not by feature requests from anonymous bystanders.

**Features that compromise the core mission for marginal users.** Every project gets pulled toward feature accumulation by users with edge cases. The project will resist this. Features that work against the economic efficiency goal will not be added even if individual users would find them convenient. The integrity of the mission is more valuable than any single feature.

## What the project will consider

Proposals are evaluated against the mission, not against the proposer's preferences. A few categories of contributions are particularly welcome:

- Improvements to economic efficiency: better memory packing, lower-overhead scheduling, faster startup, smaller binaries.
- Local LLM serving: partitioning a model into compiled stages, streaming weights from storage, direct storage-to-device transfer, and the generation support built on them.
- New conversion paths from upstream frameworks to StableHLO, especially for model architectures the project does not yet handle well.
- Documentation and tooling that make the project easier to evaluate and adopt for the target audience.
- Benchmarks against alternative inference servers on realistic workloads.
- Bug reports and fixes, especially for edge cases discovered in production deployments.
- Performance regressions, especially with reproducers. The project takes performance as a feature, not as an afterthought.

Contributions are weighed against the mission. Changes that improve efficiency, reduce cognitive load, or make the project more useful to the target audience are likely to land. Changes that move in the opposite direction are unlikely to, regardless of how well-implemented they are.