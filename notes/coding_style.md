# coding\_style.md

## Coding Philosophy for Experiments

This project embraces a coding philosophy rooted in minimalism, clarity, and functional necessity. It intentionally resists speculative overengineering, favoring infrastructure that emerges naturally from use. This approach draws on decades of software engineering experience and is heavily influenced by Guy Steele's *Growing a Language*.

### Core Principles

1. **Infrastructure Grows From Usage**

   * Abstractions should not be invented preemptively.
   * New components are introduced only when demanded by repeated use or clear generality.
   * We avoid the trap of building frameworks before we understand the problem domain in depth.

2. **Minimum Supported Implementation**

   * Each piece of code exists to serve a specific, immediate research purpose.
   * All modules and functions are minimal but complete in their context.
   * We trim, remove, or refactor aggressively if something is not being used or needed.

3. **Research-Oriented over Product-Oriented**

   * This is an experimental system, not a general-purpose framework.
   * Ease of modification, debugging, and reading takes precedence over feature completeness or reuse.

4. **Flat Before Modular**

   * We prefer simple, flat structure in early stages.
   * We refactor into modules or layers only once there are natural boundaries formed by usage.

5. **Trust the Experiment Loop**

   * We write just enough code to run the next experiment.
   * Analysis and infrastructure are built in response to what the data shows us we need.
   * Nothing is sacred. Everything is provisional.

### Anti-Goals

* No premature generalization.
* No API design for its own sake.
* No code included "just in case" it might be useful later.

### Relation to LLM-Generated Code

Large Language Models (LLMs) can be useful scaffolds, but they tend to overengineer:

* They try to anticipate all future use cases.
* They create large abstractions before core functionality is proven.
* They often separate concerns too early, leading to rigid or overly complex architectures.

This project acknowledges the utility of LLMs while actively pruning and rewriting their output to fit our methodology. We treat LLMs as drafting assistants, not design authorities.

### Inspiration: "Growing a Language"

The style here follows Guy Steele's idea that languages (and by extension, systems) should grow from the bottom up:

* Starting small.
* Empowering extension by composition.
* Avoiding top-down dictates.

A parallel title for this project might be *Growing a Research Tool*.

### Intended Audience

This coding style is for:

* Researchers working independently or in small teams.
* Those prioritizing clarity, evolution, and experimental focus.
* Anyone building tools to think with, not platforms to sell.

---

This document is living. It evolves with the project.
