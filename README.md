# ParaDetox-Baseline

A small multilingual text detoxification project based on the **ParaDetox** task.

The goal of this project is to explore whether a multilingual sequence-to-sequence model can transform toxic text into a less toxic / neutral version while preserving the original meaning.

> This repository contains a baseline implementation and experiments. It is not intended to reproduce the full-scale ParaDetox research system.

---

## What is Text Detoxification?

Text detoxification is a text style-transfer task where a toxic sentence is rewritten into a non-toxic or neutral version while keeping its meaning as similar as possible.

For example:

```text
Input:
You made a mistake you ass.

Output:
You made a mistake.
