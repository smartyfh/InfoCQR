# Enhancing Conversational Search: Large Language Model-Aided Informative Query Rewriting

## Introduction
Query rewriting plays a vital role in enhancing conversational search by transforming context-dependent user queries into standalone forms. Existing approaches primarily leverage human-rewritten queries as labels to train query rewriting models. However, human rewrites may lack sufficient information for optimal retrieval performance. 

<p align="center">
  <img src="assets/intro_exp.png" width="60%" />
  <p align="center">An example showing that human rewrites may overlook valuable contextual information.</p>
</p>

To overcome this limitation, we propose utilizing large language models (LLMs) as query rewriters, enabling the generation of informative query rewrites through well-designed instructions. We define four essential properties for well-formed rewrites and incorporate all of them into the instruction. In addition, we introduce the role of rewrite editors for LLMs when initial query rewrites are available, forming a "rewrite-then-edit" process. Furthermore, we propose distilling the rewriting capabilities of LLMs into smaller models to reduce rewriting latency. Our experimental evaluation on the QReCC dataset demonstrates that informative query rewrites can yield substantially improved retrieval performance compared to human rewrites, especially with sparse retrievers.


## Prompt Design

First, we identify four essential properties that a well-crafted rewritten query should possess:
+ **Correctness:** The rewritten query should preserve the meaning of the original query, ensuring that the user's intent remains unchanged.
+ **Clarity:** The rewritten query should be unambiguous and independent of the conversational context, enabling it to be comprehensible by people outside the conversational context. This clarity can be achieved by addressing coreference and omission issues arising in the original query.
+ **Informativeness:** The rewritten query should incorporate as much valuable and relevant information from the conversational context as possible, thereby providing more useful information to the off-the-shelf retriever.
+ **Nonredundancy:** The rewritten query should avoid duplicating any query previously raised in the conversational context, as it is important to ensure that the rewritten query only conveys the intent and meaning of the current query.

Then, we propose to prompt LLMs as query rewriters and rewrite editors using well-designed instructions that take all four properties into account.
<p align="center">
  <img src="assets/prompts.png" width="96%" />
  <p align="center">Our proposed approach involves prompting LLMs as query rewriters and rewrite editors through clear and well-designed instructions, along with appropriate demonstrations. In the absence of demonstrations, the LLM functions as a zero-shot query rewriter. We explicitly incorporate the requirement that rewritten queries should be as informative as possible into the instructions for generating informative query rewrites.</p>
</p>

## Results
