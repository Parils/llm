### LLM资源合集

- [开源模型和评测榜单](开源模型.MD)
- [开源推理，微调，Agent，RAG，propmt 框架](开源框架.MD)
- [开源SFT，RLHF，Pretrain 数据集](开源数据.MD)
- [AIGC各领域应用汇总](AIGC各领域应用.MD)
- [Prompt教程，经典博客和AI会议访谈](教程博客会议.MD)

#### 论文
[论文合集](https://github.com/asimsinan/LLM-Research/blob/main/Papers.md)(截止2024/9/26)


#### 课程

[课程合集](https://github.com/asimsinan/LLM-Research/blob/main/UniversityCourses.md)：

1. [CS324: Large Language Models, Stanford University](https://stanford-cs324.github.io/winter2022/)
2. [COS 597G: Understanding Large Language Models, Princeton University](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/)
3. [263-5354-00L: Large Language Models, ETH Zürich](https://rycolab.io/classes/llm-s23/)
4. [Foundations of Large Language Models: Tools, Techniques, and Applications, University of Waterloo](https://watspeed.uwaterloo.ca/programs-and-courses/course-foundations-of-llms.html?id=1025128)
5. [CSC 6201/CIE 6021: Large Language Models, School of Data Science, The Chinese University of Hong Kong](https://llm-course.github.io)
6. [705.651.8VL: Large Language Models: Theory and Practice, Johns Hopkins Engineering for Professionals](https://apps.ep.jhu.edu/syllabus/fall-2023/705.651.8VL)
7. [TECH 16: Large Language Models for Business with Python, Stanford University](https://continuingstudies.stanford.edu/courses/professional-and-personal-development/large-language-models-for-business-with-python/20232_TECH-16)
8. [CS 194/294-267: Understanding Large Language Models: Foundations and Safety, Berkley University](http://rdi.berkeley.edu/understanding_llms/s24)
9. [EECS 498-016 / EECS 598-016: Foundations of Large Language Models, University of Michigan](https://cse.engin.umich.edu/wp-content/uploads/2023/10/EECS_498_LLM.pdf)
10. [CSC6203: Selected Topics in CS III (Large Language Models), The Chinese University of Hong Kong](https://github.com/FreedomIntelligence/CSC6203-LLM)
11. [CS 395T: Topics in Natural Language Processing, UT Austin](https://eunsol.github.io/courses/cs395t.html)
12. [CS 324: Advances in Foundation Models, Stanford University](https://stanford-cs324.github.io/winter2023/)
13. [11-667: Large Language Models Methods and Applications, Carnegie Mellon University](https://cmu-llms.org)
14. [CS224N: Natural Language Processing with Deep Learning, Stanford University](https://web.stanford.edu/class/cs224n/)
15. [6.5940: TinyML and Efficient Deep Learning Computing, MIT](https://hanlab.mit.edu/courses/2023-fall-65940)
16. [COMP790-101: Large Language Models, University of Toronto](https://github.com/craffel/llm-seminar)
17. [CS25: Transformers United V4, Stanford University](https://web.stanford.edu/class/cs25/)
18. [CS 601.471/671 NLP: Self-supervised Models, John Hopkins University](https://self-supervised.cs.jhu.edu/sp2023/index.html)
19. [IAP 2024: Introduction to Data-Centric AI, MIT](https://dcai.csail.mit.edu)
20. [CS388: Natural Language Processing, UT Austin](https://www.cs.utexas.edu/~gdurrett/courses/online-course/materials.html)

#### 框架

[框架合集](https://github.com/asimsinan/LLM-Research/blob/main/ToolsFrameworks.md)
1. [Langchain:](https://www.langchain.com) LangChain is a framework for developing applications powered by language models. It enables applications that:
   * **Are context-aware:** connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)
   * **Reason:** rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)

2. [llamaindex:](https://www.llamaindex.ai) LlamaIndex is a data framework for LLM-based applications which benefit from context augmentation. Such LLM systems have been termed as RAG systems, standing for “Retrieval-Augmented Generation”. LlamaIndex provides the essential abstractions to more easily ingest, structure, and access private or domain-specific data in order to inject these safely and reliably into LLMs for more accurate text generation. It’s available in Python (these docs) and Typescript.

3. [Haystack:](https://haystack.deepset.ai) Haystack is the open source Python framework by deepset for building custom apps with large language models (LLMs). It lets you quickly try out the latest models in natural language processing (NLP) while being flexible and easy to use. Our inspiring community of users and builders has helped shape Haystack into what it is today: a complete framework for building production-ready NLP apps.
......

#### 模型
[模型汇总](https://github.com/DSXiangLi/DecryptPrompt/blob/main/%E5%BC%80%E6%BA%90%E6%A8%A1%E5%9E%8B.MD)

##### 模型评测
|榜单|结果|
|----|-----|
|[Arena Hard](https://github.com/lm-sys/arena-hard-auto)|Lmsys Org开源的大模型评估基准，与人类偏好排名有更高一致性|
|[AlpacaEval 2.0：LLM-based automatic evaluation ](https://tatsu-lab.github.io/alpaca_eval/)| 开源模型王者vicuna,openchat, wizardlm|
|[Huggingface Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)|MMLU只评估开源模型，Falcon夺冠，在Eleuther AI4个评估集上评估的LLM模型榜单,vicuna夺冠| 
|...|...|

##### 国外开源模型
|模型链接     | 模型描述    |
| --- | --- |
|[Phi-3-MINI-128K](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)|还是质量>数量的训练逻辑，微软的3B小模型|
|[LLama3](https://llama.meta.com/llama3/)|Open Meta带着可商用开源的羊驼3模型来了，重回王座~|
|[GROK](https://x.ai/blog/grok-os)|马斯克开源Grok-1：3140亿参数迄今最大，权重架构全开放|
|...|...|

##### 开源多模态模型
|模型|描述|
|-----|-------|
|[Kosmos-2.5](https://github.com/microsoft/unilm/tree/master/kosmos-2.5)|微软推出的多模态擅长识别多文字、表格图片|
|[LLAVA-1.5](https://github.com/haotian-liu/LLaVA)  |升级后的LLAVA 13B模型浙大出品    |
| [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)    | 认知类任务评分最高 |
|[InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer)|书生浦语·灵笔2，擅长自由图文理解|
|[mPLUG-DocOwl](https://github.com/X-PLUG/mPLUG-DocOwl)|阿里出品面向文档理解的多模态模型|

##### 垂直领域模型&进展
|领域|模型链接     | 模型描述  
| ---| --- | --- | 
|医疗|[MedGPT](https://medgpt.co/home/zh)|医联发布的|
|金融|[OpenGPT](https://github.com/CogStack/OpenGPT)|领域LLM指令样本生成+微调框架|
|编程|[codegeex](http://keg.cs.tsinghua.edu.cn/codegeex/index_zh.html)|
|...|...|

#### 数据

[开源数据](https://github.com/DSXiangLi/DecryptPrompt/blob/main/%E5%BC%80%E6%BA%90%E6%95%B0%E6%8D%AE.MD)