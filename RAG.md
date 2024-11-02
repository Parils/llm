## RAG
---

>本文介绍了RAG以及RAG pipeline的整个流程，包括请求转换、路由和请求构造、索引和检索、生成和评估等，
>
>_参考[Advanced RAG Series: Generation and Evaluation](https://div.beehiiv.com/p/advanced-rag-series-generation-evaluation)_

[toc]
### RAG概述

为什么需要RAG？

*   [问题1](https://docs.google.com/presentation/d/1C9IaAwHoWcc4RSTqo-pCoN3h0nCgqV2JEYZUJunv_9Q/edit#slide=id.g26685277936_0_261)：LLMs并不了解你的数据，且无法获取与此相关的最新数据，它们是事先使用来自互联网的公共信息训练好的，因此并不是专有数据库的专家也不会针对该数据库进行更新。
    <center>
    <img src="https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240326120038592-523124284.png"  width=70% />
    </center>

*   问题2：上下文窗口-每个LLM都有一个tokens的最大限制(通常平均为100tokens，约75个单词)，用于限制用户每次提交的tokens数据，这会导致丢失上下文窗口之外的上下文，进而影响准确性、产生检索问题和幻觉等。 
    <center>
    <img src="https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240326115848672-1669064045.png"  width=50% />
    </center>

*   问题3：[中间遗失](https://arxiv.org/abs/2307.03172?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=rag-say-what)^[_Liu N F, Lin K, Hewitt J, et al. Lost in the middle: How language models use long contexts[J]. Transactions of the Association for Computational Linguistics, 2024, 12: 157-173._] -即使LLMs可以一次性接收所有的数据，但它存在根据信息在文档中的位置来检索信息的问题。研究表明如果相关信息位于文档中间(而非开头或结尾时)时就会导致严重的性能降级。  
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240326141231215-572943224.png)

因此有了RAG(*Retrieval-Augmented Generation 检索增强生成*)。RAG的基本流程是，当回答问题时，首先从大量文档中检索到相关信息，然后基于这些信息，让LLMs生成答案。这样通过附加一个外部知识库，无需为每个特定任务重新训练整个大型模型。

因此RAG模型特别适合于知识密集型任务。



#### RAG范式的演变
RAG分为`Naive RAG`，`Advanced RAG`，和`Modular RAG`三种范式。

`Naive RAG` 就是传统的RAG 流程，先`Indexing`，然后`Retrieval`，最后`Generation`。Naive RAG在检索、生成和增强方面面临着许多挑战，因此随后提出了`Advanced RAG`范式，增加了预检索和检索后处理中的额外处理。在检索之前，可以使用`query改写`、`routing路由`和`query扩展`等方法来对齐问题和文档块之间的语义差异。检索后，对检索到的doc进行一个rerank，可以避免“中间丢失”现象，也可以对上下文进行过滤压缩，缩短窗口长度。

随着RAG技术的进一步发展和进化，产生了模块化RAG的概念。在结构上，它更自由、更灵活，引入了更具体的功能模块，如查询搜索引擎和多个答案的融合。在技术上，它将检索与微调、强化学习和其他技术相结合。在流程方面，RAG模块经过设计和编排，形成了各种RAG模式。

然而，模块化RAG也不是突然出现的，三种方式存在继承与发展的关系。可以这么理解Advanced RAG是模块化RAG的一个特例，而Naive RAG是Advanced RAG的特例。
![](https://s2.loli.net/2024/03/15/4FUaI1rteNLKXDO.png)


#### RAG的关键问题

- 检索粒度 可以是token、短语，还是chunk，段落、实体或者知识图谱
- 什么时候检索
- 如何利用检索到的信息
 
![](https://s2.loli.net/2024/03/15/d5uoz7AmDcrkQyT.png)

关于检索什么层级的内容，我们可以从检索粒度的粗细，以及数据结构化的高低来看业界研究结果.

![](https://s2.loli.net/2024/03/15/qOxAvzmug1eGd5j.png)

三个代表：

- `Chunk`级别，非结构化数据，搜索会召回大量信息，但是准确度低，会包含冗余信息
- `知识图谱`，丰富的语义和结构化数据，检索效率低，效果严重依赖KG的质量
- `KNN-LMM` 擅长处理长尾和跨域问题，计算效率高，但需要大量存储

如何使用检索到的内容?

在推理过程中，将检索到的信息集成到生成模型的不同层中

![](https://s2.loli.net/2024/03/15/38v42HZIoxbuiJN.png)

检索的时机

按照检索的频率从低到高，有：

![](https://s2.loli.net/2024/03/15/u4U2TpfvKAQGWOs.png)

- 一次检索，只检索一次，效率高，但可能导致检索结果相关度低
- 自适应检索，平衡效率和检索效果
- 每N个token检索1次，会导致检索次数过多，并召回大量冗余信息

RAG 技术发展树

![](https://s2.loli.net/2024/03/15/zU716QZ5JXSwrTD.png)





### RAG关键技术

**![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUfjnIeg74h1klbjyLlhYAV-bFUhaHYjMwt1ynD7A051RtNTrXwzRMytaaFNDnpM-21y15WRynSsDGcW28MEaFb2vuOK-AERkeOLhI0yilG0Ud6OzGeNK_HmmhKU0BgAYB2Awr0zsRj5XSZY2ZwXvfzyEutfpAM=nw?key=ZqWaY0jBtWrtvi15fPc-Fw)**


#### 请求转换(Query Translation)
![](./figures/下载.png)
_[请求转换](https://div.beehiiv.com/p/rag-say)是为了提高查询结果的准确性而对用户请求进行重构、优化的过程。_

##### 请求分解

由于用户问题可能太含糊、太特殊或缺少上下文，因此LLM可能无法很好地处理这些问题。通常会建议在将请求发送到嵌入模型之前对其进行重构。下面是一些重构方式：

*   _[重写-检索-读取](https://arxiv.org/abs/2305.14283?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=rag-say-what)_^[_Ma X, Gong Y, He P, et al. Query rewriting for retrieval-augmented large language models[J]. arXiv preprint arXiv:2305.14283, 2023._]: 这种方式注重对用户的查询进行重构(而不仅仅是采用retriever或reader，左侧图示)。它使用一个LLM生成一个查询，然后使用web查询引擎来检索内容(中间图示)。此外还需要在pipeline中使用一个小型语言模型来进一步对齐结果(右侧图示)。
    
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240326094543980-869476377.png)

*   _[对问题进行浓缩或改写](https://smith.langchain.com/hub/langchain-ai/weblangchain-search-query?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=rag-say-what&tab=0)_：通常用于会话中，通过把对话改写成一个好的独立问题来给聊天机器人提供上下文。Langchain的一个提示模板示例如下：
    ```json
        "Given the following conversation and a follow up question, rephrase the follow up \ question to be a standalone question.
        
        Chat History: {chat_history}
        
        Follow Up Input: {question}
        
        Standalone Question:"
    ```    
    
*   [_RAG Fusion_](https://flight.beehiiv.net/v2/clicks/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJodHRwczovL2dpdGh1Yi5jb20vbGFuZ2NoYWluLWFpL2xhbmdjaGFpbi9ibG9iL21hc3Rlci9jb29rYm9vay9yYWdfZnVzaW9uLmlweW5iP3V0bV9zb3VyY2U9ZGl2LmJlZWhpaXYuY29tJnV0bV9tZWRpdW09cmVmZXJyYWwmdXRtX2NhbXBhaWduPXJhZy1zYXktd2hhdCIsInBvc3RfaWQiOiIxMTBmMzk4YS04Y2UyLTRkNGYtYTRkYy0xMDJlNDUzNDA0YWIiLCJwdWJsaWNhdGlvbl9pZCI6IjQ5NjYzOGM1LTY5ZWQtNDdiZS1hNzllLTgxZDVhZGRmNWYwNCIsInZpc2l0X3Rva2VuIjoiYmY5YWM3MTYtNDhlYS00N2YxLWE1MzAtYjQwZGY4NjFkMzRiIiwiaWF0IjoxNzExNDIxMDg5LCJpc3MiOiJvcmNoaWQifQ.B7oRHav524GxRGcTVm3NHA92KdOQ66WQMUDHbfLIDik)^[_Rackauckas Z. Rag-fusion: a new take on retrieval-augmented generation[J]. arXiv preprint arXiv:2402.03367, 2024._]: 将RAG和倒数排名融合([RRF](https://arxiv.org/abs/2402.03367?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=rag-say-what))结合。生成多个查询(从多个角度添加上下文)，并使用倒数分数对查询结果重新排序，然后将文档和分数进行融合，从而得到更全面和准确的回答。
    
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240326113052846-769262618.png)

*   _[Step-Back Prompting](https://arxiv.org/abs/2310.06117?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=rag-say-what)_^[_Zheng H S, Mishra S, Chen X, et al. Take a step back: Evoking reasoning via abstraction in large language models[J]. arXiv preprint arXiv:2310.06117, 2023._]：这是一种更加技术性的提示技术，通过LLM的抽象来衍生出更高层次的概念和首要原则。这是一个对用户问题的迭代过程，用于生成一个"后退一步"的问题(step back question)，然后使用该问题对应的回答来生成最终的回答。  
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240326113250384-1843949440.png)
    
*   _[Query Expansion](https://arxiv.org/abs/2305.03653?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=rag-say-what)_^[_Jagerman R, Zhuang H, Qin Z, et al. Query expansion by prompting large language models[J]. arXiv preprint arXiv:2305.03653, 2023._]：这是一个通过为LLM提供一个查询并生成新的内容来扩展查询的过程。适用于文档检索，特别是Chain-of-Thought(CoT 思维链)提示。
    
    - **使用生成的答案进行扩展**：该方式中，让LLM基于我们的查询生成一个假设的回答，然后将该回答追加到我们的查询中，并执行嵌入搜索。通过使用假设的答案来扩展查询，这样在进行嵌入搜索时可以检索到其他相关向量空间，进而可以提供更准确的回答。
    
    - **使用多个查询进行扩展**：使用LLM基于我们的查询来生成额外的类似查询，然后将这些额外的查询和原始查询一起传给向量数据库进行检索，从而可以大大提升准确性。**注意**需要通过对提示进行迭代来评估哪些提示会产生最佳结果。
    
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240326143811418-424120882.png)

##### 伪文档(Psuedo documents)

[伪文档嵌入 (HyDE)](https://arxiv.org/abs/2212.10496?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=rag-say-what)^[_Precise Zero-Shot Dense Retrieval without Relevance Labels_]：当向向量数据库询问问题时，该数据库有可能不会很好地标记相关性。因此更好的方式是先创建一个假设的回答，然后再查询匹配的向量。需要注意的是，虽然这比直接查询到答案的嵌入匹配要好，但对于高度不相关的问题和上下文，产生幻觉的可能性也会更高，因此需要对过程进行优化，并注意这些边缘情况。

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240326115331031-964614899.png)

作为RAG流程的第一个步骤，查询转换并不存在正确或错误的方式。这是一个带有实验性质的领域，只有通过构建才能知道哪些方式最适合你的使用场景。



#### 路由(Routing)

_路由为了将请求发送到与与请求内容相关的存储。_

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240320094949193-964165550.png)

由于环境中可能存在多个数据库和向量存储，而答案可能位于其中其中任何一个，因此需要对查询进行路由。基于用户查询和预定义的选择，LLM可以决定：

*   正确的数据源
    
*   需要执行的动作：例如，概括 vs 语义搜索
    
*   是否并行执行多个选择，并校对结果([多路由功能](https://flight.beehiiv.net/v2/clicks/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJodHRwczovL2RvY3MubGxhbWFpbmRleC5haS9lbi9zdGFibGUvbW9kdWxlX2d1aWRlcy9xdWVyeWluZy9yb3V0ZXIvcm9vdC5odG1sP3V0bV9zb3VyY2U9ZGl2LmJlZWhpaXYuY29tJnV0bV9tZWRpdW09cmVmZXJyYWwmdXRtX2NhbXBhaWduPXJvdXRpbmctYW5kLXF1ZXJ5LWNvbnN0cnVjdGlvbiIsInBvc3RfaWQiOiJmMjBjYTYxMy1jYWQ3LTQzNzktOWVlMy1hNzY2YWZkMGY2NTkiLCJwdWJsaWNhdGlvbl9pZCI6IjQ5NjYzOGM1LTY5ZWQtNDdiZS1hNzllLTgxZDVhZGRmNWYwNCIsInZpc2l0X3Rva2VuIjoiNDgxYmZmOTAtZjY0ZC00YWU0LTk0OTMtMjg1ZjcyN2Q2Y2VjIiwiaWF0IjoxNzEwNzUwMzM5LCJpc3MiOiJvcmNoaWQifQ.yHvmWg9TdPGOwiu6Q3z849jjbMYCOhHDIo1sDjbX-Bw))
    

下面是一些路由请求的方式：

* 逻辑路由：在这种情况下，我们让LLM根据预定义的路径来决定参考知识库的哪个部分。这对于创建非确定性链非常有用，一个步骤的输出产生下一个步骤。**该方式用于通过LLM来选择知识库**。
    
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240320100046487-1460124478.png)
    
    下面是一个逻辑路由的[例子](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_10_and_11.ipynb)，用于根据用户的编程语言来选择合适的数据源：
    ```python
        from typing import Literal
        
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.pydantic_v1 import BaseModel, Field
        from langchain_openai import ChatOpenAI
        
        # Data model
        class RouteQuery(BaseModel):
            """Route a user query to the most relevant datasource."""
        
            datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
                ...,
                description="Given a user question choose which datasource would be most relevant for answering their question",
            )
        
        # LLM with function call 
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        structured_llm = llm.with_structured_output(RouteQuery)
        
        # 构造提示，根据用户问题中的编程语言类型类选择合适的数据源
        system = """You are an expert at routing a user question to the appropriate data source.
        
        Based on the programming language the question is referring to, route it to the relevant data source."""
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )
        
        # 定义一个 router，通过给LLM(structured_llm)输入提示(prompt)来产生结果
        router = prompt | structured_llm
    ```    
    
    下面是使用方式。用户给出`question`，然后调用`router.invoke`让LLM找出合适的数据源，最终会在`result.datasource`中返回`python_docs`。
    ```python
        question = """Why doesn't the following code work:
        
        from langchain_core.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
        prompt.invoke("french")
        """
        
        result = router.invoke({"question": question})
    ```    
    
*   语义路由：使用基于上下文的提示来增强用户查询的强大方法。可以帮助LLMs快速、经济地选择路由(可以预定义或自定义选项)，产生确定性的结果。**该方式用于通过LLM选择提示**。
    
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240320100442783-321770073.png)
    
    下面是一个使用语义路由的例子，它嵌入了两个提示模板，分别用于处理物理和数学问题，然后通过匹配用户问题和提示模板的相似性程度来选择合适的提示模板，然后应用到LLM中。
    ```python
        from langchain.utils.math import cosine_similarity
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        # 创建两个提示模板，一个用于解决物理问题，另一个用于解决数学问题
        physics_template = """You are a very smart physics professor. \
        You are great at answering questions about physics in a concise and easy to understand manner. \
        When you don't know the answer to a question you admit that you don't know.
        
        Here is a question:
        {query}"""
        
        math_template = """You are a very good mathematician. You are great at answering math questions. \
        You are so good because you are able to break down hard problems into their component parts, \
        answer the component parts, and then put them together to answer the broader question.
        
        Here is a question:
        {query}"""
        
        # 嵌入提示
        embeddings = OpenAIEmbeddings()
        prompt_templates = [physics_template, math_template]
        prompt_embeddings = embeddings.embed_documents(prompt_templates)
        
        # Route question to prompt 
        def prompt_router(input):
            # 嵌入查询
            query_embedding = embeddings.embed_query(input["query"])
            # 计算查询和提示模板的相似度
            similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
            most_similar = prompt_templates[similarity.argmax()]
            # 根据相似度来选择最合适的提示模板 
            print("Using MATH" if most_similar == math_template else "Using PHYSICS")
            return PromptTemplate.from_template(most_similar)
        
        
        chain = (
            {"query": RunnablePassthrough()}
            | RunnableLambda(prompt_router)
            | ChatOpenAI()
            | StrOutputParser()
        )
        
        # 用户提问，什么是黑洞
        print(chain.invoke("What's a black hole"))
    ```    
    

#### 请求构造(Query Construction)

_请求构造是为了解决针对特定类型数据库查询的问题。_

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240320112853716-2117553512.png)

在定义好路由之后是否就可以跨数据存储发送请求？如果使用的是非结构化数据存储就可以，但实际中，大部分数据都保存在结构化数据库中，因此在构造请求的同时需要考虑到数据库的类型。

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240319160931282-613884505.png)

我们很容易会假设和LLMs交互使用的是自然语言，但这是不对的，查询采用的语言类型取决于和数据存储的交互方式，因此在构建查询时需要牢记数据库的查询语言(从使用SQL的关系型数据库到使用相关结构化元数据的非结构化数据)。

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240319161807757-1882185518.png)

**a.** [自查询检索器-Self-query retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=routing-and-query-construction) (文本->元数据的过滤器)：向量数据库中带有**清晰元数据文件**的非结构化数据可以启用此类retriever。任何用户问题(自然语言格式)都可以被拆分为一个查询和一个过滤器(如根据年、电影、艺术家)。通过提升发送到LLM的数据质量，可以大大提升Q&A的工作流表现。

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240319162708278-1259842871.png)

**b.** **文本-> SQL:** 通常LLMs对Text2SQL的表现不佳，去年有很多初创公司就将焦点放在如何解决此类问题上。从创建虚构的表和字段到用户查询中的拼写错误等原因都可能导致LLMs执行失败。鉴于这种数据库相当普遍，因此出现了很多帮助LLM准确创建SQL查询的方式，但都与特定的知识库相关，因此不存在可以用于多种数据库的通用方式。对于一个新的数据库，只能通过构建、迭代、修复来优化LLMs的表现。

*   Create Table + Select 3：在了解更多高级技术之前，这是获得模型在数据库上的表现基线的最直接的方法。在Prompt设计中，对于每张表，可以包括一个CREATE TABLE描述，并在一个SELECT语句中提供三个示例行。

    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240319164633823-835983418.png)

*   [少量样本示例](https://flight.beehiiv.net/v2/clicks/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJodHRwczovL2FyeGl2Lm9yZy9hYnMvMjMwNS4xMjU4Nj91dG1fc291cmNlPWRpdi5iZWVoaWl2LmNvbSZ1dG1fbWVkaXVtPXJlZmVycmFsJnV0bV9jYW1wYWlnbj1yb3V0aW5nLWFuZC1xdWVyeS1jb25zdHJ1Y3Rpb24iLCJwb3N0X2lkIjoiZjIwY2E2MTMtY2FkNy00Mzc5LTllZTMtYTc2NmFmZDBmNjU5IiwicHVibGljYXRpb25faWQiOiI0OTY2MzhjNS02OWVkLTQ3YmUtYTc5ZS04MWQ1YWRkZjVmMDQiLCJ2aXNpdF90b2tlbiI6IjQ4MWJmZjkwLWY2NGQtNGFlNC05NDkzLTI4NWY3MjdkNmNlYyIsImlhdCI6MTcxMDc1MDMzOSwiaXNzIjoib3JjaGlkIn0.idZ2UxiXfgFV3pCOdpvacl4EqoSU2uPXNd2oEGpnBws)：为LLM提供少量Q&A示例来帮助它理解如何构建请求，通过这种方式可以提升10~15%的准确性。根据示例的质量和使用的模型，添加更多的示例可以获得更高的准确性。在有大量示例的情况下，可以将其保存在向量存储中，然后通过对输入查询进行语义搜索，动态选择其中的一部分。
    
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240319170729968-144058817.png)

*   此外还有一篇不错的[博客](https://www.anyscale.com/blog/fine-tuning-llama-2-a-comprehensive-case-study-for-tailoring-models-to-unique-applications?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=routing-and-query-construction)展示了fine-tuning带来的巨大提升：  
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240319171522599-131778484.png)
    
*   RAG + Fine-tuning：相比于只是为了让语言模型能够理解而将整个schema添加到提示中，使用经过调优的schema RAG和ICL的模型可以将准确性提高20%以上。  
    
*   用户拼写错误：通过搜索合适的名词而不是使用包含正确拼写的向量存储，是减少用户特定错误的一种好方法。这在早期的Text2SQL中是一个特别令人烦恼的问题。

**c. 文本-> Cypher**：与图数据库有关，用于表达无法使用表格形式表示的关系。Cypher是这类数据库的查询语言。text-2-Cypher 是一项复杂的任务，对于此类任务，推荐使用GPT4。

**[知识图谱](https://arxiv.org/abs/2311.07509?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=routing-and-query-construction)^[_Sequeda J, Allemang D, Jacob B. A benchmark to understand the role of knowledge graphs on large language model's accuracy for question answering on enterprise SQL databases[C]//Proceedings of the 7th Joint Workshop on Graph Data Management Experiences & Systems (GRADES) and Network Data Analytics (NDA). 2024: 1-12._]可以有效提高检索的准确性**，因此数据格式不应该局限于表和2D结构。

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240319175618037-703238488.png)

请求转换、构造和路由可以帮助我们和请求的数据库进行交互。接下来，我们将进入称为索引(Indexing)的工作流程部分，在那里，我们将更深入地研究拆分、索引嵌入以及如何通过配置这些功能来实现准确检索。


#### 索引(Indexing)

_在LLM中，文档会以chunk进行切分，[索引](https://div.beehiiv.com/p/advanced-rag-series-indexing)用于找到与查询相关的chunk_

在上文中，我们讨论了在构造查询时需要考虑到与数据库交互所使用的语言。这里要讲的索引类似被查询的数据，索引的实现方式有很多种，但目的都是为了在不丢失上下文的情况下方便LLM的理解。由于对应用户问题的答案可能位于文档中的任何地方，且考虑到LLMs在实时数据、上下文窗口和"中间遗失"问题中的不足，因此有效划分chunk并将其添加到上下文中非常重要。

chunk划分和嵌入是实现准确检索的索引核心部分。简单来说，嵌入向量是将一个大而复杂的数据集转化为一组能够捕捉所嵌入的数据本质的数字。这样就可以将用户请求转化为一个嵌入向量(一组数字)，然后基于语义相似性来检索信息。它们在高纬度空间的呈现如下([相似的词的距离相近](https://14h034160212.github.io/Qiming_Bao_IEEE_VTS_Natural_Language_Processing_Reasoning_Invited_Talk_Final.pdf?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-indexing))：

![image](https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/79de9a81-2320-4160-90d3-e15d07ff4e16/image.png?t=1709275173)

回到chunk划分，为了方便理解，假设有一个大型文档，如电子书。你希望从这本书中解答问题，但由于它远超LLMs的上下文窗口，我们可能需要对其分块，然后将于用户问题相关的部分提供给LLM。而当执行分块时，我们不希望因为分块而丢失故事的角色上下文，此时就会用到索引。

下面是一些索引方式：

##### chunk优化

首先需要考虑数据本身的长度，它定义了chunk的划分策略以及使用的模型。例如，如果要嵌入一个句子，使用[sentence transformer](https://huggingface.co/sentence-transformers?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-indexing)可能就足够了，但对于大型文档，可能需要根据tokens来划分chunk，如使用 [text-embedding-ada-002](https://openai.com/blog/new-and-improved-embedding-model?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-indexing)。

其次需要考虑的是这些嵌入向量的最终使用场景。你需要创建一个Q&A机器人？一个摘要工具？还是作为一个代理工具，将其输出导入到其他LLM中进一步处理？如果是后者，可能需要限制传递到下一个LLM的上下文窗口的输出长度。

![image](https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/98b7e7bf-45c0-4e32-9c70-abcc5a9e4e37/image.png?t=1709275260)

下面提供了几种实现策略：

###### 基于规则

使用分隔符(如空格、标点符号等)来切分文本：

*   **固定长度**：最简单的方式是根据固定数目的字符来划分chunk。在这种方式下，缓解上下文遗失的方式是在每个chunk中添加重叠的部分(可自定义)。但这种方式并不理想，可以使用langchain的[CharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/character_text_splitter)进行[测试](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)：
    
    ![image](https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/4f553bc6-4a31-4752-aef1-bcacdd1e8f50/image.png?t=1709275293)
    
    然后是所谓的结构感知拆分器，即基于句子、段落等划分chunk。
    
*   **NLTK语句分词器(**[Sentence Tokenizer](https://www.youtube.com/watch?app=desktop&v=vyyTpL8QKgc&utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-indexing)**)**：将给定的文本切分为语句。这种方式虽然简单，但仍然受限于对底层文本的语义理解，即便在一开始的测试中表项良好，但在上下文跨多个语句或段落的场景下仍然不理想(而这正是我们希望LLM查询的文本)。
    
    ![image](https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/cbd803c1-714c-4648-bd6d-8a5765d94851/image.png?t=1709275350)

*   **[Spacy](https://ashutoshtripathi.com/2020/05/04/how-to-perform-sentence-segmentation-or-sentence-tokenization-using-spacy-nlp-series-part-5/?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-indexing)语句分割器(**Sentence Splitter**)**：另一种是基于语句进行拆分，在希望引用小型chunks时有用。但仍存在和NLTK类型的问题。
    
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240320162953354-1720510632.png)

###### 递归结构感知拆分(Recursive structure aware splitting)

结合固定长度和结构感知策略可以得到递归结构感知拆分策略。Langchain文档中大量采用了这种方式，其好处是可以更好地控制上下文。为了方便语义搜索，此时chunk大小不再相等，但仍然**不适合结构化数据**。

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240320170304494-125838605.png)

###### [内容感知拆分](https://medium.com/kx-systems/guide-to-multimodal-rag-for-images-and-text-10dab36e3117)

对于非结构化数据来说，上面几种策略可能就足够了，但对于结构化数据，就需要根据结构本身类型进行拆分。这就是为什么有专门用于Markdown、LaTeX、HTML、带表格的pdf、多模式(即文本+图像等)的文本拆分器。

![image](https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/682ad824-a3b0-4e75-a02a-9f139cea8f8c/image.png?t=1709275477)


##### 多表示索引

相比于将整个文档进行拆分，然后根据语义相似性检索出 top-k的结果，那如果将文本转换为压缩的检索单元会怎样？例如，压缩为摘要。

###### 父文档

这种场景下可以根据用户的请求检索到最相关的chunk并传递该chunk所属的[父文档](https://clusteredbytes.pages.dev/posts/2023/langchain-parent-document-retriever/?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-indexing)(Parent Document)，而不是仅仅将某个chunk传递给LLM。通过这种方式可以提升上下文并增强检索能力。但如果父文档比LLM的上下文窗口还要大怎么办？为了匹配上下文窗口，我们可以将较大的chunks和较小的chunks一起传递给LLM(而不是整个父文档)。

![image](https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/1a99f677-08f2-42c5-a15b-74664b6c8626/image.png?t=1709275570)

上面vectorstores中存储的是较小的 chunks，使用InMemoryStore存储较大的 chunk。

在使用时文档会被分为大小两种chunk，其中小chunk都有其所属的大chunk。由于"chunk越小，其表达的语义精确性更高"，因此在查询时，首先检索到较小的chunk，而较小的chunk的元数据中保存了其所属的大chunk，因而可以将小chunk和其所属的大chunk一起传递给LLM。

###### 密集检索(Dense X Retrieval)

密集检索是一种使用非语句或段落chunk进行上下文检索的新方法。在下面[论文](https://arxiv.org/pdf/2312.06648.pdf?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-indexing)中，作者将其称之为"proposition"，一个proposition包含：

*   文本中的不同含义：需要捕获这些含义，这样所有propositions一起就能在语义上覆盖整个文本。
*   最小单元：即不能再进一步拆分propositions。
*   上下文相关和自包含：即每个propositions应该包含所有必需的上下文。

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240320174055039-1313735929.png) ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240320174138951-2129280763.png)

Proposition级级别的检索比语句级别的检索和篇章级别的检索分别高35%和22.5%(显著提高)。

##### 特定嵌入

领域特定和/或高级嵌入模型。

*   Fine-tuning：对嵌入模型进行微调可以帮助改进RAG pipeline检索相关文档的能力。这里，我们使用LLM生成的查询、文本语料库以及两者之间的交叉参考映射来磅数嵌入模型理解需要查找的语料库。微调嵌入模型可以帮助提升大约5~10%的表现。
    
    下面是[Jerry Liu对](https://twitter.com/jerryjliu0/status/1702111739549192512?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1702111739549192512%7Ctwgr%5E20f772fc032debe35c32d983dae803dd325ddf11%7Ctwcon%5Es1_&ref_url=https%3A%2F%2Fdiv.beehiiv.com%2Fp%2Fadvanced-rag-series-indexing)微调嵌入模型的建议：
    
    1.  在项目开始时，需要在嵌入文档前对基础模型进行微调
    2.  由于生产环境中的文档分布可能会发生变化，因此在微调查询适配器(query adapter)时需要确保嵌入的文档不会发生变化。
   
*   [ColBERT](https://github.com/stanford-futuredata/ColBERT?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-indexing)：这是一个检索模型，它能够基于BERT在大规模数据集上实现可扩展的搜索(毫秒级别)。这里，快速和准确检索是关键。
    
    它会将每个段落编码为token级嵌入的矩阵(如下蓝色所示)。当执行一个搜索时，它会将用户查询编码为另一个token级嵌入的矩阵(如下绿色所示)。然后基于上下文匹使用"可扩展的向量相似性(_MaxSim_)操作"来匹配查询段落。
    
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240321164739126-314985425.png)
    
    后期交互是实现快速和可扩展检索的关键。虽然交叉编码器会对每个可能的查询和文档对进行评估，从而提高准确性，但对于大规模应用程序而言，该特性会导致计算成本不断累积。为了实现大型数据集的快速检索，需要提前计算文档嵌入，因而需要平衡计算成本和检索质量。
    

##### 分层索引

斯坦福大学研究人员基于不同层次的文档摘要树提出了[RAPTOR](https://arxiv.org/html/2401.18059v1?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-indexing)^[_Sarthi P, Abdullah S, Tuli A, et al. Raptor: Recursive abstractive processing for tree-organized retrieval[J]. arXiv preprint arXiv:2401.18059, 2024._]模型，即通过对文本块的聚类进行摘要来实现更准确的检索。文本摘要涵盖了更大范围的上下文，跨越不同的尺度，包括主题理解和细粒度的内容。检索增强的文本摘要涵盖了更大范围不同主题理解和粒度的上下文。

论文声称，通过递归摘要进行检索可以显著提升模型表现。"_在涉及复杂多步骤推理的问答任务中，我们展示了最佳结果。例如，通过将RAPTOR检索与GPT-4相结合，我们可以在QuALITY基准测试的最佳表现上提高20%的绝对准确率。_"

![image](https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/5ee46600-6975-42c1-8e01-5659d586b62d/image.png?t=1709275859)

> llamaindex提供了这种[实现](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-raptor/examples/raptor.ipynb)。



#### 检索(Retrieval)

_[检索](https://div.beehiiv.com/p/advanced-rag-series-retrieval)可以看做是对索引到的数据的进一步提炼。_
![](https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/8368de64-741a-4488-982b-d3e4245334ba/RAG_-_Retrieval.png?t=1709798274)

在完成数据检索之后，下一步需要根据用户的请求来获取相关数据。最常见和最直接的方法是从之前索引的数据(最近的邻居)中识别并获取与用户查询在语义上最接近的chunks。类似[如下](https://weaviate.io/blog/distance-metrics-in-vector-search?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-retrieval)向量空间：

![image](https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/17e905bb-b3d1-4fd9-a378-c668db42bb37/image.png?t=1709790732)

检索并不是一步到位的，而是从查询转换开始的一些列步骤，以及如何通过检索来提升获取到相关的chunks之后的检索质量。假设我们已经根据相似性搜索，从向量数据库中找到前k个chunks，但由于大部分chunks存在重复内容或无法适应LLM的上下文窗口。因此在将chunks传递给LLM之前，我们需要通过一些检索技术来提升上下文的质量。

LLMs的世界中，并不存在一劳永逸的方法，需要根据使用场景和chunks的特性来找到合适的技术。下面是一些典型的方法：

##### Ranking

###### Reranking

如果我们想要从数据库的chunk中查找答案，可以选择Reranking，它是一种可以给LLM提供最相关上下文的有效策略。有多种实现方式：

*   提升多样性：最常见的方法是[最大边缘相关性](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-retrieval)(Maximum Marginal Relevance-MMR)，这可以通过因子(a)与请求的相似度或(b)与已选的文档的距离等因子来实现。
    
    这里有必要提一下[Haystack](https://towardsdatascience.com/enhancing-rag-pipelines-in-haystack-45f14e2bc9f5)的DiversityRanker：
    
    1.  首先计算每个文档的嵌入向量，然后使用一个sentence-transformer模型进行查询搜索
    2.  将语义上和查询最接近的文档作为第一个选择的文档
    3.  对于剩下的每个文档，会计算和所选择文档的平均相似性
    4.  然后选择和所选择文档最不相似的文档
    5.  然后重复上述步骤，直到选出所有文档，这样就得到了一个整体多样性从高到低排序的文档列表。
    
    下面介绍了几种重排序的方式，称为**reranker**。
    
    ![image](https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/0ceff087-c27f-4959-bd82-c8a4c1e46ff8/image.png?t=1709793437)

*   [_LostInTheMiddleReranker_](https://arxiv.org/abs/2307.03172?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-retrieval)^[_Liu N F, Lin K, Hewitt J, et al. Lost in the middle: How language models use long contexts[J]. Transactions of the Association for Computational Linguistics, 2024, 12: 157-173._]：这是Haystack的另一个解决LLM不擅长从文档中间检索信息的问题的方法。它通过重排序来让最佳的文档位于上下文窗口的开头和结尾。建议在相关性和多样性之后再使用该Reranker。  
    ![image](https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/f4b74cc1-6df2-42bf-b016-43b1f50ec6f5/image.png?t=1709793561)
    
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240321181544635-1460620542.png)

*   [_**CohereRerank**_](https://txt.cohere.com/rerank/?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-retrieval)：通过Cohere的Rerank endpoint实现，它会将一开始的搜索结果和用户的请求进行比较，并基于请求文本和文档之间的语义相似性来重新计算结果，而不仅仅根据向量的查询结果。
    
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240322114840873-857003941.png)

*   [_bge-rerank_](https://www.llamaindex.ai/blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-retrieval)：除了要选择出最适合你的数据的嵌入模型外，还需要重点关注可以和该嵌入模型配合使用的检索器(retriever)。我们使用命中率和平均倒数排名(Mean Reciprocal Rank-MRR)作为retriever的评估指标。命中率是指在前k个检索到的chunks中找到正确答案的频率，MRR是排名中最相关的文档在排名中的的位置。从下面可以看到，JinaAI-Base嵌入中用到的bge-rerank-large 或 cohere-reranker 看起来非常适用于该数据集。[下表](https://www.llamaindex.ai/blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-retrieval)中需要注意的是，嵌入模型并不是最好的rerankers。Jina最擅长嵌入，而bge reranker则最擅长重新排序。  
  
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240325093713238-311446975.png)
    
*   _mxbai-rerank-v1_：最新的一种重排序模型是由 Mixedbread 团队开发的开源项目，称为 SOTA。其表现声称要优于 Cohere 和 bge-large。
    
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240325093951844-1474261260.png)

*   _[RankGPT](https://arxiv.org/pdf/2304.09542.pdf)_：这是使用现有的LLMs(如GPT3.5)重排检索文档的方法之一，且重排质量好于 Cohere。为了解决检索上下文大于 LLM 的上下文窗口的问题，该方法使用了一种"滑动窗口"策略，即在滑动窗口内实现渐进排名。这种方式击败了其他大部分 reranker，其中包括 Cohere。这种方式下需要注意的是延迟和成本，因此适用于优化小型开源模型。
    
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240325100601555-1649602187.png)
    

###### 提示压缩(Prompt Compression)

这里有必要介绍一下**Prompt Compression**，它和Reranking关系密切。这是一种通过压缩无关信息(即和用户查询无关)来降低检索到的文档的技术，有如下方式：

*   [_LongLLMLingua_](https://arxiv.org/abs/2310.06839?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-retrieval)^[_Jiang H, Wu Q, Luo X, et al. Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression[J]. arXiv preprint arXiv:2310.06839, 2023._]：该方式基于[Selective-Context](https://arxiv.org/abs/2310.06201?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-retrieval)^[_Li Y, Dong B, Lin C, et al. Compressing context to enhance inference efficiency of large language models[J]. arXiv preprint arXiv:2310.06201, 2023._] 和 [LLMLingua](https://arxiv.org/abs/2310.05736?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-retrieval)^[_Jiang H, Wu Q, Lin C Y, et al. Llmlingua: Compressing prompts for accelerated inference of large language models[J]. arXiv preprint arXiv:2310.05736, 2023._] 框架，是用于提示压缩的SOTA方法，针对成本和延迟进行了优化。除此之外，LongLLMLingua采用了一种"_采用问题感知由粗到细的压缩方法、文档重新排序机制、动态压缩比例以及压缩后的子序列恢复等策略来提升大型语言模型对关键信息的感知能力_"的方式来提升检索质量。该[方法](https://llmlingua.com/?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-retrieval)特别适用于长上下文文档，解决"[中间遗失](https://arxiv.org/abs/2307.03172?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-retrieval)"问题。
    
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240325151358985-1503032890.png)

*   [_RECOMP_](https://arxiv.org/abs/2310.04408?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-retrieval)^[_Xu F, Shi W, Choi E. Recomp: Improving retrieval-augmented lms with compression and selective augmentation[J]. arXiv preprint arXiv:2310.04408, 2023._]：使用"压缩器"，该方法使用文本摘要作为LLMs的上下文来降低成本并提升推理质量。两个压缩器分别表示： a)提取压缩器，用于从检索的文档中选择相关语句；b)抽象压缩器，用于根据多个文档的合成信息来创建摘要。
    
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240325152328876-404798371.png)

*   [_Walking Down the Memory Maze_](https://arxiv.org/abs/2310.05029?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-retrieval)^[_Chen H, Pasunuru R, Weston J, et al. Walking down the memory maze: Beyond context limit through interactive reading[J]. arXiv preprint arXiv:2310.05029, 2023._]：该方法引入了MEMWALKER的概念，即按照树的格式来处理上下文。该方法会对上下文划分chunk，然后对每个部分进行概括。为了回答一个查询，该模型会沿着树结构来迭代提示，以此来找出包含问题答案的段。特别适用于较长序列。
    
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240325153523127-1659336800.png)

###### RAG-fusion

这是 [Adrian Raudaschl](https://www.linkedin.com/in/adrian-raudaschl/?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-retrieval)提出的一种方法，使用[Reciprocal Rank Fusion (RRF)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-retrieval)和生成查询来提高检索质量。该方法会使用一个LLM根据输入生成多个用户查询，并为每个查询运行一个向量搜索，最后根据RRF聚合并优化结果。在最后一步中，LLM会使用查询和重排序列表来生成最终输出。这种方法在响应查询时具有更好的深度，因此非常流行。

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240325154026701-1203751509.png)

##### 改进

###### [CRAG](https://github.com/HuskyInSalt/CRAG?tab=readme-ov-file) (Corrective Retrieval Augmented Generation)

该方法旨在解决从静态和有限数据中进行次优检索的局限性，它使用一个轻量的检索评估器以及外部数据源(如web搜索)来补充最终的生成步骤。检索评估器会对检索的文档和输入进行评估，并给出一个置信度，用于触发下游的knowledge动作。

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240325154922321-2020702691.png)

从[下面](https://github.com/HuskyInSalt/CRAG?tab=readme-ov-file)报告结果可以看出其极大提升了基于RAG方法的表现：

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240325155347094-1244145526.png)

###### FLARE
[FLARE](https://arxiv.org/abs/2305.06983?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-retrieval) (Forward Looking Active Retrieval)^[_Jiang Z, Xu F F, Gao L, et al. Active retrieval augmented generation[J]. arXiv preprint arXiv:2305.06983, 2023._]特别适用于长文本生成，它会生成一个临时的"下一个语句"，并根据该语句是否包含[低概率tokens](https://www.linkedin.com/pulse/difference-between-high-probability-low-text-robert-koch/)来决定是否执行检索。该方法可以很好地决定什么时候进行检索，可以降低幻觉以及非事实的输出。

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240325155525370-857770978.png)

> 上图展示了FLARE的工作原理。当用户输入`x` ，并检索出结果Dx，FLARE会迭代生成一个临时的"下一句"(灰色字体)，然后检查它是否包含低概率的tokens(下划线表示)，如果包含(步骤2和3)，则系统会检索相关的文档并重新生成语句。

考虑到需要优化的变量数目，检索可能比较棘手，但通过对用户场景和数据类型进行映射，可以大大降低成本、延迟并提升准确性。

下面将面对文本生成(Generation)，并思考一些评估策略。

#### 生成和评估(Generation and Evaluation)

![](https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/afc0a4cc-7e4c-432c-a7cf-596503581e68/RAG_-_Generation.png?t=1710487820)

##### 生成(generation)

使用索引和检索可以保证输出的完整性，接下来，需要在为用户生成结果之前对结果进行评估，并通过决策步骤来触发相应的动作。Language Agents ([CoALA](https://arxiv.org/abs/2309.02427?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-generation-and-evaluation)) 的认知架构是一个很好的框架，可以将其放在上下文中，通过对检索到的信息的评估来获得一组可能性。如下图所示：

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240325171549803-747513033.png)

有如下几种方法可以实现这一动作选择决策程序：

###### Corrective RAG (CRAG)

在上一章(检索)中有提到过[Corrective RAG (CRAG)](https://arxiv.org/abs/2401.15884?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-generation-and-evaluation)^[_Yan S Q, Gu J C, Zhu Y, et al. Corrective retrieval augmented generation[J]. arXiv preprint arXiv:2401.15884, 2024._]，但它与本章节内容有所重叠，因此有必要展开介绍一下。

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240325154922321-2020702691.png)

CRAG使用轻量级"检索评估器"来增强generation，检索评估器会为每个检索到的文档生成一个置信值。该值确定了需要触发的检索动作。例如，评估器可以根据置信值来为检索到的文档标记到三个桶(正确、模糊、不正确)中的某个桶中。

如果所有检索到的文档的置信值都低于阈值，retriever会认为"不正确"，然后会通过触发外部知识源(如web搜索)动作来生成合格的结果。

如果至少有一个检索到的文档的置信值大于阈值，则会假设此次检索是"正确的"，然后会触发 knowledge refinement来改进检索到的文档。 knowledge refinement需要将文档拆分为"知识条(knowledge strips)"，然后根据相关性对每个strip打分，最相关的知识会被重新组合为generation使用的内部知识。

当检索评估器对自己的判断没有信心时会标记为"模糊"，此时会同时采用上述策略。见如下决策树：

![image](https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/dd6fa8c0-4e9a-4cb0-a54f-ca6fa0ef7e26/image.png?t=1710489189)

上述方法跨了四个数据来生成最佳结果。下表中可以看到，CRAG显著优于RAG。self-CRAG则使这些差距更加明显，并且展示了CRAG作为RAG流水线"即插即用"的适应性。另外一个CRAG由于其他方法(如Self-RAG)的点是它可以灵活地替换底层LLM，如果未来可能要采用更加强大的LLM，这一点至关重要。

![image](https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/bc711df1-bde3-4914-b2de-c0f2d2371207/image.png?t=1710489241)

CRAG的一个明显限制是它严重依赖检索评估器的质量，并且容易受到网络搜索可能引入的偏见的影响。因此可能需要对检索评估器进行微调，且需要分配必要的"护栏"来确保输出的质量和准确性。

###### Self-RAG

[Self-RAG](https://arxiv.org/abs/2310.11511?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-generation-and-evaluation)^[_Asai A, Wu Z, Wang Y, et al. Self-rag: Learning to retrieve, generate, and critique through self-reflection[J]. arXiv preprint arXiv:2310.11511, 2023._]是另一种可以提升LLM的质量和事实性，同时保持其多功能性的框架。与其检索固定数目的段落(不管这些段落是相关还是不相关的)，该框架则更关注按需检索和自我反省。

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240325174531200-209881446.png)

1.  将任一LLM训练为可以对自己生成的结果进行自我反省的模型(通过一种称为reflection tokens(retrieval 和 critique) 的特殊tokens)。检索是由retrieval token触发的，该token是由LLM根据输入提示和之前的生成结果输出的。
1.  然后LLM会通过并行处理这些检索到的段落来并发生成结果。该步骤会触发LLM生成用于评估这些结果的critique tokens。
2.  最后根据"真实性"和"相关性"来选择用于最终generation的最佳结果。论文中描述了算法，其中tokens的定义如下：  
    ![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240325175806896-1139728749.png)

下面Langchain的[示意图](https://blog.langchain.dev/agentic-rag-with-langgraph/?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-generation-and-evaluation)展示了基于上述定义的reflection token的Self-RAG 推理决策算法。

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240325180012273-1138595467.png)

就表现而言，无论是否进行检索，Self-RAG都显著优于基线(参考CRAG的Benchmark)。需要注意的是，可以通过在CRAG之上构建Self-CRAG来进一步提升模型表现。

该框架的成本和延迟受到LLM调用的数量的限制。可以通过一些变通方法，比如一次生成一代，而不是两次为每个块生成一代进行优化。

###### RRR (Rewrite-Retrieve-Read)

> _[RRR模式](https://arxiv.org/pdf/2305.14283.pdf?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-generation-and-evaluation)^[_Ma X, Gong Y, He P, et al. Query rewriting for retrieval-augmented large language models[J]. arXiv preprint arXiv:2305.14283, 2023._] [Ma et al., 2023a]引入了 重写-检索-读写 流程，利用 LLM 来强化 rewriter 模块，从而实现对检索查询的微调，进而提高 reader 的下游任务表现。_

该框架假设用户查询可以被LLM进一步优化(及重写)，从而实现更精确的检索。虽然这种方式可以通过LLMs的查询重写过程来提升表现，但也存在推理错误或无效查询之类的问题，因此可能不适用于部署在生产环境中。

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240326094543980-869476377.png)

为了解决上图中描述的问题，需要在rewriter上添加一个可训练的小型LLM(如上图红色部分所示)来持续提升训练的表现，使之兼具可扩展性和高效性。这里的训练包含两个阶段："预热"和增强学习，其关键是将可训练模块集成到较大的LLM中。

##### 评估

RAG pipeline中实现评估的方式有很多种，如使用一组Q&A作为测试数据集，并将输出与实际答案进行对比验证。但这种方式的缺点是不仅需要花费大量时间，并且增加了针对数据集本身优化pipeline的风险(而非可能存在很多边缘场景的现实世界)。

- 评测方法层面，可以检索和生成独立评测，也可以端到端进行评测
- RAG 的评估办法很丰富，主要包括三个质量分数：上下文相关度、答案忠实度、答案相关度
- 评估涉及四项核心能力：鲁棒性、拒识能力、信息整合能力和反事实解释
- 评估框架方面，有 RGB、RECALL 等基准指标，以及 RAGAS、ARES、TruLens 等自动化评估工具，可以较全面地衡量 RAG 模型的性能。
![](https://s2.loli.net/2024/03/15/EaT3tZ9hUzNB7Yv.png)


具体来说，评估方法主要围绕其两个关键组件：检索（Retrieval）和生成（Generation）。评估这些组件的性能涉及到多个质量分数和能力，这些分数和能力共同反映了 RAG 模型在信息检索和生成过程中的效率。

检索质量评估指标：
1. Hit Rate (HR)：命中率，衡量检索结果中相关文档的比例。高命中率意味着检索系统能够更准确地找到用户查询相关的信息。
2. Mean Reciprocal Rank (MRR)：平均倒数排名，衡量检索结果中相关文档的平均排名的倒数。MRR 越高，表示检索系统的性能越好。
3. Normalized Discounted Cumulative Gain (NDCG)：归一化折扣累积增益，用于衡量检索结果列表中相关文档的排名质量。NDCG 考虑了文档的相关性和排名位置。
4. Precision：精确率，衡量检索结果中被正确识别为相关的文档的比
5. Recall：召回率，衡量检索系统找到的相关文档占所有相关文档的比
6. R-Rate (Reappearance Rate)：再次出现率，衡量检索结果中信息在后续生成文本中的出现频率。

生成质量评估指标：
1. Context Relevance：上下文相关性，评估生成的文本与检索到的上下文之间的相关性。
2. Answer Faithfulness：答案忠实度，确保生成的答案忠实于检索到的上下文，保持一致性。
3. Answer Relevance：答案相关性，要求生成的答案直接相关于提出的问题，有效解决问题。
4. Accuracy：准确性，衡量生成的信息的准确性。

R
###### RAGAs

[RAGAs](https://blog.langchain.dev/evaluating-rag-pipelines-with-ragas-langsmith/)(_RAG Assessment_)是一个用于评估RAG pipeline的开源框架，它可以通过如下方式评估pipeline：

*   提供基于"[ground truth](https://domino.ai/data-science-dictionary/ground-truth)"(真实数据)来生成测试数据的方式
*   为检索和generation阶段以独立或端到端的方式提供基于指标的评估

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240326101441529-839833537.png)

它使用如下RAG特征作为指标：

1.  真实性：事实一致性
2.  回答相关性：问题和查询的相关性
3.  上下文准确性：校验相关chunks的排名是否更高
4.  批判(Critique)：根据预定义的特征(例如无害和正确性)评估提交
5.  上下文回顾：通过比较ground truth和上下文来校验是否检索到所有相关信息
6.  上下文实体回顾：评估上下文检索到的和ground truth中的实体数目
7.  上下文相关性：检索到的上下文和提示的相关性
8.  回答语义相似性：生成的答案和实际的语义相似性
9.  回答正确性：评估生成答案与实际答案的准确性和一致性

上面提供了一组可以全面评估RAG的列表，推荐阅读[**Ragas**](https://docs.ragas.io/en/stable/?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-generation-and-evaluation#)文档。

###### Langsmith

Langchain的[Langsmith](https://www.langchain.com/langsmith?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-generation-and-evaluation)可以作为上面上下文中的可观测性和监控工具。

> _[LangSmith](https://docs.smith.langchain.com/?ref=blog.langchain.dev&utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-generation-and-evaluation)_ 是一个可以帮助调试、测试、评估和监控基于任何LLM空间构建的chains和agents平台。

通过将Langsmith和RAG进行结合可以帮助我们更深入地了解结果。通过Langsmith评估logging和tracing部分可以更好地了解到对哪个阶段进行优化可以提升retriever或generator。

![image](https://img2024.cnblogs.com/blog/1334952/202403/1334952-20240326103621103-1611745663.png)

###### DeepEval

另一种值得提及的评估框架称为 [DeepEval](https://github.com/confident-ai/deepeval?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-generation-and-evaluation)。它提供了14种以上涵盖RAG和微调的指标，包括G-Eval、RAGAS、Summarization、Hallucination、Bias、Toxicity 等。

这些指标的自解释信很好地说明了指标评分的能力，使之更易于调试。这是跟上面的RAGAs的关键区别。

此外，它还有一些不错的特性，如集成[Pytest](https://docs.pytest.org/en/stable/contents.html?utm_source=div.beehiiv.com&utm_medium=referral&utm_campaign=advanced-rag-series-generation-and-evaluation) (开发者友好)，其模块化组件等也是开源的。

**Evaluate → Iterate → Optimize**

通过生成和评估阶段，我们不仅可以部署一个RAG系统，还可以评估、迭代并针对我们设计的特定场景进行优化。



### 总结与展望

#### 相关工作总结

**![](https://lh7-rt.googleusercontent.com/slidesz/AGV_vUfjnIeg74h1klbjyLlhYAV-bFUhaHYjMwt1ynD7A051RtNTrXwzRMytaaFNDnpM-21y15WRynSsDGcW28MEaFb2vuOK-AERkeOLhI0yilG0Ud6OzGeNK_HmmhKU0BgAYB2Awr0zsRj5XSZY2ZwXvfzyEutfpAM=nw?key=ZqWaY0jBtWrtvi15fPc-Fw)**

![](https://s2.loli.net/2024/03/15/CaWpDAZhPE753Jw.png)

#### 相关技术要点

<img src = './figures/8bcf9de22af50e0d74e7d5e4d0eabde.png'/>

#### RAG 技术框架
![](https://s2.loli.net/2024/03/15/Sk6mgsKFuzJPyrN.png)

#### RAG 的三个研究热点
![](https://s2.loli.net/2024/03/15/zj7AmBVOIe5WL2U.png)

#### RAG 的挑战
- 长上下文
- 与FT的协同
- 如何应用好LLM，充分挖掘利用LLM
- 提升鲁棒性，比如如何处理错误的召回内容，如何过滤和验证召回内容
- RAG 是否也遵循Scaling Law
- 最佳工程实践，比如提升在大数据量下的检索延迟，如何保障隐私的检索内容不被LLM泄露

![](https://s2.loli.net/2024/03/15/L9b6at48WTOnBDd.png)

#### 多模态扩展
将RAG从文本扩展到多模态
![](https://s2.loli.net/2024/03/15/rxOjqaiUZcJE1tD.png)

#### RAG 开发生态建设
扩展RAG下游任务，改善生态建设

![](https://s2.loli.net/2024/03/15/AJtDsSQp4iWTfVo.png)

#### RAG 技术栈与工业界实践#
当前有`LangChain`、`LlamaIndex`、`AutoGen`等流行的开发框架，可以方便开发RAG应用。

![](https://s2.loli.net/2024/03/15/OoHTMZzPR3IYjCx.png)

工业界也有很多RAG应用。

![](https://s2.loli.net/2024/03/15/MxZKr9WYy4UnSBv.png)

