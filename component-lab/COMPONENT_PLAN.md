# 组件清单与离线化决策

## 1. 上游基线

- 仓库：`https://github.com/datawhalechina/Hello-Agents`
- 克隆路径：`/root/Hello-Agents`
- Commit：`2eb1c7547018b519b85cb57acf5533f12d65bf8e`
- 阅读范围：
  - `README.md`
  - `docs/chapter8/第八章 记忆与检索.md`
  - `docs/chapter9/第九章 上下文工程.md`
  - `code/chapter8/*.py`
  - `code/chapter9/*.py`、`code/chapter9/README.md`、`code/chapter9/codebase/*`

上游没有根级 `pyproject.toml` 或 `requirements.txt`。仓库内能定位到的依赖文件只属于第六章和第十章子示例；第八、九章依赖的是外部安装的 `hello-agents[all]` 包和若干服务，而不是本仓库内的可安装源码。

教程自身存在版本不一致：第八章安装 `hello-agents[all]==0.2.0`（并注明可切到 `0.2.9`），第九章安装 `hello-agents[all]==0.2.8`。所以本实验只以教程中的接口叙述为设计参考，不声称复刻某个确切发行版的内部实现。

## 2. 上游组件清单

| 组件 | 上游位置/概念 | 上游职责 | 外部依赖 | 离线骨架结论 |
| --- | --- | --- | --- | --- |
| `MemoryTool` | 第八章 8.2.2–8.2.4；`code/chapter8/01`、`02`、`08` | Agent 面向的 action 入口：add/search/summary/stats/update/remove/forget/consolidate | `hello-agents`、`python-dotenv` | 适合先做；用 JSON façade 取代 Agent 工具注册 |
| `MemoryManager` | 第八章 8.2.4；`code/chapter8/03`、`06`、`09` | 统一协调四类记忆、评分、遗忘、整合 | `hello-agents` | 已做内存版；scope、阈值与生命周期可离线验证 |
| 工作/情景/语义/感知记忆 | 第八章 8.2.5；`code/chapter8/09` | 不同 TTL、持久性和检索策略 | `hello-agents` | 骨架保留四种类型标签，但不实现各类型专属存储 |
| Qdrant 向量存储 | 第八章 8.1.3、8.3.3；环境变量示例 | 向量相似检索、命名空间隔离 | Qdrant 容器；`qdrant-client` | 已增加可选 `QdrantRAGStore`；离线路径仍不依赖 Qdrant |
| Neo4j 图存储 | 第八章 8.1.3；环境变量示例 | 语义记忆的实体关系与知识图谱 | Neo4j Aura 或 Docker；`neo4j` 驱动 | 已增加可选 `Neo4jGraphStore`；namespace 隔离、MERGE 幂等、邻居与证据关系 |
| SQLite 文档存储 | 第八章 8.1.3 | 结构化记忆/文档持久化 | 标准库 `sqlite3` 或外部封装 | 未实现；所有离线记忆只存在进程内 |
| Embedding 服务 | 第八章 8.1.3、8.1.4 | DashScope、LocalTransformer 或 TF-IDF 嵌入 | `dashscope`、`sentence-transformers`、模型权重 | 未实现神经 embedding；Qdrant 模式使用确定性 feature hashing |
| `RAGTool` | 第八章 8.3；`code/chapter8/04`、`05`、`07`、`10`、`11` | 文档接入、检索、问答、知识库管理 | `hello-agents`、MarkItDown、Qdrant、embedding、LLM | 离线版仅 `add_text/search/ask_skeleton`；真实向量由独立 Qdrant adapter 提供 |
| MarkItDown 文档处理 | 第八章 8.3.4；`code/chapter8/04` | PDF/Office/图片/音频转 Markdown | `markitdown`，PDF/OCR/转写附加能力 | 已增加可选 `MarkItDownAdapter`；当前启用 PDF/Office 子集，保持原始 source_path 并衔接 `add_text` |
| MQE / HyDE | 第八章 8.3.5；`code/chapter8/05` | 查询扩展与假设文档检索 | 真实 LLM + 向量检索 | 未实现；需要真实模型，不适合纯离线首版 |
| `ContextBuilder` | 第九章 9.3；`code/chapter9/01`、`02` | GSSC：Gather-Select-Structure-Compress | `hello-agents.context`、`hello_agents.core.message`、Memory/RAG、LLM（集成示例） | 最适合离线骨架；核心逻辑已实现 |
| `ContextPacket`/`ContextConfig` | 第九章 9.3.2 | 候选信息、token 预算、评分阈值与权重 | 无 | 已实现，并增加显式校验 |
| `NoteTool` | 第九章 9.4；`code/chapter9/03`、`04` | Markdown + YAML 前置元数据笔记 CRUD/search/list/summary | `hello-agents`、PyYAML | 适合离线；实现了受限 YAML 子集 CRUD |
| `TerminalTool` | 第九章 9.5；`code/chapter9/05` | 白名单只读文件系统探索、`cd`、超时、输出限制 | `hello-agents`、系统命令 | 可离线，但必须重写安全模型：上游示例使用 `shell=True` |
| Agent 集成 | 第八章 `08_Agent_Tool_Integration.py`；第九章 `04`、`06` | `SimpleAgent`/`FunctionCallAgent`、ToolRegistry、LLM | `hello-agents`、LLM API | 未实现，避免联网和模型依赖 |

## 3. 外部依赖归类

### 3.1 服务/API 依赖

- Qdrant Cloud 或本地 Docker：向量索引。
- Neo4j Aura 或本地 Docker：图存储。
- DashScope 百炼、或本地 `sentence-transformers`：embedding。
- 任意 OpenAI 兼容 LLM 服务：问答、MQE、HyDE、摘要压缩和 Agent 演示。
- spaCy 模型 `zh_core_web_sm`、`en_core_web_sm`：教程环境准备步骤。

### 3.2 Python 包依赖（教程示例导入）

- `hello-agents[all]`：第八、九章实际使用的框架入口，不在本仓库源码内。
- `python-dotenv`：加载 `.env`。
- `PyYAML`：上游 NoteTool 的 front matter 序列化/解析。
- `qdrant-client`、`neo4j`：存储驱动（由框架内部使用）。
- `markitdown`：多格式文档转 Markdown。
- `dashscope`、`sentence-transformers`：embedding 后端。
- `spacy` 及语言模型：教程环境。
- `gradio`、`pandas`、`requests`：问答界面与示例代码库。
- Docker（可选）：本地运行 Qdrant/Neo4j。

系统 Python 环境没有安装上述包。`pyproject.toml` 的默认 `dependencies` 仍为空，并有测试扫描 `components/` 的导入，只允许标准库和本地 `components`。可选真实依赖组 `real` 安装到 `component-lab/.venv`，不会污染系统环境。本次 `real` 新增 `neo4j>=5.26,<6` 与 `markitdown[pdf,docx,pptx,xlsx,xls]>=0.1.9,<1`；PDF 使用 `pdfminer-six`/`pdfplumber`，不需要 `pypdf`。

## 4. API 漂移与风险

1. **教程内部版本不一致**：第八章与第九章分别要求 `0.2.0` 和 `0.2.8`；不能假定两者行为完全相同。
2. **MemoryTool 调用形式不统一**：正文有的位置展示 `execute("search", ...)`，配套代码使用 `run({"action": "search", ...})`。本实验同时提供 `run`/`execute`，并接受字符串 action 或映射。
3. **包源码不可见**：本仓库只有教程与调用示例，真正的 `hello_agents` 实现由外部包提供；因此参数默认值、异常类型和返回文本无法从本仓库精确锁定。
4. **RAGTool 参数较丰富**：教程展示 `knowledge_base_path`、`qdrant_url`、`rag_namespace`、`enable_advanced_search`、MQE/HyDE 等；本实验只保留可离线验证的最小接口。
5. **ContextBuilder 细节**：
   - 上游当 `relevance_score == 0.5` 时会重新计算相关性；本实验始终尊重调用方提供的分数。
   - 上游选择阶段遇到第一个装不下的包就停止；本实验跳过超大包并继续尝试后续小包。
   - 上游压缩以 `max_tokens` 为目标；本实验以扣除预留后的 `available_tokens` 为硬上限。
   - 上游用空白分词计算相关性；本实验使用中英文兼容的确定性词法工具。
6. **NoteTool 格式**：上游通过 PyYAML 处理完整 YAML，并维护 `notes_index.json`；本实验使用受限扁平 YAML 子集，不维护索引，`list_notes` 扫描文件。
7. **TerminalTool 安全冲突**：上游教程实现示例明确使用 `subprocess.run(..., shell=True)`，且允许 `cd`、管道、通配符和复杂参数。本实验按任务要求改为只读白名单 + 参数数组 + `shell=False`，因此不支持上游示例中的管道、glob、引号分组和 `cd`。这是有意的安全偏离。
8. **真实语义差距**：本实验的“语义近似”仅是词形和词项重叠近似，无法匹配同义词、跨语言语义或复杂指代。不能把离线分数解释为 embedding 相似度。

## 5. 真实后端（可选，2026-11-26）

实现位于 `real_backends/`，不从 `components/` 导入，因此原有离线包和 40 项测试保持无第三方依赖。

### 5.1 Qdrant

本次运行环境：Docker `29.8.1`、Compose `v5.5.1`、根分区可用约 `36G`；启动前 6333/6334/7474/7687 均无监听，仅有占用 3458 的 Router 容器。官方镜像 `qdrant/qdrant:latest`（版本标签 `v1.20.2`，digest `sha256:d277592721d2e3b73f9b797b53cf05e6a5bb2134f6d13fdbda4ca51782afbcac`）已拉取并部署：

- 容器：`hello-agents-qdrant`
- Volume：`hello-agents-qdrant` → `/qdrant/storage`
- 端口：`127.0.0.1:6333 -> 6333/tcp`、`127.0.0.1:6334 -> 6334/tcp`
- 重启策略：`unless-stopped`
- 健康检查：`GET /healthz` → `healthz check passed`
- 真实验证：`scripts/smoke_qdrant.py` 写入 `smoke-pandas`/`smoke-qdrant`，重复写入幂等，namespace 点数保持 2；查询 top hit 为 `smoke-qdrant`，`source=qdrant`，score `0.707107`

- `real_backends/qdrant_store.py::QdrantRAGStore`
- 可选依赖：`qdrant-client>=1.12,<2`，本次安装 `1.20.0`
- 单一 collection（默认 `hello_agents_component_lab`），namespace 写入 payload 并使用 `Filter` 精确隔离
- document id 经 UUIDv5 映射为 point id；重复写入同 namespace/id/content 幂等，同 id 不同内容抛出 `ValueError`
- 返回 `QdrantHit(document, score, source="qdrant", collection)`
- 向量：signed feature hashing，384 维并 L2 归一化；真实 Qdrant 负责点存储、payload 过滤和 cosine 检索。它不是神经 embedding
- API 漂移：`qdrant-client 1.20.0` 已没有 `QdrantClient.search`，实现使用 `query_points(...).points`
- 检索只支持向量相似度；没有实现 Qdrant 全文索引、混合检索、batching 或快照管理
- `tests_real/test_backends.py` 使用 `QdrantClient(location=":memory:")` 验证幂等、namespace、source 和 hash embedding；不替代真实容器 smoke

### 5.2 OpenAI 兼容 LLM

- `real_backends/llm.py::OpenAICompatibleLLM`
- 可选依赖：`openai>=1.40,<3`，本次安装 `2.54.0`
- 读取 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL_ID`；key 为空时发送占位 bearer，适配匿名本机网关
- 提供 `list_models`、`health`、`chat`；chat 强制调用方给出 1–4096 的 `max_tokens`
- 错误只暴露异常类型和 HTTP 状态；smoke 输出不包含 key、请求头或回答正文
- 2026-11-26 检测：`http://127.0.0.1:3458/v1/models` 使用本机路由器既有凭据返回 12 个模型；匿名请求返回 401
- 模型测试中 `DeepSeek/*`、`kilo/*` 路由返回 401，多个 OpenRouter 路由不可用；可用通用聊天路由选择为 `OpenRouter/recursive/flint-v1.0:free`
- 该路由会先产生 reasoning tokens，再返回正文；不是 reasoning-only，但小 token 上限可能只得到空正文。最终 smoke 使用 64 token 上限，延迟约 6.0 秒、返回正文 5 字符
- 没有实现流式输出、重试、tool calling、RAG 答案生成或模型自动选择

### 5.3 Neo4j

- `real_backends/neo4j_store.py::Neo4jGraphStore`
- 可选依赖：`neo4j>=5.26,<6`，本次安装 `5.28.8`
- Docker 部署：容器 `hello-agents-neo4j`、命名卷 `hello-agents-neo4j-data`、端口仅绑定 `127.0.0.1:7474/7687`、重启策略 `unless-stopped`
- 当前状态：2026-11-27 已部署官方 `neo4j:5.26-community`，服务版本 `5.26.34 Community`，digest `sha256:c5fd91e41d44858ea678813fd53f8e60602e771a7656c502385247ffbcb0aec0`
- Docker daemon 通过 systemd drop-in 使用 `http://192.168.1.16:7890`；`NO_PROXY` 包含 localhost、回环地址和 `192.168.1.0/24`
- 启动前检查：无同名容器/卷、无 7474/7687 监听、根分区有足够空间；未覆盖已有数据
- 随机密码只写入 `component-lab/.env`（`.gitignore` 已忽略，权限 `600`）；示例仅含占位符
- 提供 `health`、`upsert_entity`、`upsert_relation`、`search_entities`、`neighbors`，并支持 `close()` 与上下文管理器
- 固定 label/type，namespace、业务 id、实体类型和关系类型都是绑定参数；不使用用户输入拼接 Cypher 标识符
- `MERGE` 以 namespace/id 保证实体与关系幂等；关系带 `source/evidence`，端点必须已存在于同一 namespace；关系 id 不能改绑端点或类型
- `health()` 验证 Bolt 并读取 `dbms.components()`；smoke 另行检查 HTTP 根端点，输出不包含密码
- Neo4j Python 驱动 `5.28.8` 的 `execute_query()` 使用 `RoutingControl.READ/WRITE`；session 仍使用 `READ_ACCESS/WRITE_ACCESS`
- `tests_real/test_neo4j.py` 使用真实容器验证健康、实体/关系幂等、namespace 隔离、邻居方向/类型/limit 和关系身份冲突；本次 5 项全部通过

### 5.4 MarkItDown

- `real_backends/markitdown_adapter.py::MarkItDownAdapter`
- 可选依赖：`markitdown[pdf,docx,pptx,xlsx,xls]>=0.1.9,<1`，本次安装 `0.1.9`
- `load_document(file_path) -> LoadedDocument(content, source_path, format)`；`source_path` 原样保留，format 为小写扩展名
- 当前启用文本/Markdown/JSON/JSONL、HTML、PDF、DOCX、PPTX、XLSX、XLS 转换；图片 OCR、音频转写和云服务未启用
- 对文件不存在、非普通文件、转换失败、空内容给出明确错误；不静默返回空文档
- 通过把 `loaded.content` 与来源 metadata 交给现有 `RAGTool.add_text` 或 `QdrantRAGStore.add_text` 完成衔接
- `tests_real/test_markitdown.py` 实际验证 Markdown/TXT/HTML、错误路径与 RAG 幂等衔接；临时生成 DOCX/PPTX/XLSX 的转换 smoke 通过
- 项目没有已有 PDF fixture，本次 PDF 转换未验证；Office 依赖已安装并通过 DOCX/PPTX/XLSX smoke，但未做旧版 XLS 与全格式 fixture 回归

## 6. 离线骨架的边界

已完成：

- 记忆：内存记录、作用域、幂等添加、词法检索、更新/删除、重要性/时间/容量遗忘、类型整合。
- RAG：命名空间、幂等入库、TF-IDF/Jaccard 检索、精确短语标记、中文近似、无 LLM 的证据提示词骨架。
- 上下文：系统指令优先保留、相关性/新近性评分、预算贪心选择、固定模板、确定性压缩。
- 笔记：Markdown + YAML 子集、原子更新、CRUD、搜索、目录穿越和符号链接防护。
- 终端：只读命令与选项白名单、workspace 路径限制、`shell=False`、shell 元字符拒绝、输出上限、超时。
- 真实后端：可选 Qdrant payload/namespace 检索、幂等 point 写入、source/score；可选 OpenAI 兼容模型列表与有界 chat；可选 Neo4j 图存储与 MarkItDown 文档转换。

未完成：

- 神经 embedding、Qdrant 混合/全文检索、持久化记忆与迁移。
- 已接入 MarkItDown 文本/Markdown/HTML/PDF/Office 子集；仍需 PDF fixture、旧版 XLS/全格式回归、图片 OCR/音频转写、分块、嵌入与重排。
- 基于真实检索结果的 LLM 答案生成、查询扩展、答案评估。
- 完整 YAML、索引一致性、并发锁、跨进程写入。
- OS 级终端沙箱、seccomp/namespaces、TOCTOU 防护、细粒度资源限制。
- 与 Hello-Agents Agent/ToolRegistry 的直接适配层。

## 7. 推荐演进顺序

3. 用神经 embedding 和可替换 scorer 增强 Qdrant 检索；保留当前 feature-hash adapter 作为无模型回退。
4. 在已接入 MarkItDown 的基础上增加分块器与 PDF/Office fixture 回归，保持 `add_text` 作为最终公共入口。
5. 基于 Qdrant 检索与真实 LLM 增加答案编排和 MQE/HyDE；答案生成必须有超时、错误分类和离线回退。
6. 终端若进入生产，应置于 OS 沙箱中，而不是仅依赖本实验的 argv 白名单。

## 8. 验证命令

```bash
cd /root/Hello-Agents/component-lab

# 离线模式，系统 Python
python3 -m unittest discover -s tests -v
python3 -m compileall -q components tests

# 可选真实模式
.venv/bin/python -m unittest discover -s tests_real -v
.venv/bin/python -m compileall -q components real_backends scripts tests tests_real
.venv/bin/python scripts/smoke_qdrant.py
.venv/bin/python scripts/smoke_neo4j.py
.venv/bin/python scripts/smoke_markitdown.py
# 可选：对已有 PDF fixture 做转换 smoke
.venv/bin/python scripts/smoke_markitdown.py /path/to/document.pdf

LLM_BASE_URL='<base-url>' \
LLM_MODEL_ID='<general-chat-model-id>' \
LLM_API_KEY='<from-secure-shell-or-anonymous>' \
.venv/bin/python scripts/smoke_llm.py
```

本目录的创建不包含 Git 提交或推送，上游文件未改动。