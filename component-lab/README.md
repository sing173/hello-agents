# Hello-Agents Component Lab

这是一个独立的离线组件实验目录，用于拆解并验证 Hello-Agents 第八章“记忆与检索”和第九章“上下文工程”的核心概念。上游内容位于本目录的父目录；本实验没有修改上游源码。

- 上游仓库：`https://github.com/datawhalechina/Hello-Agents`
- 克隆基线：`2eb1c7547018b519b85cb57acf5533f12d65bf8e`
- 默认离线模式：Python 3.11+，仅使用标准库
- 可选真实模式：`qdrant-client`、`openai`、`neo4j`、`markitdown`、`python-dotenv` 安装到本目录 `.venv`
- 离线测试不调用 LLM、Qdrant、Neo4j、DashScope、Markdown 转换器或网络 API

## 快速运行

```bash
cd /root/Hello-Agents/component-lab

# 全部单元测试
python3 -m unittest discover -s tests -v

# 语法编译检查
python3 -m compileall -q components tests
```

无需 `pip install`，无需 `PYTHONPATH`，也无需创建 `.env`。在 `component-lab/` 根目录执行即可；若要安装为包，`pyproject.toml` 声明了零运行时依赖。

可选 Notebook：`notebooks/offline_walkthrough.ipynb`。已提交版本不包含输出；执行后另存为 `*.executed.ipynb` 会被 `.gitignore` 忽略。

## 真实模式（可选）

离线组件的默认行为不受影响。真实后端位于独立的 `real_backends/`，只有显式导入时才需要第三方包：

```bash
cd /root/Hello-Agents/component-lab
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[real]'
```

`real` extra 包含 MarkItDown 的 `pdf/docx/pptx/xlsx/xls` 格式依赖。PDF 由 `pdfminer-six` 与 `pdfplumber` 处理，不依赖 `pypdf`；图片 OCR、音频转写和云文档服务不属于当前安装范围。

Qdrant 使用外部运行的本地实例，不在每个测试中自动启动：

```bash
docker volume create hello-agents-qdrant
docker run -d \
  --name hello-agents-qdrant \
  --restart unless-stopped \
  -p 127.0.0.1:6333:6333 \
  -p 127.0.0.1:6334:6334 \
  -v hello-agents-qdrant:/qdrant/storage \
  qdrant/qdrant:latest

curl --noproxy '*' http://127.0.0.1:6333/healthz
.venv/bin/python scripts/smoke_qdrant.py
```

本次本机验证状态：容器 `hello-agents-qdrant` 运行中，重启策略 `unless-stopped`，命名 volume `hello-agents-qdrant`，镜像版本 v1.20.2，loopback 映射 `6333/6334`；`/healthz` 返回 `healthz check passed`。smoke 写入两条文本并保留在 collection `hello_agents_component_lab` 的 `component-lab-smoke` namespace。

`QdrantRAGStore` 使用单一 collection，`namespace` 映射为 payload filter；`document_id` 先映射为 UUIDv5 point id，重复写入相同内容幂等，同 ID 不同内容会报错。检索结果是 `QdrantHit`，包含 `document`、`score`、`source="qdrant"` 和 `collection`。向量由标准库确定性 feature hashing 生成，不是神经 embedding；语义质量低于真实 embedding。

Neo4j 已部署：容器 `hello-agents-neo4j`、命名卷 `hello-agents-neo4j-data`、仅回环绑定的 `127.0.0.1:7474/7687`，重启策略 `unless-stopped`。官方镜像 `neo4j:5.26-community` 对应服务版本 **5.26.34 Community**。随机密码只写入已被 `.gitignore` 忽略的 `.env`（权限 `600`），示例变量见 `.env.example`：

```bash
# .env 变量名：NEO4J_URI、NEO4J_HTTP_URL、NEO4J_USER、NEO4J_PASSWORD、
#             NEO4J_DATABASE、NEO4J_NAMESPACE
curl --noproxy '*' http://127.0.0.1:7474/
.venv/bin/python scripts/smoke_neo4j.py
.venv/bin/python -m unittest discover -s tests_real -p 'test_neo4j.py' -v
```

`Neo4jGraphStore` 用 `MERGE` 按 `namespace + entity_id/relation_id` 幂等写入；namespace 是节点和关系属性，检索与邻居查询都精确过滤。关系必须显式提供 `source` 与 `evidence`，端点必须已存在于同一 namespace。关系 ID 不能改绑到不同端点或关系类型。`health()` 验证 Bolt 并返回 Neo4j 版本/edition，不返回凭据。

MarkItDown 将本地文档转换为 Markdown 文本，保持调用方传入的 `source_path`，并按扩展名返回小写 `format`。当前安装支持文本/Markdown/JSON/JSONL、HTML、PDF、DOCX、PPTX、XLSX、XLS；自动测试覆盖 Markdown、TXT、HTML，临时生成的 DOCX/PPTX/XLSX smoke 也通过。没有 PDF fixture，因此本次 PDF 转换未验证。

```python
from components.rag import RAGTool
from real_backends.markitdown_adapter import MarkItDownAdapter

loaded = MarkItDownAdapter().load_document("data/sample_knowledge.md")
rag = RAGTool()
document = rag.add_text(
    loaded.content,
    doc_id="sample-knowledge",
    namespace="documents",
    metadata={
        "source": "markitdown",
        "source_path": loaded.source_path,
        "format": loaded.format,
    },
)
```

`load_document` 对不存在、非文件、转换失败和空内容分别给出明确错误；不会回退为静默空字符串。Qdrant 也可用相同方式把 `loaded.content` 交给 `QdrantRAGStore.add_text(document_id=...)`。

OpenAI 兼容 LLM 从 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL_ID` 读取配置。key 可缺省，此时发送占位 bearer，供匿名本机网关使用：

```bash
export LLM_BASE_URL='http://127.0.0.1:3458/v1'
export LLM_MODEL_ID='<general-chat-model-id>'
export LLM_API_KEY='<set only in your shell>'
.venv/bin/python scripts/smoke_llm.py
```

`scripts/smoke_llm.py` 只输出 endpoint、模型、延迟、token 用量和回答字符数，不输出 key 或回答正文。不要提交 `.env`；`.env.example` 只有占位符。真实 smoke 的失败不会改变离线模式的回退行为，直接不要导入 `real_backends` 即可保持纯标准库运行。

## 目录

```text
component-lab/
├── README.md
├── COMPONENT_PLAN.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── components/
│   ├── text.py
│   ├── memory/
│   ├── rag/
│   ├── context/
│   ├── notes/
│   └── terminal/
├── real_backends/
│   ├── qdrant_store.py
│   ├── llm.py
│   ├── neo4j_store.py
│   └── markitdown_adapter.py
├── scripts/
│   ├── smoke_qdrant.py
│   ├── smoke_llm.py
│   ├── smoke_neo4j.py
│   └── smoke_markitdown.py
├── tests/
├── tests_real/
├── notebooks/
├── data/
└── runs/
```

`runs/`、SQLite/数据库文件、缓存、Notebook 检查点与执行输出均已加入忽略规则。`data/sample_knowledge.md` 仅供离线检索演示。

## 已实现接口

| 组件 | 主要入口 | 行为 |
| --- | --- | --- |
| 记忆 | `MemoryManager.add/search/update/remove/forget/consolidate` | 进程内、按 `scope` 隔离；相同 scope/type/规范化内容幂等；支持重要性、时间、容量遗忘；按阈值提升记忆类型 |
| 记忆工具 | `MemoryTool.run/execute` | 教程风格 action façade，返回确定性 JSON 字符串 |
| RAG | `RAGTool.add_text/search/ask_skeleton` | 内存索引；TF-IDF cosine + Jaccard + 查询覆盖率；命名空间隔离；`ask_skeleton` 只生成证据化提示词，不生成答案 |
| 上下文 | `ContextBuilder.gather/select/structure/compress/build` | GSSC 四阶段；系统指令保留；贪心预算选择；固定模板；确定性截断压缩 |
| 笔记 | `NoteTool.create/read/update/delete/list_notes/search` | Markdown + 受限 YAML front matter；原子更新；按 ID 防目录穿越；拒绝符号链接逃逸 |
| 终端 | `TerminalTool.run` | 只读命令白名单；参数数组调用；`shell=False`；拒绝绝对路径、`..`、shell 元字符、符号链接逃逸和危险选项 |
| Qdrant 真实后端 | `QdrantRAGStore.add_text/search/stats/health` | 可选导入；payload namespace 过滤；UUIDv5 幂等；返回 source/score；确定性 feature-hash 向量 |
| LLM 真实后端 | `OpenAICompatibleLLM.list_models/health/chat` | 可选导入；读取 `LLM_*`；支持匿名占位 key；聊天必须显式限制 max tokens |
| Neo4j 真实后端 | `Neo4jGraphStore.health/upsert_entity/upsert_relation/search_entities/neighbors` | 可选导入；参数化 Cypher；namespace 隔离；MERGE 幂等；关系带 source/evidence |
| MarkItDown 真实后端 | `MarkItDownAdapter.load_document` | 可选导入；返回 content/source_path/format；明确错误；可衔接到现有 `add_text` |

## 关键语义

- **记忆作用域**：`scope` 是精确匹配字符串，不做继承。`global`、`user:alice`、`session:123` 都是不同隔离域。跨域读取/更新按不存在处理。
- **幂等添加**：同一作用域、同一类型、规范化后相同的内容返回已有记录；首次写入的重要性、标签和元数据不会被后续重复添加覆盖。
- **整合与遗忘**：`consolidate` 只改变记忆类型，不合并正文；`forget` 是不可逆删除。时间遗忘以 `updated_at` 为准，容量遗忘按重要性和更新时间保留高价值记录。
- **离线“语义近似”**：没有向量模型。检索通过大小写折叠、英文轻量词干、中文单字/相邻双字、Jaccard 与 TF-IDF 近似词形相关查询；不能替代真实 embedding 的语义理解。
- **上下文预算**：`ContextConfig.available_tokens = max_tokens × (1 - reserve_ratio)`。选择阶段不超过该预算；结构化后的模板开销由压缩兜底。关闭压缩且超限时抛出 `ContextBudgetError`，不静默超限。
- **笔记 YAML 子集**：只支持扁平键值、JSON 风格字符串/数组，以及上游示例中的 `tags: [a, b]`。不支持嵌套映射、锚点、别名、标签对象或多行块。
- **终端安全**：命令和选项都需白名单；所有路径必须位于 workspace；不做 shell 展开、管道、重定向、引号分组、glob 或解释器调用。它是应用层约束，不是操作系统级沙箱。
- **Qdrant 存储映射**：默认 collection `hello_agents_component_lab`，可用 `QDRANT_COLLECTION` 覆盖；namespace 存于 payload 并用精确 filter 隔离。query 与文档都使用同一 signed feature-hash 编码，所以是真实 Qdrant 余弦检索，但不是神经语义检索。
- **真实 LLM 鉴权**：`LLM_API_KEY` 只从环境读取；未设置时发送占位 bearer。适配器不会输出请求头或回答正文，错误只保留异常类型和 HTTP 状态。
- **Neo4j 图隔离**：节点 label 固定为 `ComponentLabEntity`、关系 type 固定为 `ComponentLabRelation`；namespace 与业务 id 都是属性。实体类型和关系类型只作为参数值，不拼接 Cypher 标识符。
- **文档来源**：MarkItDown 的 `source_path` 原样保留；送入 RAG 时应同时写入 `source_path/format/source` metadata，避免转换后丢失出处。

## 测试覆盖

`tests/` 使用 `unittest`、临时目录和固定时钟，覆盖：

- 记忆幂等、scope 隔离、更新/删除、三种遗忘策略、整合阈值与重复整合
- RAG 精确查询、词形近似、中文近似、命名空间、重复写入、`ask_skeleton` 无答案行为
- ContextPacket/Config 校验、预算内选择、超预算跳过、系统指令保留、模板顺序、压缩不超限
- 笔记 CRUD、跨实例持久化、手工 YAML 子集、路径穿越、符号链接逃逸、格式错误
- Terminal 白名单、绝对路径、父目录、shell 元字符、危险选项、符号链接逃逸、`shell=False`、输出截断
- 组件导入仅限标准库或本地 `components`

原有 40 项离线测试保持不变，不需要 `.venv` 或任何服务。可选 adapter 测试和真实 smoke 使用独立目录/脚本：

```bash
.venv/bin/python -m unittest discover -s tests_real -v
.venv/bin/python scripts/smoke_qdrant.py
.venv/bin/python scripts/smoke_llm.py
.venv/bin/python scripts/smoke_neo4j.py
.venv/bin/python scripts/smoke_markitdown.py
# 可选：对任意已有文档做转换 smoke，包括已有 PDF fixture
.venv/bin/python scripts/smoke_markitdown.py /path/to/document.pdf
```

`tests_real/` 不在默认 `tests/` 发现范围内，因此系统 Python 的离线测试仍只依赖标准库。新增测试覆盖 Neo4j 实体/关系幂等、namespace 隔离、邻居方向/类型/limit，以及 MarkItDown Markdown/TXT/HTML、错误路径和 RAG 衔接；本次 5 项 Neo4j 真实测试全部通过。服务不可用时集成测试仍会明确 skip，`smoke_neo4j.py` 会返回失败状态。

## 未实现与后续方向

- 神经 embedding、混合检索、真实 Qdrant 快照/备份与多 collection 策略
- SQLite 记忆持久化、多模态图片 OCR/音频转写、PDF fixture 验证与 Office 全格式回归
- 真实 LLM 问答编排、MQE/HyDE、LLM 摘要压缩、Agent 工具注册表
- 笔记全量 YAML 与索引文件、并发写入锁、跨进程一致性
- 终端操作系统级沙箱、资源限制和完整的 TOCTOU 防护

详细的上游组件清单、外部依赖和 API 漂移见 [COMPONENT_PLAN.md](COMPONENT_PLAN.md)。