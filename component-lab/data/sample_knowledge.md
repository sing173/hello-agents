# Sample knowledge for the offline RAG lab

Pandas memory optimization commonly starts with choosing compact dtypes. Replacing
repeated strings with categorical values and downcasting numeric columns often reduces
RAM substantially.

Agent memory can be scoped by user or session. Important working memories may be
consolidated into episodic memories, while low-importance records can be forgotten.

记忆系统负责保存用户偏好，并支持按作用域检索。上下文工程在固定预算内选择证据，
再通过结构化模板和确定性压缩保持输出可控。