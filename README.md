# 装配线低换型工单排程 API（纯后端）

Python 3.12 + FastAPI 实现。给定工单及其 `before → after` 前置关系，输出
**满足全部前置** 的完整作业顺序，并按以下两级目标精确优化：

1. **最小化换型次数**：相邻两张工单的 `family` 不同即计一次换型，首单不计；
2. **字典序最小**：换型次数相同的方案中，按工单 id 的 UTF-8 字节序取最小的顺序。

若前置图存在有向环，返回 `CYCLE` 与一条**真实有向环**（从环中最小 id 开始，
逐边闭合），且**不返回任何部分排程**。

精确算法：位掩码子集动态规划（n ≤ 18），不是贪心或仅拓扑排序。

---

## 目录结构

```
app/
  main.py        FastAPI 路由与 422 语义校验
  models.py      Pydantic 请求/响应模型（extra="forbid"）
  scheduler.py   环检测（Kahn + 回溯取证）与位掩码 DP
tests/
  test_scheduler_optimality.py  小图穷举全部拓扑序对拍
  test_cycles.py                环证据校验（每条边必须存在）
  test_api.py                   HTTP 层与全部 422 场景
Dockerfile
docker-compose.yml
```

## 快速开始

```bash
docker compose up --build
```

服务监听 `http://localhost:8000`，交互文档在 `/docs`。

健康检查：

```bash
curl localhost:8000/health
# {"status":"ok"}
```

### 请求示例

```bash
curl -X POST localhost:8000/schedule -H 'Content-Type: application/json' -d '{
  "jobs": [
    {"id": "a", "family": "RED"},
    {"id": "b", "family": "RED"},
    {"id": "c", "family": "BLUE"},
    {"id": "d", "family": "RED"}
  ],
  "edges": [
    {"before": "a", "after": "c"}
  ]
}'
```

响应（`200`）：

```json
{
  "status": "OK",
  "order": ["a", "b", "d", "c"],
  "changeover_count": 1,
  "changeovers": [
    {"position": 4, "job_id": "c", "from_family": "RED", "to_family": "BLUE"}
  ]
}
```

- `order`：全部工单的完整顺序（`a,b,d` 连续 RED，最后 `c` 为 BLUE）；
- `changeover_count`：换型总数；
- `changeovers[].position`：换型发生在 `order` 中的 **1-based 位置**
  （首单位置 1，永不出现；上例中第 4 张工单 `c` 是唯一换型点）。

### 环响应示例

```json
{
  "status": "CYCLE",
  "order": [],
  "changeover_count": 0,
  "changeovers": [],
  "cycle": ["a", "b", "c"]
}
```

`cycle` 中相邻元素构成一条真实输入边，最后一个元素回到第一个
（`a→b, b→c, c→a`），且 `cycle[0]` 是该环上最小的 id。

## 输入约束与 422

请求体：

```json
{
  "jobs":  [{"id": "<ASCII 非空>", "family": "<ASCII 非空>"}],
  "edges": [{"before": "<id>", "after": "<id>"}]
}
```

- 工单数 **2 至 18**；id 唯一；family 非空；所有字符串仅可含 ASCII 字符；
- 边数至多 **100**；以下情况一律返回 **422**：
  - 自环（`before == after`）；
  - 重复边（相同的 `before → after` 出现两次）；
  - 边引用了未知工单 id；
  - 任意层级的额外/未知字段（`extra="forbid"`）；
  - 空 id/空 family、非 ASCII、类型错误、数量越界、缺字段、重复工单 id。

## 算法说明

### 为什么朴素方法不够

- 只做拓扑排序：无法顾及连续同 family 生产，产生大量无谓换型；
- 先按 family 分组：可能违反 `before → after` 前置。

两级目标需要精确组合优化。

### 位掩码 DP

工单按 id 的 UTF-8 字节序编号（编号序即字典序）。`S` 为已排工单的位掩码，
仅考虑前置闭包（ideal）集合：凡 `S` 中成员的前驱都在 `S` 内。
状态 `(S, k)` 表示“恰好排完 `S`、末单为 `k`”的最佳序列，记录：

- 最小换型数 `cost`；
- 末单的前一张工单（回溯指针）；
- 该层 `|S|` 内的稠密字典序名次 `rank`。

转移：把当前可加工的工单 `j`（前驱均已在 `S` 中）接到 `k` 后，
`cost' = cost + [family[k] != family[j]]`。

**字典序 tie-break（不保存整段序列）**：同一层的不同状态代表不同序列，
候选序列形如“父序列 + j”，故等换型代价时比较 `(父序列 rank, j)`。
每层转移结束后，将全部状态按 `(父 rank, j)` 排序赋稠密 rank —— 注意 rank
表达的是**纯字典序**（不含 cost）；而为每个状态挑选父序列时使用的是
`(cost', 父 rank, j)`，两者是不同的序。最终在全集合层按
`(cost, rank, 末单)` 取最优并回溯。

键打包为一个 36 位整数（cost 6 位 + rank 24 位 + job 6 位），
避免对象分配；`n=18` 最坏（无边、family 全不同）约 10 秒级、内存约 100 MB，
存在约束边的实例通常快数个数量级（如 100 条边约 0.1 秒）。

### 环检测与取证

Kahn 剥离所有入度可归零的节点；剩余节点都处在某个环上。从任一剩余节点
沿“仍在剩余集合中的前驱”反向行走直到重复，重复段即一条真实有向环
（反向收集后翻转），再旋转到环中最小 id 起始。

## 测试

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

测试策略：

- **穷举对拍**：n ≤ 4 时枚举全部有向图（4 节点共 65536 种边子集）× 多种
  family 布局；随机 DAG（n ≤ 8）共数百例。每例枚举**所有**拓扑序，
  断言 DP 的换型数与字典序顺序与暴力最优完全一致，并核对每个换型位置；
- **环证据**：对返回的 `cycle` 断言每一条边（含闭合边）都在输入边集合中、
  无重复节点、首元素为环上最小 id，且 `order` 为空（无部分排程）；
- **API**：正常排程、CYCLE、确定性、以及全部 422 情形（自环、重复边、
  未知引用、额外字段、空/非 ASCII、越界与类型错误等）。
