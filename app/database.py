"""SQLite 账存：连接管理、建表与就地升级。

账表
----
batches          批登记（入账后固定，不提供修改/删除）
refund_orders    退单主表（active / revoked 两态）
refund_segments  退单号段（只存首尾，不逐张展开；撤销后行保留，供历史查询）
occupied_ranges  当前 active 退单的号段物化表（只存首尾），仅供 O(log n)
                 的“最大首号前驱”重叠核查；由触发器随 refund_orders /
                 refund_segments 在同一事务内维护，业务代码不直接写入
ledger_entries   账目明细：入账(+实收)、退票冲减(-)、撤销冲回(+)，带逐单余额

重叠核查的性能要点
------------------
同批 active 号段两两不相交，故对申请区间 [a,b]，只需取
``first_ticket <= b`` 中 ``first_ticket`` 最大的一条（前驱），再在应用层
判断其 ``last_ticket >= a``。该查询走覆盖索引的单次定位 + LIMIT 1，
工作量只与 B 树深度有关，不随同批已撤销号段（根本不在 occupied_ranges
中）或不相交 active 号段数线性增长。

注意：``last_ticket >= a`` 不能放进 SQL WHERE——当前驱不满足该条件时，
SQLite 会沿索引继续向前回溯全部不相交号段，重新退化成线性扫描。

并发
----
写事务一律 ``BEGIN IMMEDIATE``：同一时刻只允许一个写者，重叠校验、
occupied_ranges 维护（触发器）与账目写入在同一事务内原子提交，失败整体
回滚保留原账。配合 WAL 与 busy_timeout 支持并发读。
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from collections.abc import Iterator

import sqlite3

# 测试可通过环境变量指定临时库；默认挂载到 compose 的 data 卷。
DB_PATH = os.environ.get("REFUND_DB_PATH", "/data/refunds.db")

# 核心账表（结构与初版完全一致，不新增/拆分 refund_segments 的列，
# 以保证旧库文件与按原列序写入的脚本直接可用）。
CORE_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS batches (
    batch_no        TEXT PRIMARY KEY,
    first_ticket    INTEGER NOT NULL,
    last_ticket     INTEGER NOT NULL,
    ticket_count    INTEGER NOT NULL,
    amount_received INTEGER NOT NULL,           -- 实收金额，整数分
    base_share      INTEGER NOT NULL,           -- 单票基础金额
    extra_shares    INTEGER NOT NULL,           -- 多分 1 分的票数
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS refund_orders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_no    TEXT NOT NULL REFERENCES batches(batch_no),
    refund_no   TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('active', 'revoked')),
    amount      INTEGER NOT NULL,               -- 按原分摊金额计算的冲减额（分）
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    revoked_at  TEXT,
    UNIQUE(batch_no, refund_no)
);

CREATE TABLE IF NOT EXISTS refund_segments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_no    TEXT NOT NULL,                   -- 冗余批次号
    order_id    INTEGER NOT NULL REFERENCES refund_orders(id),
    first_ticket INTEGER NOT NULL,              -- 只存首尾，绝不逐张展开
    last_ticket  INTEGER NOT NULL
);

-- 历史号段查询按 order_id 取回本单全部号段；无此索引会全表扫描，
-- 工作量随同批累积的撤销/历史号段数线性增长。
CREATE INDEX IF NOT EXISTS idx_seg_order
    ON refund_segments(order_id, first_ticket);

-- 兼容保留的批次维度索引。
CREATE INDEX IF NOT EXISTS idx_seg_batch_start
    ON refund_segments(batch_no, first_ticket, last_ticket);
CREATE INDEX IF NOT EXISTS idx_seg_batch_end
    ON refund_segments(batch_no, last_ticket);

CREATE INDEX IF NOT EXISTS idx_orders_batch ON refund_orders(batch_no);

CREATE TABLE IF NOT EXISTS ledger_entries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_no     TEXT NOT NULL REFERENCES batches(batch_no),
    entry_type   TEXT NOT NULL CHECK (entry_type IN ('batch_posting', 'refund', 'reversal')),
    ref_no       TEXT NOT NULL,                 -- 批号 / 退单号
    signed_amount INTEGER NOT NULL,             -- 入账为正，冲减为负，冲回为正
    balance_after INTEGER NOT NULL,             -- 该批逐单余额（分）
    detail_json  TEXT NOT NULL,                 -- 号段等明细，供下游逐单核对
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_ledger_batch
    ON ledger_entries(batch_no, id);
"""

# active 占用表及其维护对象。每条都带 IF NOT EXISTS：既会拼进 init_db 的
# executescript 全量脚本，也可由 ensure_occupancy_schema 在单个连接上逐条
# execute（不触发隐式 COMMIT），从而安全用于已开启的写事务内。
OCCUPANCY_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS occupied_ranges (
        batch_no     TEXT NOT NULL,
        first_ticket INTEGER NOT NULL,
        last_ticket  INTEGER NOT NULL,
        order_id     INTEGER NOT NULL,
        refund_no    TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_occ_batch_first
        ON occupied_ranges(batch_no, first_ticket DESC, last_ticket, refund_no)
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_seg_occ_insert
    AFTER INSERT ON refund_segments
    BEGIN
        INSERT INTO occupied_ranges
            (batch_no, first_ticket, last_ticket, order_id, refund_no)
        SELECT NEW.batch_no, NEW.first_ticket, NEW.last_ticket,
               NEW.order_id, ro.refund_no
        FROM refund_orders AS ro
        WHERE ro.id = NEW.order_id
          AND ro.status = 'active';
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_seg_occ_delete
    AFTER DELETE ON refund_segments
    BEGIN
        DELETE FROM occupied_ranges
        WHERE order_id = OLD.order_id
          AND first_ticket = OLD.first_ticket
          AND last_ticket = OLD.last_ticket;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_order_occ_insert
    AFTER INSERT ON refund_orders
    BEGIN
        -- 兼容直接造数时“先插单时已有号段”的顺序：单插入时物化其号段。
        INSERT INTO occupied_ranges
            (batch_no, first_ticket, last_ticket, order_id, refund_no)
        SELECT NEW.batch_no, rs.first_ticket, rs.last_ticket,
               NEW.id, NEW.refund_no
        FROM refund_segments AS rs
        WHERE rs.order_id = NEW.id
          AND NEW.status = 'active';
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_order_occ_status
    AFTER UPDATE OF status ON refund_orders
    BEGIN
        -- active -> revoked：随状态翻转在同一事务内释放全部原号段。
        DELETE FROM occupied_ranges
        WHERE order_id = NEW.id
          AND NEW.status = 'revoked';

        -- 防御性的 revoked -> active（正常接口不会发生）：重新占用。
        INSERT INTO occupied_ranges
            (batch_no, first_ticket, last_ticket, order_id, refund_no)
        SELECT NEW.batch_no, rs.first_ticket, rs.last_ticket,
               NEW.id, NEW.refund_no
        FROM refund_segments AS rs
        WHERE rs.order_id = NEW.id
          AND NEW.status = 'active'
          AND NOT EXISTS (
              SELECT 1 FROM occupied_ranges AS o
              WHERE o.order_id = NEW.id
                AND o.first_ticket = rs.first_ticket
                AND o.last_ticket = rs.last_ticket
          );
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_order_occ_delete
    AFTER DELETE ON refund_orders
    BEGIN
        DELETE FROM occupied_ranges WHERE order_id = OLD.id;
    END
    """,
)

SCHEMA = CORE_SCHEMA + "\n" + ";\n".join(OCCUPANCY_STATEMENTS) + ";\n"

# 旧库升级：把现存 active 退单的号段幂等回填到 occupied_ranges。
# 已撤销退单不回填（不占号）；NOT EXISTS 保证重复执行不产生重复行。
_BACKFILL_OCCUPIED = """
INSERT INTO occupied_ranges
    (batch_no, first_ticket, last_ticket, order_id, refund_no)
SELECT rs.batch_no, rs.first_ticket, rs.last_ticket, ro.id, ro.refund_no
FROM refund_segments AS rs
JOIN refund_orders   AS ro ON ro.id = rs.order_id
WHERE ro.status = 'active'
  AND NOT EXISTS (
      SELECT 1 FROM occupied_ranges AS o
      WHERE o.order_id = ro.id
        AND o.first_ticket = rs.first_ticket
        AND o.last_ticket = rs.last_ticket
  );
"""


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(
        db_path or DB_PATH,
        timeout=30.0,
        isolation_level=None,  # 手工事务
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={30_000}")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def ensure_occupancy_schema(conn: sqlite3.Connection) -> None:
    """在给定连接上幂等补齐 occupied_ranges 相关对象并回填。

    逐条 ``execute`` 而非 ``executescript``，不会隐式提交，故可在调用方
    已开启的写事务内执行（用于重叠核查发现旧库尚未升级时的惰性补救）。
    """
    for stmt in OCCUPANCY_STATEMENTS:
        conn.execute(stmt)
    conn.execute(_BACKFILL_OCCUPIED)


def init_db(db_path: str | None = None) -> None:
    """幂等建库建表，并把旧版 SQLite 库就地升级到带占用表的结构。

    可重复调用：新表/索引/触发器均 IF NOT EXISTS，回填带 NOT EXISTS。
    原有 batches / refund_orders / refund_segments / ledger_entries
    数据原样保留。
    """
    path = db_path or DB_PATH
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    conn = _connect(path)
    try:
        conn.executescript(SCHEMA)
        conn.execute(_BACKFILL_OCCUPIED)
    finally:
        conn.close()


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """请求级连接（短连接），退出即关闭。"""
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def write_tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """立即型写事务：提交或整体回滚，失败保留原账与号段占用状态。"""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
