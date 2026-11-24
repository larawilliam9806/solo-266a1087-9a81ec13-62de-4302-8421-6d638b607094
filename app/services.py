"""业务服务层。

关键不变量
==========
1. 票号永不逐张展开：号段只存首尾；重叠判定走 active 占用表
   ``occupied_ranges`` 的“最大首号前驱”单次索引定位（见
   ``_find_active_overlap``），不把该批历史/已撤销退单读进内存逐条遍历，
   单段核查工作量为 O(log active 号段数)，不随同批已撤销或不相交号段数
   线性增长；号段金额走 ``allocation`` 的 O(1) 公式。
2. 号段状态与账目明细在同一 ``BEGIN IMMEDIATE`` 事务内提交，占用表由
   触发器随号段写入/状态翻转维护；任何一步失败整体回滚，原账与占用保留。
3. 重送幂等：同批号 + 同退单号 + 同号段直接回放原单，不产生新账目行。
4. 撤销按建单时固化的原金额冲回，释放原号段（status -> revoked，
   触发器同步删除 occupied_ranges 行；refund_segments 历史行保留）。
"""

from __future__ import annotations

import json
import sqlite3

from fastapi import HTTPException, status

from . import allocation
from .database import ensure_occupancy_schema, get_conn, write_tx
from .schemas import (
    BatchCreate,
    BatchLedgerOut,
    BatchOut,
    BatchRefundsOut,
    LedgerEntryOut,
    RefundCreate,
    RefundOut,
    ReversalCreate,
    ReversalOut,
    SegmentOut,
)


# ---------------------------------------------------------------- 批登记

def register_batch(payload: BatchCreate) -> BatchOut:
    if payload.first_ticket > payload.last_ticket:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "first_ticket 不能大于 last_ticket",
        )
    count = allocation.ticket_count(payload.first_ticket, payload.last_ticket)
    base, extra = allocation.split_amount(payload.amount_received, count)

    with get_conn() as conn:
        try:
            with write_tx(conn):
                conn.execute(
                    """INSERT INTO batches
                       (batch_no, first_ticket, last_ticket, ticket_count,
                        amount_received, base_share, extra_shares)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        payload.batch_no,
                        payload.first_ticket,
                        payload.last_ticket,
                        count,
                        payload.amount_received,
                        base,
                        extra,
                    ),
                )
                _append_ledger(
                    conn,
                    batch_no=payload.batch_no,
                    entry_type="batch_posting",
                    ref_no=payload.batch_no,
                    signed_amount=payload.amount_received,
                    detail={
                        "first_ticket": payload.first_ticket,
                        "last_ticket": payload.last_ticket,
                        "ticket_count": count,
                        "base_share": base,
                        "extra_shares": extra,
                    },
                )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(
                status.HTTP_409_CONFLICT, f"批号已存在: {payload.batch_no}"
            ) from exc
        row = conn.execute(
            "SELECT * FROM batches WHERE batch_no = ?", (payload.batch_no,)
        ).fetchone()
    return _batch_out(row)


# ---------------------------------------------------------------- 退票

def create_refund(payload: RefundCreate) -> RefundOut:
    raw = [(s.first_ticket, s.last_ticket) for s in payload.segments]
    normalized = sorted(raw)

    # 纯输入校验（不触库）：首尾顺序、单内号段重叠
    prev_last: int | None = None
    for first_no, last_no in normalized:
        if first_no > last_no:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"号段首号大于尾号: [{first_no}, {last_no}]，整单退回",
            )
        if prev_last is not None and first_no <= prev_last:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"号段在单内重叠: [{first_no}, {last_no}]，整单退回",
            )
        prev_last = last_no

    with get_conn() as conn:
        # 查批次、幂等判定、重叠校验、号段写入、记账全部在同一立即型
        # 写事务内完成：并发写者在 BEGIN IMMEDIATE 处串行，杜绝
        # “先查后写”的检查时序窗口（TOCTOU）。
        with write_tx(conn):
            batch = _get_batch(conn, payload.batch_no)

            # 1) 批内越界校验
            for first_no, last_no in normalized:
                if (
                    first_no < batch["first_ticket"]
                    or last_no > batch["last_ticket"]
                ):
                    raise HTTPException(
                        status.HTTP_400_BAD_REQUEST,
                        f"号段 [{first_no}, {last_no}] 越出批范围 "
                        f"[{batch['first_ticket']}, {batch['last_ticket']}]，整单退回",
                    )

            # 2) 幂等 / 退单号冲突
            existing = conn.execute(
                "SELECT * FROM refund_orders WHERE batch_no = ? AND refund_no = ?",
                (payload.batch_no, payload.refund_no),
            ).fetchone()
            if existing is not None:
                existing_segs = _order_segments(conn, existing["id"])
                if existing["status"] == "revoked":
                    raise HTTPException(
                        status.HTTP_409_CONFLICT,
                        f"退单号 {payload.refund_no} 已用于已撤销退单，请更换退单号",
                    )
                if [
                    (s["first_ticket"], s["last_ticket"]) for s in existing_segs
                ] != normalized:
                    raise HTTPException(
                        status.HTTP_409_CONFLICT,
                        f"退单号 {payload.refund_no} 已存在且号段不同",
                    )
                # 完全重送：回放原单，不重复记账
                return _refund_out(conn, existing, idempotent=True)

            # 3) 与未撤销退单的重叠：索引区间查询，不遍历历史退单
            for first_no, last_no in normalized:
                overlap = _find_active_overlap(
                    conn, payload.batch_no, first_no, last_no
                )
                if overlap is not None:
                    raise HTTPException(
                        status.HTTP_409_CONFLICT,
                        f"号段 [{first_no}, {last_no}] 与未撤销退单 "
                        f"{overlap['refund_no']} 的号段 "
                        f"[{overlap['first_ticket']}, {overlap['last_ticket']}] 重叠，整单退回",
                    )

            # 4) 金额按固定分摊规则逐号段闭式计算（与拆单/顺序无关）
            segment_amounts = [
                allocation.segment_sum(
                    first_no,
                    last_no,
                    batch["first_ticket"],
                    batch["base_share"],
                    batch["extra_shares"],
                )
                for first_no, last_no in normalized
            ]
            total_amount = sum(segment_amounts)

            # 5) 号段状态与账目明细同事务原子生效
            cur = conn.execute(
                """INSERT INTO refund_orders (batch_no, refund_no, status, amount)
                   VALUES (?, ?, 'active', ?)""",
                (payload.batch_no, payload.refund_no, total_amount),
            )
            order_id = cur.lastrowid
            conn.executemany(
                """INSERT INTO refund_segments
                   (batch_no, order_id, first_ticket, last_ticket)
                   VALUES (?, ?, ?, ?)""",
                [
                    (payload.batch_no, order_id, first_no, last_no)
                    for first_no, last_no in normalized
                ],
            )
            _append_ledger(
                conn,
                batch_no=payload.batch_no,
                entry_type="refund",
                ref_no=payload.refund_no,
                signed_amount=-total_amount,
                detail={
                    "segments": [list(pair) for pair in normalized],
                    "segment_amounts": segment_amounts,
                    "amount": total_amount,
                },
            )
            order = conn.execute(
                "SELECT * FROM refund_orders WHERE id = ?", (order_id,)
            ).fetchone()
        return _refund_out(conn, order, idempotent=False)


# ---------------------------------------------------------------- 撤销

def reverse_refund(payload: ReversalCreate) -> ReversalOut:
    with get_conn() as conn:
        # 状态判定与状态翻转、冲回记账在同一写事务内，并发撤销在
        # BEGIN IMMEDIATE 处串行，第二个撤销只能看到 revoked 并走幂等回放。
        with write_tx(conn):
            order = conn.execute(
                "SELECT * FROM refund_orders WHERE batch_no = ? AND refund_no = ?",
                (payload.batch_no, payload.refund_no),
            ).fetchone()
            if order is None:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND,
                    f"退单不存在: {payload.batch_no}/{payload.refund_no}",
                )

            if order["status"] == "revoked":
                # 撤销重送：回放结果，不再写账
                return ReversalOut(
                    batch_no=payload.batch_no,
                    refund_no=payload.refund_no,
                    status="revoked",
                    released_amount=order["amount"],
                    idempotent=True,
                    revoked_at=order["revoked_at"],
                )

            segs = _order_segments(conn, order["id"])
            conn.execute(
                """UPDATE refund_orders
                   SET status = 'revoked',
                       revoked_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                   WHERE id = ?""",
                (order["id"],),
            )
            # 按原金额冲回；原号段随状态释放（重叠查询只认 active）
            _append_ledger(
                conn,
                batch_no=payload.batch_no,
                entry_type="reversal",
                ref_no=payload.refund_no,
                signed_amount=order["amount"],
                detail={
                    "released_amount": order["amount"],
                    "segments": [
                        [s["first_ticket"], s["last_ticket"]] for s in segs
                    ],
                },
            )
            refreshed = conn.execute(
                "SELECT * FROM refund_orders WHERE id = ?", (order["id"],)
            ).fetchone()
        return ReversalOut(
            batch_no=payload.batch_no,
            refund_no=payload.refund_no,
            status="revoked",
            released_amount=refreshed["amount"],
            idempotent=False,
            revoked_at=refreshed["revoked_at"],
        )


# ---------------------------------------------------------------- 查询

def get_batch(batch_no: str) -> BatchOut:
    with get_conn() as conn:
        return _batch_out(_get_batch(conn, batch_no))


def list_batches() -> list[BatchOut]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM batches ORDER BY rowid"
        ).fetchall()
    return [_batch_out(r) for r in rows]


def list_refunds(batch_no: str) -> BatchRefundsOut:
    with get_conn() as conn:
        _get_batch(conn, batch_no)
        orders = conn.execute(
            "SELECT * FROM refund_orders WHERE batch_no = ? ORDER BY id",
            (batch_no,),
        ).fetchall()
        refunds = [_refund_out(conn, o, idempotent=False) for o in orders]
    return BatchRefundsOut(batch_no=batch_no, refunds=refunds)


def batch_ledger(batch_no: str) -> BatchLedgerOut:
    with get_conn() as conn:
        batch = _get_batch(conn, batch_no)
        rows = conn.execute(
            "SELECT * FROM ledger_entries WHERE batch_no = ? ORDER BY id",
            (batch_no,),
        ).fetchall()
        entries = [
            LedgerEntryOut(
                id=r["id"],
                entry_type=r["entry_type"],
                ref_no=r["ref_no"],
                signed_amount=r["signed_amount"],
                balance_after=r["balance_after"],
                detail=json.loads(r["detail_json"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]
    total_refund = -sum(
        e.signed_amount for e in entries if e.entry_type == "refund"
    )
    total_reversal = sum(
        e.signed_amount for e in entries if e.entry_type == "reversal"
    )
    return BatchLedgerOut(
        batch_no=batch_no,
        posted_amount=batch["amount_received"],
        total_refund=total_refund,
        total_reversal=total_reversal,
        balance=batch["amount_received"] - total_refund + total_reversal,
        entries=entries,
    )


# ---------------------------------------------------------------- 内部工具

def _get_batch(conn: sqlite3.Connection, batch_no: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM batches WHERE batch_no = ?", (batch_no,)
    ).fetchone()
    if row is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"批号不存在: {batch_no}"
        )
    return row


def _order_segments(conn: sqlite3.Connection, order_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT first_ticket, last_ticket FROM refund_segments "
        "WHERE order_id = ? ORDER BY first_ticket",
        (order_id,),
    ).fetchall()


def _find_active_overlap(
    conn: sqlite3.Connection, batch_no: str, first_no: int, last_no: int
) -> sqlite3.Row | None:
    """区间重叠核查：单次覆盖索引定位，工作量不随历史号段数线性增长。

    同批 active 号段两两不相交。对申请区间 ``[first_no, last_no]``，
    取 ``first_ticket <= last_no`` 中首号最大的一条 active 号段（前驱）：

    * 若该前驱的 ``last_ticket >= first_no``，二者重叠，返回它；
    * 否则它完全落在申请区间左侧；更早的号段尾号更小（不相交且有序），
      不可能重叠，直接返回 None。

    已撤销号段由触发器从 ``occupied_ranges`` 删除，故同批撤销单积得再多
    也不增加本查询的工作量；大量与本单不相交的 active 号段也只贡献
    B 树深度（O(log n)），不会被逐行扫描。

    关键：``last_ticket >= first_no`` 只能在取回首行后于应用层判断。
    若放进 SQL WHERE，当前驱不满足时 SQLite 会沿索引继续回溯全部不相交
    号段，重新退化成 O(n) 扫描。
    """
    sql = """
        SELECT refund_no, first_ticket, last_ticket
        FROM occupied_ranges
        WHERE batch_no = ?
          AND first_ticket <= ?
        ORDER BY first_ticket DESC
        LIMIT 1
        """
    params = (batch_no, last_no)
    try:
        predecessor = conn.execute(sql, params).fetchone()
    except sqlite3.OperationalError as exc:
        # 直接在“仅初版结构”的旧库连接上调用本函数、尚未经过 init_db
        # 升级时的惰性补救：补齐占用表并回填后重试一次。
        if "occupied_ranges" not in str(exc):
            raise
        ensure_occupancy_schema(conn)
        predecessor = conn.execute(sql, params).fetchone()

    if predecessor is not None and predecessor["last_ticket"] >= first_no:
        return predecessor
    return None


def _append_ledger(
    conn: sqlite3.Connection,
    *,
    batch_no: str,
    entry_type: str,
    ref_no: str,
    signed_amount: int,
    detail: dict,
) -> None:
    row = conn.execute(
        "SELECT balance_after FROM ledger_entries WHERE batch_no = ? "
        "ORDER BY id DESC LIMIT 1",
        (batch_no,),
    ).fetchone()
    prev_balance = row["balance_after"] if row is not None else 0
    conn.execute(
        """INSERT INTO ledger_entries
           (batch_no, entry_type, ref_no, signed_amount, balance_after, detail_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            batch_no,
            entry_type,
            ref_no,
            signed_amount,
            prev_balance + signed_amount,
            json.dumps(detail, ensure_ascii=False, separators=(",", ":")),
        ),
    )


def _batch_out(row: sqlite3.Row) -> BatchOut:
    return BatchOut(
        batch_no=row["batch_no"],
        first_ticket=row["first_ticket"],
        last_ticket=row["last_ticket"],
        ticket_count=row["ticket_count"],
        amount_received=row["amount_received"],
        base_share=row["base_share"],
        extra_shares=row["extra_shares"],
        created_at=row["created_at"],
    )


def _refund_out(
    conn: sqlite3.Connection, order: sqlite3.Row, *, idempotent: bool
) -> RefundOut:
    batch = _get_batch(conn, order["batch_no"])
    seg_rows = _order_segments(conn, order["id"])
    segments = [
        SegmentOut(
            first_ticket=r["first_ticket"],
            last_ticket=r["last_ticket"],
            amount=allocation.segment_sum(
                r["first_ticket"],
                r["last_ticket"],
                batch["first_ticket"],
                batch["base_share"],
                batch["extra_shares"],
            ),
        )
        for r in seg_rows
    ]
    return RefundOut(
        batch_no=order["batch_no"],
        refund_no=order["refund_no"],
        status=order["status"],
        amount=order["amount"],
        segments=segments,
        idempotent=idempotent,
        created_at=order["created_at"],
        revoked_at=order["revoked_at"],
    )
