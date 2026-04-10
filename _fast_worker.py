#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_fast_worker.py — fast_evaluator 的輕量多進程 worker。

用途：被 RunAnalysisTaskThread 以 subprocess 啟動，
      只包含 fast_evaluator + 必要 helpers，不載入主程式，
      避免 tg_bot / sys_db 等模組級初始化造成 crash。

呼叫方式（由主程式啟動，不要手動執行）：
    python _fast_worker.py --args-file=... --output-file=...
"""

import sys, os, json, sqlite3, argparse, warnings, math
import numpy as np
import pandas as pd

# ── 讓 stdout/stderr 支援 UTF-8 ──
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


# ══════════════════════════════════════════════════
# 僅從主程式 JSON 傳入的固定設定（取代 sys_config）
# ══════════════════════════════════════════════════
_cfg: dict = {}   # 由 main() 填入

def _get(attr, default=0):
    return _cfg.get(attr, default)

def _trunc2(value):
    try:
        return math.floor(float(value) * 100) / 100.0
    except (ValueError, TypeError):
        return value

def _np_corr(np_tp, t2idx, n_times, sym_a, sym_b, t_start, t_end):
    """numpy Pearson 相關係數（取代 pandas merge+corr，~10x faster）"""
    i0 = t2idx.get(t_start, 0)
    i1 = t2idx.get(t_end, n_times - 1) + 1
    a = np_tp.get(sym_a)
    b = np_tp.get(sym_b)
    if a is None or b is None: return 0.0
    a, b = a[i0:i1], b[i0:i1]
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3: return 0.0
    a, b = a[mask], b[mask]
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0: return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ══════════════════════════════════════════════════
# Helper: 停損跳格
# ══════════════════════════════════════════════════
def get_stop_loss_config(price):
    if   price <   10: return _get('below_50', 500),        0.01
    elif price <   50: return _get('below_50', 500),        0.05
    elif price <  100: return _get('price_gap_50_to_100',  1000), 0.1
    elif price <  500: return _get('price_gap_100_to_500', 2000), 0.5
    elif price < 1000: return _get('price_gap_500_to_1000',3000), 1
    else:              return _get('price_gap_above_1000', 5000), 5

import math
def round_to_tick(price: float, direction: str = 'up') -> float:
    """將價格對齊至台股合法報價檔位。direction: 'up'=進位, 'down'=捨去"""
    _ticks = [
        (0, 10, 0.01), (10, 50, 0.05), (50, 100, 0.1),
        (100, 500, 0.5), (500, 1000, 1.0), (1000, float('inf'), 5.0)
    ]
    for lower, upper, t_size in _ticks:
        if lower <= price < upper:
            if direction == 'down':
                return round(math.floor(price / t_size) * t_size, 2)
            else:
                return round(math.ceil(price / t_size) * t_size, 2)
    return round(price, 2)


# ══════════════════════════════════════════════════
# Helper: DTW Pearson（只在 dtw_thresh > 0 時呼叫）
# ══════════════════════════════════════════════════
def calculate_dtw_pearson(df_lead, df_follow, window_start, window_end):
    if isinstance(window_start, str):
        try:    window_start = pd.to_datetime(window_start, format="%H:%M:%S").time()
        except: window_start = pd.to_datetime(window_start).time()
    if isinstance(window_end, str):
        try:    window_end = pd.to_datetime(window_end, format="%H:%M:%S").time()
        except: window_end = pd.to_datetime(window_end).time()

    sub_l = df_lead[(df_lead['time'] >= window_start) & (df_lead['time'] <= window_end)].copy()
    sub_f = df_follow[(df_follow['time'] >= window_start) & (df_follow['time'] <= window_end)].copy()
    if len(sub_l) < 3 or len(sub_f) < 3:
        return 0.0
    merged = pd.merge(
        sub_l[['time', 'high', 'low', 'close']],
        sub_f[['time', 'high', 'low', 'close']],
        on='time', suffixes=('_l', '_f')
    )
    if len(merged) < 3:
        return 0.0
    tp_l = (merged['high_l'] + merged['low_l'] + merged['close_l']) / 3
    tp_f = (merged['high_f'] + merged['low_f'] + merged['close_f']) / 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        corr = tp_l.corr(tp_f)
    return 0.0 if pd.isna(corr) else corr


# ══════════════════════════════════════════════════
# fast_evaluator（與主程式版本完全一致）
# ══════════════════════════════════════════════════
def fast_evaluator(params, dates, cache, groups, dispo_list, trial=None):
    p_wait            = params['wait_mins']
    p_dtw             = params['dtw_thresh']
    p_lead            = params['leader_pull']
    p_foll            = params['follow_pull']
    p_vmult           = params['vol_mult']
    p_vabs            = params['vol_abs']
    p_wait_min_avg    = params['wait_min_avg_vol']
    p_wait_max_single = params['wait_max_single_vol']
    p_sl_cushion      = params.get('sl_cushion_pct', 0.0)
    p_hold_mins       = params.get('hold_mins', 240)
    p_cutoff_raw      = params.get('cutoff_mins', 270)
    if isinstance(p_cutoff_raw, str):
        h, m_t = map(int, p_cutoff_raw.split(':'))
        p_cutoff_mins = (h - 9) * 60 + m_t
    else:
        p_cutoff_mins = int(p_cutoff_raw)

    tp_sum, wins, losses, trades_count = 0, 0, 0, 0
    winning_days, days_traded = set(), set()
    wins_a, losses_a, trades_a, tp_sum_a = 0, 0, 0, 0

    _max_daily_stops = _get('max_daily_stops', 3)
    _risk_enabled = _get('risk_control_enabled', True)

    for step_idx, t_date in enumerate(dates):
        day_data = cache.get(t_date, {})
        if not day_data:
            continue
        day_pnl = 0
        _daily_stops_b = 0

        for grp, syms in groups.items():
            dispo_today = dispo_list.get(t_date, []) if isinstance(dispo_list, dict) else dispo_list
            valid_syms  = [s for s in syms if s not in dispo_today and s in day_data]
            if len(valid_syms) < 2:
                continue

            stock_dfs  = {s: day_data[s] for s in valid_syms}
            first3_vol = {s: sum(r['volume'] for r in recs[:3]) / 3 if len(recs) >= 3 else 0
                          for s, recs in stock_dfs.items()}

            _np_tp, _t2idx, _n_times = {}, {}, 0
            if p_dtw > 0:
                _all_times = sorted({r['time'] for s in valid_syms for r in stock_dfs[s]})
                _t2idx = {t: i for i, t in enumerate(_all_times)}
                _n_times = len(_all_times)
                for sym in valid_syms:
                    arr = np.full(_n_times, np.nan)
                    for r in stock_dfs[sym]:
                        idx = _t2idx.get(r['time'])
                        if idx is not None:
                            arr[idx] = (r.get('high', 0) + r.get('low', 0) + r.get('close', 0)) / 3
                    _np_tp[sym] = arr

            leader, tracking = None, set()
            in_wait, wait_cnt, start_t, leader_peak_rise = False, 0, None, None
            leader_rise_before_decline = None   # 天花板：反轉時記錄，突破時解除等待（對齊 process_group_data）
            reentry_count = 0                   # 停損再進場計數器
            pull_up, limit_up = False, False
            is_busy, exit_at  = False, -1

            num_bars = min(len(recs) for recs in stock_dfs.values())
            if num_bars == 0:
                continue

            _cum_vol  = {s: 0       for s in valid_syms}
            _run_high = {s: -999.0  for s in valid_syms}
            _run_low  = {s: 999999.0 for s in valid_syms}

            for m in range(num_bars):
                for _s in valid_syms:
                    if m < len(stock_dfs[_s]):
                        _bar = stock_dfs[_s][m]
                        _cum_vol[_s]   += _bar.get('volume', 0)
                        _run_high[_s]   = max(_run_high[_s], _bar.get('high', -999))
                        _run_low[_s]    = min(_run_low[_s],  _bar.get('low', _bar.get('close', 999999)))

                if is_busy:
                    if m >= exit_at: is_busy = False
                    continue

                if m >= p_cutoff_mins:
                    is_busy, exit_at = True, 999
                    continue

                trigger_list = []
                for sym in valid_syms:
                    row, avgv = stock_dfs[sym][m], first3_vol[sym]
                    if round(row['high'], 2) >= round(row['漲停價'], 2):
                        if m == 0 or round(stock_dfs[sym][m - 1]['high'], 2) < round(row['漲停價'], 2):
                            trigger_list.append((sym, 'limit'))
                    elif (row['pct_increase'] >= p_lead and
                          (row['volume'] >= p_vabs or (avgv > 0 and row['volume'] >= p_vmult * avgv))):
                        trigger_list.append((sym, 'pull'))

                for sym, cond in trigger_list:
                    if cond == 'limit':
                        tracking.add(sym); leader, in_wait, wait_cnt = sym, True, 0
                        if not (pull_up or limit_up): start_t = row['time']
                        pull_up, limit_up = False, True
                    else:
                        if not pull_up and not limit_up:
                            pull_up, limit_up, start_t = True, False, row['time']
                            tracking.clear()
                        tracking.add(sym)

                if pull_up or limit_up:
                    for sym in valid_syms:
                        if sym not in tracking and stock_dfs[sym][m]['pct_increase'] >= p_foll:
                            tracking.add(sym)

                if tracking:
                    max_sym, max_r = None, -999
                    for sym in tracking:
                        if stock_dfs[sym][m]['rise'] > max_r:
                            max_r, max_sym = stock_dfs[sym][m]['rise'], sym
                    if leader is None:
                        leader, leader_peak_rise = max_sym, max_r
                    elif max_sym == leader:
                        if max_r is not None and (leader_peak_rise is None or max_r > leader_peak_rise):
                            leader_peak_rise = max_r
                    elif max_r is not None and leader_peak_rise is not None and max_r > leader_peak_rise:
                        leader, start_t, in_wait, wait_cnt, leader_peak_rise = \
                            max_sym, stock_dfs[max_sym][m]['time'], False, 0, max_r
                    if leader and stock_dfs[leader][m]['high'] <= stock_dfs[leader][max(0, m - 1)]['high'] and not in_wait:
                        in_wait, wait_cnt = True, 0
                        leader_rise_before_decline = stock_dfs[leader][m].get('highest', stock_dfs[leader][m]['high'])

                if in_wait:
                    wait_cnt += 1
                    if p_dtw > 0 and wait_cnt >= p_wait - 1 and leader and len(tracking) > 1 and m >= 10:
                        curr_t_dtw = stock_dfs[leader][m]['time']
                        win_start  = stock_dfs[leader][0]['time']
                        to_rm = [s for s in tracking if s != leader and
                                 _np_corr(_np_tp, _t2idx, _n_times, leader, s, win_start, curr_t_dtw) < p_dtw]
                        for s in to_rm: tracking.discard(s)

                    if wait_cnt >= p_wait:
                        if _risk_enabled and _daily_stops_b >= _max_daily_stops:
                            break
                        _did_reentry = False
                        eligible = []
                        for sym in tracking:
                            if sym == leader: continue
                            df_wait   = [r for r in stock_dfs[sym][:m + 1] if r['time'] >= start_t]
                            if not df_wait: continue
                            df_wait_v = [r['volume'] for r in df_wait]
                            # OR 邏輯：v>=倍數均量 OR v>=絕對量 任一達標即放行（對齊 process_group_data）
                            if not any((first3_vol[sym] > 0 and v >= p_vmult * first3_vol[sym]) or v >= p_vabs
                                       for v in df_wait_v): continue
                            if (sum(df_wait_v) / len(df_wait_v) < p_wait_min_avg and
                                    max(df_wait_v) < p_wait_max_single): continue
                            r_now, p_now = stock_dfs[sym][m]['rise'], stock_dfs[sym][m]['close']
                            rlb = _get('rise_lower_bound', -10)
                            rub = _get('rise_upper_bound', 9.6)
                            if not (rlb <= r_now <= rub): continue
                            ldr_rise_now = stock_dfs[leader][m]['rise'] if leader in stock_dfs and m < len(stock_dfs[leader]) else None
                            _p_lag = params.get('min_lag_pct', _get('min_lag_pct', 0))
                            if ldr_rise_now is not None and (ldr_rise_now - r_now) < _p_lag: continue
                            prev_close = stock_dfs[sym][0].get('昨日收盤價', 0) if stock_dfs[sym] else 0
                            hi_today   = stock_dfs[sym][m].get('highest', p_now)
                            _p_height = params.get('min_height_pct', _get('min_height_pct', 0))
                            if prev_close and prev_close > 0 and (hi_today - prev_close) / prev_close * 100 < _p_height: continue
                            # 不過高條件
                            _req_nbh = params.get('require_not_broken_high', _get('require_not_broken_high', False))
                            if _req_nbh:
                                _c_now = stock_dfs[sym][m].get('close', 0)
                                _h_now = stock_dfs[sym][m].get('highest', _c_now)
                                if _c_now >= _h_now and _h_now > 0: continue
                            _min_elig = params.get('min_eligible_avg_vol', _get('min_eligible_avg_vol', 0))
                            if _min_elig > 0 and _cum_vol.get(sym, 0) / (m + 1) < _min_elig: continue
                            _vmin_range = params.get('volatility_min_range', _get('volatility_min_range', 0))
                            if _vmin_range > 0 and m >= 10 and prev_close and prev_close > 0:
                                if (_run_high[sym] - _run_low[sym]) / prev_close * 100 < _vmin_range: continue
                            _ptol = params.get('pullback_tolerance', _get('pullback_tolerance', 999))
                            if len(df_wait) >= 2 and df_wait[-1].get('rise', 0) > max(r.get('rise', 0) for r in df_wait[:-1]) + _ptol: continue
                            # max_entry_price: 收盤價必須 < capital_per_stock * 15（對齊 process_group_data）
                            _cap = _get('capital_per_stock', 15)
                            if p_now >= _cap * 15: continue
                            # min_close_price: 收盤價必須 >= min_close_price（對齊 process_group_data）
                            _min_cp = _get('min_close_price', 0)
                            if _min_cp > 0 and p_now < _min_cp: continue
                            eligible.append({'sym': sym, 'rise': r_now, 'p_ent': p_now, 'hi': hi_today, 'total_vol': _cum_vol.get(sym, 0)})

                        # allow_leader_entry: 領漲股也加入候選（對齊 process_group_data 85克策略）
                        if _get('allow_leader_entry', True) and leader and leader in stock_dfs and m < len(stock_dfs[leader]) and leader not in [e['sym'] for e in eligible]:
                            _ldr_row = stock_dfs[leader][m]
                            _ldr_close = _ldr_row['close']
                            _ldr_rise = _ldr_row.get('rise', 0)
                            _cap_le = _get('capital_per_stock', 15)
                            _min_cp_le = _get('min_close_price', 0)
                            rlb_le = _get('rise_lower_bound', -10)
                            rub_le = _get('rise_upper_bound', 9.6)
                            if (rlb_le <= _ldr_rise <= rub_le and
                                    _ldr_close < _cap_le * 15 and
                                    (_min_cp_le <= 0 or _ldr_close >= _min_cp_le)):
                                _ldr_hi = _ldr_row.get('highest', _ldr_close)
                                eligible.append({'sym': leader, 'rise': _ldr_rise, 'p_ent': _ldr_close, 'hi': _ldr_hi, 'total_vol': _cum_vol.get(leader, 0)})

                        if eligible:
                            _p_sort = params.get('stock_sort_mode', 'volume')
                            if _p_sort == 'volume':
                                eligible.sort(key=lambda x: -x.get('total_vol', 0))
                            else:
                                eligible.sort(key=lambda x: x['rise'])

                            _fee_r = _get('transaction_fee', 0.1425) * 0.01 * _get('transaction_discount', 18.0) * 0.01
                            for item_a in eligible:
                                p_a    = item_a['p_ent']
                                shrs_a = round((_get('capital_per_stock', 15) * 10000) / (p_a * 1000))
                                sell_a = shrs_a * p_a * 1000
                                fee_a  = int(sell_a * _fee_r)
                                tax_a  = int(sell_a * _get('trading_tax', 0.15) * 0.01)
                                gap_a, tick_a = get_stop_loss_config(p_a)
                                hi_a   = item_a['hi'] or p_a
                                base_a = hi_a + tick_a if (hi_a - p_a) * 1000 >= gap_a else p_a + gap_a / 1000
                                stop_a = round_to_tick(round(base_a + p_a * (p_sl_cushion / 100.0), 2), 'up')
                                _lup_a = stock_dfs[item_a['sym']][m].get('漲停價')
                                if _lup_a:
                                    _tlup = 0.01 if _lup_a < 10 else 0.05 if _lup_a < 50 else 0.1 if _lup_a < 100 else 0.5 if _lup_a < 500 else 1 if _lup_a < 1000 else 5
                                    if stop_a > _lup_a - 2 * _tlup: stop_a = _lup_a - 2 * _tlup
                                max_risk_a = (stop_a - p_a) * 1000
                                if max_risk_a > gap_a: continue
                                max_h_a = num_bars - 1 if (m + p_hold_mins) >= 270 else (m + p_hold_mins)
                                for me_a in range(m + 1, num_bars):
                                    r_a = stock_dfs[item_a['sym']][me_a]
                                    if _trunc2(r_a['high']) >= _trunc2(stop_a) or me_a >= max_h_a:
                                        pe_a  = stop_a if r_a['high'] >= stop_a else r_a['close']
                                        bt_a  = shrs_a * pe_a * 1000
                                        pnl_a = sell_a - bt_a - fee_a - int(bt_a * _fee_r) - tax_a
                                        trades_a += 1; tp_sum_a += pnl_a
                                        if pnl_a > 0: wins_a += 1
                                        else:         losses_a += 1
                                        break

                            if p_dtw <= 0:
                                eligible_dtw = list(eligible)
                            else:
                                curr_t = stock_dfs[leader][m]['time']
                                eligible_dtw = [
                                    item for item in eligible
                                    if _np_corr(_np_tp, _t2idx, _n_times, leader, item['sym'],
                                        start_t, curr_t) >= p_dtw
                                ]

                            mode_b_exit_m    = m + 1
                            _mode_b_stop_loss = False
                            if eligible_dtw:
                                target     = eligible_dtw[0]
                                p_ent      = target['p_ent']
                                shrs       = round((_get('capital_per_stock', 15) * 10000) / (p_ent * 1000))
                                sell_total = shrs * p_ent * 1000
                                _fee_r_b   = _fee_r
                                ent_fee    = int(sell_total * _fee_r_b)
                                tax        = int(sell_total * _get('trading_tax', 0.15) * 0.01)
                                gap, tick  = get_stop_loss_config(p_ent)
                                hi_on_e    = target['hi'] or p_ent
                                base_stop  = hi_on_e + tick if (hi_on_e - p_ent) * 1000 >= gap else p_ent + gap / 1000
                                stop_p     = round_to_tick(round(base_stop + p_ent * (p_sl_cushion / 100.0), 2), 'up')
                                _lup_p = stock_dfs[target['sym']][m].get('漲停價')
                                if _lup_p:
                                    _tlup = 0.01 if _lup_p < 10 else 0.05 if _lup_p < 50 else 0.1 if _lup_p < 100 else 0.5 if _lup_p < 500 else 1 if _lup_p < 1000 else 5
                                    if stop_p > _lup_p - 2 * _tlup: stop_p = _lup_p - 2 * _tlup
                                max_risk_p = (stop_p - p_ent) * 1000
                                if max_risk_p <= gap:
                                    m_end      = num_bars - 1
                                    max_hold_m = m_end if (m + p_hold_mins) >= 270 else (m + p_hold_mins)
                                    for m_exit in range(m + 1, m_end + 1):
                                        r_ex = stock_dfs[target['sym']][m_exit]
                                        if _trunc2(r_ex['high']) >= _trunc2(stop_p) or m_exit >= max_hold_m:
                                            _mode_b_stop_loss = (_trunc2(r_ex['high']) >= _trunc2(stop_p))
                                            p_exit    = stop_p if _mode_b_stop_loss else r_ex['close']
                                            buy_total = shrs * p_exit * 1000
                                            profit    = sell_total - buy_total - ent_fee - int(buy_total * _fee_r_b) - tax
                                            tp_sum += profit; day_pnl += profit; trades_count += 1
                                            if profit > 0: wins += 1
                                            else:          losses += 1
                                            mode_b_exit_m = m_exit
                                            break

                            if _mode_b_stop_loss:
                                _daily_stops_b += 1

                            if eligible_dtw and _mode_b_stop_loss:
                                _allow_re = _get('allow_reentry', False)
                                _max_re   = _get('max_reentry_times', 1)
                                _lb_bars  = _get('reentry_lookback_candles', 3)
                                if not (_allow_re and reentry_count < _max_re):
                                    break  # 預設（allow_reentry=False）：停損終止，與原邏輯完全相同
                                # Re-entry 模式（allow_reentry=True 且有剩餘次數）
                                lb_s   = max(0, mode_b_exit_m - _lb_bars)
                                _found = False
                                for r_sym in valid_syms:
                                    for lb_m in range(lb_s, min(mode_b_exit_m + 1, len(stock_dfs[r_sym]))):
                                        lb_bar  = stock_dfs[r_sym][lb_m]
                                        _is_lu  = (round(lb_bar['high'], 2) >= round(lb_bar['漲停價'], 2) and
                                                   (lb_m == 0 or round(stock_dfs[r_sym][lb_m - 1]['high'], 2) < round(lb_bar['漲停價'], 2)))
                                        _lb_vol  = lb_bar.get('volume', 0)
                                        _lb_avgv = first3_vol.get(r_sym, 0)
                                        _is_pu  = (lb_bar.get('pct_increase', 0) >= p_lead and
                                                   (_lb_vol >= p_vabs or (_lb_avgv > 0 and _lb_vol >= p_vmult * _lb_avgv)))
                                        if _is_lu or _is_pu:
                                            reentry_count  += 1
                                            leader, tracking   = r_sym, {r_sym}
                                            in_wait, wait_cnt  = True, 0
                                            pull_up, limit_up  = not _is_lu, _is_lu
                                            start_t            = lb_bar['time']
                                            leader_rise_before_decline = lb_bar.get('highest', lb_bar.get('high', 0))
                                            _found = True; break
                                    if _found: break
                                if not _found:
                                    reentry_count += 1
                                    leader, tracking  = None, set()
                                    pull_up = limit_up = in_wait = False
                                    wait_cnt = 0
                                _did_reentry = True

                            if not _did_reentry:
                                is_busy, exit_at = True, mode_b_exit_m

                        if not _did_reentry:
                            pull_up = limit_up = False
                            leader, tracking, in_wait, wait_cnt = None, set(), False, 0
                    elif leader and leader_rise_before_decline is not None and \
                            stock_dfs[leader][m]['high'] > leader_rise_before_decline:
                        # 突破前高：漲勢延續，中斷等待（對齊 process_group_data line 3554-3558）
                        leader_rise_before_decline = stock_dfs[leader][m].get('highest', stock_dfs[leader][m]['high'])
                        in_wait, wait_cnt = False, 0

        if day_pnl != 0:
            days_traded.add(t_date)
            if day_pnl > 0: winning_days.add(t_date)

        if trial is not None:
            trial.report(tp_sum, step_idx)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

    pf            = (wins / losses) if losses > 0 else (99.9 if wins > 0 else 0)
    win_rate_a    = (wins_a / trades_a * 100) if trades_a > 0 else 0
    daily_wr      = (len(winning_days) / len(days_traded) * 100) if days_traded else 0
    expectancy    = (tp_sum / trades_count) if trades_count > 0 else 0

    _min_t = max(10, len(dates) // 2)
    if trades_count < 3:
        ai_score = 0
    elif tp_sum <= 0:
        ai_score = tp_sum
    else:
        ratio = min(1.0, trades_count / _min_t)
        ai_score = tp_sum * (ratio ** 1.5)

    return {
        **params,
        'Total_PnL':     tp_sum,
        'NoFilter_PnL':  tp_sum_a,
        'WinRate':       win_rate_a,
        'Daily_WinRate': daily_wr,
        'PF': pf, 'Expectancy': expectancy, 'Count': trades_count,
        'ai_score':      ai_score,
    }


# ══════════════════════════════════════════════════
# Worker 主程式
# ══════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--args-file',   required=True)
    ap.add_argument('--output-file', required=True)
    ns = ap.parse_args()

    with open(ns.args_file, encoding='utf-8') as f:
        cfg_in = json.load(f)

    # 填入固定設定
    global _cfg
    _cfg = cfg_in.get('sys_config_snapshot', {})

    db_path      = cfg_in['db_path']
    search_space = cfg_in['search_space']
    n_trials     = cfg_in['n_trials']
    unique_dates = cfg_in['unique_dates']
    sample_ratio = cfg_in.get('sample_ratio', 0.70)
    groups       = cfg_in['groups']
    dispo        = cfg_in.get('dispo', {})
    seed         = cfg_in.get('seed', None)

    # 從 DB 載入 list-of-dicts 快取
    import random
    conn   = sqlite3.connect(db_path)
    raw_df = pd.read_sql('SELECT * FROM intraday_kline', conn)
    conn.close()
    for c in ['high', '漲停價', 'pct_increase', 'volume', 'close', 'open', 'low', '昨日收盤價', 'rise', 'highest']:
        if c in raw_df.columns:
            raw_df[c] = pd.to_numeric(raw_df[c], errors='coerce')
    lod: dict = {}
    for (_d, _s), _g in raw_df.groupby(['date', 'symbol']):
        lod.setdefault(_d, {})[_s] = _g.sort_values('time').to_dict('records')

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rnd = random.Random(seed)
    min_sample = max(5, len(unique_dates) // 3)

    def objective(trial):
        params = {}
        for k, v in search_space.items():
            if len(v) == 1:
                params[k] = v[0]
            elif isinstance(v[0], str):
                params[k] = trial.suggest_categorical(k, v)
            else:
                is_int = all(isinstance(x, int) for x in v)
                lo, hi = min(v), max(v)
                if is_int:
                    step = v[1] - v[0] if len(v) > 1 else 1
                    params[k] = trial.suggest_int(k, lo, hi, step=step)
                else:
                    params[k] = round(trial.suggest_float(k, lo, hi), 2)

        k = max(min_sample, int(len(unique_dates) * sample_ratio))
        sampled = sorted(rnd.sample(unique_dates, k))
        res = fast_evaluator(params, sampled, lod, groups, dispo, trial=trial)
        for key, val in res.items():
            trial.set_user_attr(key, val)
        return res['ai_score'] * (len(unique_dates) / len(sampled))

    # 進度回報檔（主線程輪詢用）
    progress_file = cfg_in.get('progress_file')

    def _report_progress(study, trial):
        if not progress_file: return
        done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        try:
            with open(progress_file, 'w') as f:
                json.dump({'done': done, 'pruned': pruned}, f)
        except: pass

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1, callbacks=[_report_progress])

    results = [t.user_attrs for t in study.trials
               if t.state == optuna.trial.TrialState.COMPLETE]

    with open(ns.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, default=str)

    print(f"[worker] 完成 {len(results)} trials → {ns.output_file}", flush=True)


if __name__ == '__main__':
    main()
