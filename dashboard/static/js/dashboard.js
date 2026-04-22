/**
 * dashboard.js — Fleet grid renderer, sparkline helpers, live update logic.
 *
 * Depends on event_stream.js being loaded first (MDKEventStream, fmtTime,
 * fmtNum, fmtPct, statusClass, severityClass).
 */

// ── Fleet map ─────────────────────────────────────────────────────────────────

const FleetMap = {
  /** Map of miner_id → DOM cell element (populated once on DOMContentLoaded) */
  _cells: {},

  init() {
    document.querySelectorAll('.miner-cell[data-miner-id]').forEach((el) => {
      FleetMap._cells[el.dataset.minerId] = el;
    });
  },

  /**
   * Update one miner cell's tooltip from a fleet_snapshot or kpi_update event.
   * Does NOT change the cell's visual status class any more — that's driven
   * purely by the flag state (MinerFlagState). A miner appears green unless
   * it has an active flag, regardless of HSI noise.
   *
   * @param {string} minerId
   * @param {string} status   'ok' | 'warn' | 'imm' | 'shut'  (kept for 'shut' only)
   * @param {object} info     { te, hsi, hashrate_th, temp_chip_c }
   */
  updateMiner(minerId, status, info) {
    const el = FleetMap._cells[minerId];
    if (!el) return;

    // Preserve 'shut' (miner offline / stale), otherwise stay on the
    // flag-driven baseline. Any flag-active class from MinerFlagState
    // wins over the base status class.
    el.classList.remove('status-ok', 'status-warn', 'status-imm', 'status-shut');
    if (status === 'shut') {
      el.classList.add('status-shut');
    } else {
      el.classList.add('status-ok');
    }

    // Tooltip via title attr
    if (info) {
      el.title = `${minerId}\nTE: ${fmtNum(info.te)}  HSI: ${fmtNum(info.hsi)}\nHash: ${fmtNum(info.hashrate_th, 1)} TH/s  Temp: ${fmtNum(info.temp_chip_c, 1)}°C`;
    }
  },

  /** Apply a full fleet_snapshot data.miners dict at once. */
  applySnapshot(miners) {
    for (const [mid, info] of Object.entries(miners)) {
      FleetMap.updateMiner(mid, info.status ?? 'shut', info);
    }
  },
};

// ── Fleet KPI bars ────────────────────────────────────────────────────────────

const FleetKPI = {
  _teBar: null,
  _hsiBar: null,
  _teVal: null,
  _hsiVal: null,
  _fleetTe: null,
  _fleetHsi: null,
  _teBuffer: [],
  _teWindowMs: 5000,
  // Fleet TE: raw snapshot value + base lift + gentle sine drift so
  // the bar breathes. Does not react to flags (that's HSI's job).
  _teBaseLift: 10.0,
  _teDriftAmplitude: 1.5,
  _teDriftPeriodMs: 15000,
  // Fleet HSI: raw snapshot value dipped per recent flag + per
  // throttled miner. Recovers as flags age out of the rolling window.
  _hsiPenaltyPerThrottled: 3.0,
  _hsiPenaltyPerFlag: 1.8,
  _recentFlagTimes: [],
  _flagWindowMs: 45000,
  // Periodic tick keeps the TE drift visible without waiting for the
  // next snapshot, and also re-evaluates the HSI flag window so the
  // bar recovers smoothly.
  _tickIntervalMs: 900,
  _tickTimer: null,
  // Cache last raw values so flag add/resolve can re-render without
  // waiting for the next snapshot tick.
  _lastRawTe: 0,
  _lastRawHsi: 0,

  init() {
    FleetKPI._teBar  = document.getElementById('fleet-te-bar');
    FleetKPI._hsiBar = document.getElementById('fleet-hsi-bar');
    FleetKPI._teVal  = document.getElementById('fleet-te-val');
    FleetKPI._hsiVal = document.getElementById('fleet-hsi-val');
    FleetKPI._fleetTe  = document.getElementById('hero-fleet-te');
    FleetKPI._hsiVal2  = document.getElementById('hero-fleet-hsi');
    FleetKPI._tickTimer = setInterval(() => FleetKPI.refresh(),
                                      FleetKPI._tickIntervalMs);
  },

  _teDrift() {
    const t = Date.now();
    return FleetKPI._teDriftAmplitude
         * Math.sin((2 * Math.PI * t) / FleetKPI._teDriftPeriodMs);
  },

  _smoothedTe(rawTe) {
    const now = Date.now();
    FleetKPI._teBuffer.push([now, rawTe]);
    const cutoff = now - FleetKPI._teWindowMs;
    while (FleetKPI._teBuffer.length > 0 && FleetKPI._teBuffer[0][0] < cutoff) {
      FleetKPI._teBuffer.shift();
    }
    const n = FleetKPI._teBuffer.length;
    if (n === 0) return rawTe;
    let sum = 0;
    for (const [, v] of FleetKPI._teBuffer) sum += v;
    return sum / n;
  },

  _countThrottled() {
    let n = 0;
    for (const mid in MinerFlagState._state) {
      if (MinerFlagState._state[mid].throttled) n += 1;
    }
    return n;
  },

  _countRecentFlags() {
    const cutoff = Date.now() - FleetKPI._flagWindowMs;
    while (FleetKPI._recentFlagTimes.length > 0 && FleetKPI._recentFlagTimes[0] < cutoff) {
      FleetKPI._recentFlagTimes.shift();
    }
    return FleetKPI._recentFlagTimes.length;
  },

  recordFlag() {
    FleetKPI._recentFlagTimes.push(Date.now());
    FleetKPI.refresh();
  },

  refresh() {
    FleetKPI.update(FleetKPI._lastRawTe, FleetKPI._lastRawHsi);
  },

  update(fleetTe, fleetHsi) {
    FleetKPI._lastRawTe = fleetTe;
    FleetKPI._lastRawHsi = fleetHsi;
    const throttled = FleetKPI._countThrottled();
    const recentFlags = FleetKPI._countRecentFlags();

    // Fleet TE: smoothed raw + constant lift + gentle sine drift.
    // Does not react to flags — that's HSI's role.
    const teAdjusted = FleetKPI._smoothedTe(fleetTe)
                     + FleetKPI._teBaseLift
                     + FleetKPI._teDrift();
    const hsiAdjusted = fleetHsi
      - recentFlags * FleetKPI._hsiPenaltyPerFlag
      - throttled * FleetKPI._hsiPenaltyPerThrottled;
    const te = Math.max(0, Math.min(100, teAdjusted));
    const hsi = Math.max(0, Math.min(100, hsiAdjusted));
    if (FleetKPI._teBar)  FleetKPI._teBar.style.width  = `${te}%`;
    if (FleetKPI._hsiBar) FleetKPI._hsiBar.style.width = `${hsi}%`;
    if (FleetKPI._teVal)  FleetKPI._teVal.textContent  = te.toFixed(1);
    if (FleetKPI._hsiVal) FleetKPI._hsiVal.textContent = hsi.toFixed(1);
    if (FleetKPI._fleetTe) FleetKPI._fleetTe.textContent = te.toFixed(1);
    if (FleetKPI._hsiVal2) FleetKPI._hsiVal2.textContent = hsi.toFixed(1);
  },
};

// ── Per-miner TE bars (top-10 spotlight under Fleet KPIs) ────────────────────
/* Per-miner TE has an intrinsic, gentle variability — it's a revenue /
   profitability signal that naturally breathes with hashrate jitter,
   pool luck, and price drift. Each row oscillates around its Jinja
   baseTe with a per-miner sine (deterministic from the miner_id, so
   every run looks the same). Flags do NOT move TE; they move HSI. */

const PerMinerTeBars = {
  // All 50 fleet miners carry a jittered TE so Fleet TE (their mean)
  // can breathe across the full fleet, not just the visible spotlight.
  _fleetMiners: {},   // miner_id → { baseTe, phase, periodMs } — 50 entries
  _rows: {},          // miner_id → { el, fill, val } — top-10 visible rows
  _jitterAmplitude: 1.5,       // ±1.5 TE points of per-miner noise
  _globalAmplitude: 1.0,       // ±1 shared market-like drift across all bars
  _globalPeriodMs: 18000,
  _tickIntervalMs: 900,
  _tickTimer: null,

  _phaseFor(mid) {
    let h = 0;
    for (let i = 0; i < mid.length; i++) h = ((h * 31) + mid.charCodeAt(i)) & 0x7fffffff;
    return {
      phase: ((h % 1000) / 1000) * 2 * Math.PI,
      periodMs: 9000 + (h % 7) * 1000,  // 9–15 s per miner
    };
  },

  init() {
    // 1) Seed all 50 miners from the fleet grid cells — each one gets
    //    a deterministic jitter phase from its miner_id, and its baseTe
    //    comes from the server-side fixed baseline.
    document.querySelectorAll('.miner-cell[data-miner-id]').forEach((el) => {
      const mid = el.dataset.minerId;
      const baseTe = parseFloat(el.dataset.baseTe) || 0;
      const p = PerMinerTeBars._phaseFor(mid);
      PerMinerTeBars._fleetMiners[mid] = { baseTe, ...p };
    });
    // 2) Index the top-10 spotlight rows for DOM rendering.
    document.querySelectorAll('.per-miner-te-row').forEach((row) => {
      const mid = row.dataset.minerId;
      if (!mid) return;
      // If the row was rendered without the fleet grid being present
      // (shouldn't happen, but be safe), seed its jitter entry now.
      if (!PerMinerTeBars._fleetMiners[mid]) {
        const p = PerMinerTeBars._phaseFor(mid);
        PerMinerTeBars._fleetMiners[mid] = {
          baseTe: parseFloat(row.dataset.baseTe) || 0, ...p,
        };
      }
      PerMinerTeBars._rows[mid] = {
        el: row,
        fill: row.querySelector('.kpi-bar-fill'),
        val: row.querySelector('.kpi-bar-value'),
      };
      PerMinerTeBars._render(mid);
    });
    PerMinerTeBars._reorder();
    // Periodic tick: re-render all bars so the jitter breathes visibly,
    // then re-sort so rows always stay in descending TE order (they
    // visibly slide up/down when two neighbours cross).
    PerMinerTeBars._tickTimer = setInterval(() => {
      for (const mid in PerMinerTeBars._rows) PerMinerTeBars._render(mid);
      PerMinerTeBars._reorder();
      if (typeof FleetKPI !== 'undefined') FleetKPI.refresh();
    }, PerMinerTeBars._tickIntervalMs);
  },

  refresh(minerId) {
    if (!PerMinerTeBars._rows[minerId]) return;
    PerMinerTeBars._render(minerId);
    PerMinerTeBars._reorder();
  },

  _jitter(mid) {
    const m = PerMinerTeBars._fleetMiners[mid];
    if (!m) return 0;
    const t = Date.now();
    return PerMinerTeBars._jitterAmplitude
         * Math.sin((2 * Math.PI * t) / m.periodMs + m.phase);
  },

  // Shared drift across the whole fleet so the aggregate Fleet TE also
  // breathes instead of being washed out by averaging independent sines.
  _globalDrift() {
    const t = Date.now();
    return PerMinerTeBars._globalAmplitude
         * Math.sin((2 * Math.PI * t) / PerMinerTeBars._globalPeriodMs);
  },

  _currentTe(mid) {
    const m = PerMinerTeBars._fleetMiners[mid];
    if (!m) return 0;
    const te = m.baseTe
             + PerMinerTeBars._jitter(mid)
             + PerMinerTeBars._globalDrift();
    return Math.max(0, Math.min(100, te));
  },

  _render(mid) {
    const row = PerMinerTeBars._rows[mid];
    if (!row) return;
    const te = PerMinerTeBars._currentTe(mid);
    if (row.fill) row.fill.style.width = `${te}%`;
    if (row.val) row.val.textContent = te.toFixed(0);
  },

  // TEs for the visible top-10 spotlight (used when sorting rows).
  _allCurrentTes() {
    const out = [];
    for (const mid in PerMinerTeBars._rows) {
      out.push(PerMinerTeBars._currentTe(mid));
    }
    return out;
  },

  // TEs for all 50 miners — Fleet TE is the mean of these so the
  // aggregate mirrors the whole fleet, not just the spotlight.
  _allFleetTes() {
    const out = [];
    for (const mid in PerMinerTeBars._fleetMiners) {
      out.push(PerMinerTeBars._currentTe(mid));
    }
    return out;
  },

  // Re-append rows to their parent in descending TE order. Keeps the
  // spotlight list visually sorted: when a flagged miner's bar dips
  // its row slides down, so the reader still sees a clean descending
  // gradient top-to-bottom instead of a scrambled column.
  _reorder() {
    const entries = Object.keys(PerMinerTeBars._rows).map((mid) => ({
      mid, te: PerMinerTeBars._currentTe(mid), el: PerMinerTeBars._rows[mid].el,
    })).filter((e) => e.el);
    if (entries.length === 0) return;
    entries.sort((a, b) => b.te - a.te);
    const parent = entries[0].el.parentNode;
    if (!parent) return;
    entries.forEach((e) => parent.appendChild(e.el));
  },
};

// ── Miner flag state ──────────────────────────────────────────────────────────
/* Tracks which miners currently have at least one active flag, and
   applies the flag-active-{warn,crit} class on the fleet grid cell.
   Users asked: 'a miner should change color ONLY when there's a flag
   about it.' */

const MinerFlagState = {
  /** miner_id → { severities: Multiset, flags: Set<flag_id>, throttled: bool } */
  _state: {},

  _apply(minerId) {
    const el = FleetMap._cells[minerId];
    if (el) {
      el.classList.remove('flag-active-warn', 'flag-active-crit', 'flag-throttled');
      const s = MinerFlagState._state[minerId];
      if (s) {
        if (s.flags.size > 0) {
          if (s.severities.crit > 0) {
            el.classList.add('flag-active-crit');
          } else if (s.severities.warn > 0) {
            el.classList.add('flag-active-warn');
          }
        } else if (s.throttled) {
          // No active flag — keep the throttled badge until a new flag
          // on the same miner supersedes it. (User request: throttled
          // miners stay orange.)
          el.classList.add('flag-throttled');
        }
      }
    }
    // Mirror flag state into the per-miner TE bar + the fleet aggregate.
    if (typeof PerMinerTeBars !== 'undefined') PerMinerTeBars.refresh(minerId);
    if (typeof FleetKPI !== 'undefined') FleetKPI.refresh();
  },

  addFlag(minerId, flagId, severity) {
    if (!minerId || !flagId) return;
    const s = MinerFlagState._state[minerId] ||= {
      severities: { info: 0, warn: 0, crit: 0 },
      flags: new Set(),
      throttled: false,
    };
    if (s.flags.has(flagId)) return;
    s.flags.add(flagId);
    const sev = (severity in s.severities) ? severity : 'info';
    s.severities[sev] += 1;
    MinerFlagState._apply(minerId);
  },

  resolveFlag(minerId, flagId, severity, autonomyLevel) {
    if (!minerId || !flagId) return;
    const s = MinerFlagState._state[minerId];
    if (!s || !s.flags.has(flagId)) return;
    s.flags.delete(flagId);
    const sev = (severity in s.severities) ? severity : 'info';
    s.severities[sev] = Math.max(0, s.severities[sev] - 1);

    // L3_bounded_auto (throttle/migrate) is the concrete intervention
    // where the system physically changes the miner's operating state.
    // The cell stays orange until superseded by the next event.
    if (autonomyLevel && autonomyLevel.startsWith('L3')) {
      s.throttled = true;
    }
    MinerFlagState._apply(minerId);
  },
};


// ── Flags feed ────────────────────────────────────────────────────────────────

/* Flag-type → { primary, fallback } dispatch (mirrors agents/maestro.py
   DISPATCH_TABLE). Lets the UI narrate which specialist is being
   consulted while a flag is pending. */
const _DISPATCH_TABLE = {
  voltage_drift:                { primary: 'voltage',    fallback: 'power'    },
  hashrate_degradation:         { primary: 'hashrate',   fallback: 'voltage'  },
  chip_instability_precursor:   { primary: 'hashrate',   fallback: 'voltage'  },
  hashboard_failure_precursor:  { primary: 'hashrate',   fallback: 'voltage'  },
  thermal_runaway:              { primary: 'environment',fallback: 'voltage'  },
  fan_anomaly:                  { primary: 'environment',fallback: null       },
  power_instability:            { primary: 'power',      fallback: 'voltage'  },
  chip_variance_high:           { primary: 'voltage',    fallback: 'hashrate' },
  anomaly_composite:            { primary: 'all',        fallback: null       },
};

/* Work-in-progress phase machine. Driven by elapsed-time since flag
   arrival so the narrative feels "Orchestra is thinking" during the
   ~15-25 s normally between flag and decision. */
function _orchestraStatusFor(flagType, elapsedS) {
  const d = _DISPATCH_TABLE[flagType] || { primary: '?', fallback: null };
  const primary = d.primary;
  const fallback = d.fallback;

  if (elapsedS < 3)   return `Maestro dispatching → ${primary}_agent…`;
  if (elapsedS < 10)  return `${primary}_agent reasoning…`;
  if (elapsedS < 15 && fallback) return `consulting ${fallback}_agent…`;
  return 'Maestro synthesizing decision…';
}

const FlagsFeed = {
  _container: null,
  _counter: null,
  _maxRows: 30,
  /** flag_id → { row, resolved, startTs(ms), flagType } */
  _byId: {},
  _activeCount: 0,
  _tickHandle: null,

  init() {
    FlagsFeed._container = document.getElementById('flags-feed');
    FlagsFeed._counter = document.getElementById('hero-flag-count');
    // Index pre-rendered flag rows from the initial page HTML so live
    // decisions can mark them resolved.
    if (FlagsFeed._container) {
      const now = Date.now();
      FlagsFeed._container.querySelectorAll('.flag-row').forEach((row) => {
        const fid = row.dataset.flagId;
        if (fid) {
          const minerId = row.querySelector('.flag-miner-id')?.textContent?.trim() || '';
          const severity = row.querySelector('.flag-type-chip')?.classList?.contains('flag-type-chip--crit')
            ? 'crit'
            : (row.querySelector('.flag-type-chip')?.classList?.contains('flag-type-chip--warn') ? 'warn' : 'info');
          FlagsFeed._byId[fid] = {
            row,
            resolved: false,
            startTs: now,
            flagType: row.dataset.flagType || '',
            minerId,
            severity,
          };
          FlagsFeed._activeCount += 1;
          MinerFlagState.addFlag(minerId, fid, severity);
        }
      });
    }
    FlagsFeed._syncCounter();
    FlagsFeed._startTick();
  },

  _startTick() {
    if (FlagsFeed._tickHandle !== null) return;
    FlagsFeed._tickHandle = setInterval(FlagsFeed._tick, 1000);
  },

  _tick() {
    const now = Date.now();
    for (const fid in FlagsFeed._byId) {
      const entry = FlagsFeed._byId[fid];
      if (entry.resolved) continue;
      const elapsedS = (now - entry.startTs) / 1000;
      const status = _orchestraStatusFor(entry.flagType, elapsedS);
      const pill = entry.row.querySelector('.flag-working');
      if (pill) pill.textContent = status;
    }
  },

  _syncCounter() {
    if (FlagsFeed._counter) {
      FlagsFeed._counter.textContent = String(FlagsFeed._activeCount);
    }
  },

  /** Prepend a flag_raised event to the live feed. */
  prepend(envelope) {
    if (!FlagsFeed._container) return;
    const d = envelope.data ?? {};
    const fid = d.flag_id ?? '';
    const sev = d.severity ?? 'info';
    const flagType = d.flag_type ?? '';

    const row = document.createElement('div');
    row.className = 'flag-row';
    row.dataset.flagId = fid;
    row.dataset.flagType = flagType;
    const startStatus = _orchestraStatusFor(flagType, 0);
    row.innerHTML = `
      <span class="flag-type-chip flag-type-chip--${sev}" title="severity: ${sev}">${flagType || '?'}</span>
      <span class="flag-miner-id">${d.miner_id ?? '?'}</span>
      <span class="flag-source text-mute fs-tiny">${d.source_tool ?? ''}</span>
      <span class="flag-working">${startStatus}</span>
      <span class="flag-resolution text-mute fs-tiny"></span>
      <span class="flag-ts">${fmtTime(envelope.ts)}</span>
    `;

    FlagsFeed._container.insertBefore(row, FlagsFeed._container.firstChild);
    FlagsFeed._byId[fid] = {
      row, resolved: false, startTs: Date.now(),
      flagType, minerId: d.miner_id ?? '', severity: sev,
    };
    FlagsFeed._activeCount += 1;
    FlagsFeed._syncCounter();
    MinerFlagState.addFlag(d.miner_id, fid, sev);

    // Trim overflow
    while (FlagsFeed._container.children.length > FlagsFeed._maxRows) {
      const removed = FlagsFeed._container.removeChild(FlagsFeed._container.lastChild);
      const removedFid = removed.dataset.flagId;
      if (removedFid && FlagsFeed._byId[removedFid]) {
        if (!FlagsFeed._byId[removedFid].resolved) {
          FlagsFeed._activeCount = Math.max(0, FlagsFeed._activeCount - 1);
        }
        delete FlagsFeed._byId[removedFid];
      }
    }
  },

  /** Mark a flag as resolved when its originating decision lands. */
  markResolved(flagId, autonomyLevel, action) {
    const entry = FlagsFeed._byId[flagId];
    if (!entry || entry.resolved) return;
    entry.resolved = true;
    FlagsFeed._activeCount = Math.max(0, FlagsFeed._activeCount - 1);
    FlagsFeed._syncCounter();
    entry.row.classList.add('flag-row--resolved');
    MinerFlagState.resolveFlag(entry.minerId, flagId, entry.severity, autonomyLevel);
    // Clear the "Orchestra is working…" pill
    const working = entry.row.querySelector('.flag-working');
    if (working) working.textContent = '';
    const autClass = (autonomyLevel || '').slice(0, 2).toLowerCase();
    const tag = entry.row.querySelector('.flag-resolution');
    if (tag) {
      tag.innerHTML = `✓ <span class="flag-resolution-level flag-resolution-level--${autClass}">${autonomyLevel || ''}</span>`;
    }
  },
};

// ── Reasoning trace log ───────────────────────────────────────────────────────

const TraceFeed = {
  _container: null,
  _maxRows: 20,

  init() {
    TraceFeed._container = document.getElementById('trace-feed');
  },

  prepend(envelope) {
    if (!TraceFeed._container) return;
    const d = envelope.data ?? {};

    const agents = (d.consulted_agents ?? []).join(', ') || '—';
    const autonomy = d.autonomy_level ?? '';
    const autonomyClass = autonomy ? autonomy.slice(0, 2).toLowerCase() : '';
    const trace = d.reasoning_trace ?? '';

    const row = document.createElement('div');
    row.className = 'trace-row';
    row.innerHTML = `
      <div class="trace-header">
        <span class="trace-miner">${d.miner_id ?? '?'}</span>
        <span class="trace-action">${d.action ?? '?'}</span>
        <span class="trace-autonomy trace-autonomy--${autonomyClass}">[${autonomy}]</span>
        <span class="trace-ts">${fmtTime(envelope.ts)}</span>
      </div>
      <div class="trace-agents text-mute fs-tiny mb1">agents: ${agents}</div>
      <div class="trace-reasoning">${escapeHtml(trace)}</div>
    `;

    // Click-to-expand: toggles full reasoning trace visibility.
    row.addEventListener('click', () => {
      row.classList.toggle('trace-row--expanded');
    });

    TraceFeed._container.insertBefore(row, TraceFeed._container.firstChild);

    while (TraceFeed._container.children.length > TraceFeed._maxRows) {
      TraceFeed._container.removeChild(TraceFeed._container.lastChild);
    }

    // Mark the originating flag as resolved in the flags feed.
    if (d.flag_id) {
      FlagsFeed.markResolved(d.flag_id, autonomy, d.action);
    }
  },
};

// ── Env strip ─────────────────────────────────────────────────────────────────

const EnvStrip = {
  _temp: null,
  _humidity: null,
  _elec: null,
  _hashprice: null,

  init() {
    EnvStrip._temp      = document.getElementById('env-temp');
    EnvStrip._humidity  = document.getElementById('env-humidity');
    EnvStrip._elec      = document.getElementById('env-elec');
    EnvStrip._hashprice = document.getElementById('env-hashprice');
  },

  update(env) {
    if (!env) return;
    if (EnvStrip._temp)      EnvStrip._temp.textContent      = `${fmtNum(env.site_temp_c, 1)}°C`;
    if (EnvStrip._humidity)  EnvStrip._humidity.textContent  = `${fmtNum(env.site_humidity_pct, 0)}%`;
    if (EnvStrip._elec)      EnvStrip._elec.textContent      = `$${fmtNum(env.elec_price_usd_kwh, 4)}/kWh`;
    if (EnvStrip._hashprice) EnvStrip._hashprice.textContent = `$${fmtNum(env.hashprice_usd_per_th_day, 4)}/TH·d`;
  },
};

// ── Sparkline (simple SVG path) ───────────────────────────────────────────────

const Sparkline = {
  /**
   * Draw a simple SVG sparkline into a container element.
   * @param {HTMLElement} container
   * @param {number[]} values
   * @param {object} opts
   * @param {string} opts.color  CSS color string
   * @param {number} opts.w      width in px
   * @param {number} opts.h      height in px
   */
  draw(container, values, opts = {}) {
    if (!container || !values || values.length < 2) return;
    const w = opts.w ?? container.clientWidth ?? 120;
    const h = opts.h ?? 40;
    const color = opts.color ?? 'var(--color-status-ok)';

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    const step = w / (values.length - 1);
    const pts = values.map((v, i) => {
      const x = i * step;
      const y = h - ((v - min) / range) * (h - 4) - 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });

    container.innerHTML = `
      <svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg">
        <polyline
          fill="none"
          stroke="${color}"
          stroke-width="1.5"
          stroke-linejoin="round"
          stroke-linecap="round"
          points="${pts.join(' ')}"
        />
      </svg>
    `;
  },
};

// ── HTML escape ───────────────────────────────────────────────────────────────

function escapeHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ── Main dashboard initialisation ─────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  FleetMap.init();
  FleetKPI.init();
  FlagsFeed.init();
  TraceFeed.init();
  EnvStrip.init();

  // Clock
  if (typeof startClockTicker === 'function') {
    startClockTicker('#ts-live');
  }

  // SSE — fleet snapshots (drives fleet map + KPIs + env strip)
  const snapshotStream = new MDKEventStream('snapshot', (env) => {
    if (env.event !== 'fleet_snapshot') return;
    const d = env.data ?? {};
    FleetMap.applySnapshot(d.miners ?? {});
    FleetKPI.update(d.fleet_te ?? 0, d.fleet_hsi ?? 0);
    EnvStrip.update(d.env);
  }, { replayLimit: 5 });
  snapshotStream.connect();

  // SSE — flags (drives flags feed)
  const flagStream = new MDKEventStream('flag', (env) => {
    if (env.event !== 'flag_raised') return;
    FlagsFeed.prepend(env);
    // Feed the rolling-window flag counter so Fleet TE/HSI dip with
    // flag pressure even after the decision arrives.
    FleetKPI.recordFlag();

    // Also update the specific miner cell severity hint
    const mid = env.data?.miner_id;
    const sev = env.data?.severity;
    if (mid && sev === 'crit') {
      const cell = document.querySelector(`.miner-cell[data-miner-id="${mid}"]`);
      if (cell) {
        cell.classList.remove('status-ok', 'status-warn', 'status-imm', 'status-shut');
        cell.classList.add('status-imm');
      }
    }
  }, { replayLimit: 30 });
  flagStream.connect();

  // SSE — decisions (drives trace log)
  const decisionStream = new MDKEventStream('decision', (env) => {
    if (env.event !== 'orchestrator_decision') return;
    TraceFeed.prepend(env);
  }, { replayLimit: 20 });
  decisionStream.connect();
});
