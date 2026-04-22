/**
 * event_stream.js — SSE consumer for MDK Fleet dashboard.
 *
 * Connects to /api/stream/<channel>, receives Server-Sent Events,
 * and dispatches them to registered handlers.
 *
 * Usage:
 *   const es = new MDKEventStream('flag', (env) => handleFlag(env));
 *   es.connect();
 *   // later:
 *   es.disconnect();
 */

class MDKEventStream {
  /**
   * @param {string} channel  - One of: telemetry | kpi | flag | decision | action | snapshot | live
   * @param {function} onEvent - Called with each parsed envelope object
   * @param {object} options
   * @param {number} options.replayLimit  - How many historical events to replay on connect (default 50)
   * @param {boolean} options.fromStart   - Replay full file from start (default false)
   * @param {number} options.reconnectMs  - Reconnect delay on error (default 3000)
   */
  constructor(channel, onEvent, options = {}) {
    this.channel = channel;
    this.onEvent = onEvent;
    this.replayLimit = options.replayLimit ?? 50;
    this.fromStart = options.fromStart ?? false;
    this.reconnectMs = options.reconnectMs ?? 3000;
    this._es = null;
    this._reconnectTimer = null;
    this._active = false;
  }

  connect() {
    this._active = true;
    this._open();
  }

  disconnect() {
    this._active = false;
    if (this._reconnectTimer) clearTimeout(this._reconnectTimer);
    if (this._es) {
      this._es.close();
      this._es = null;
    }
  }

  _open() {
    if (!this._active) return;
    const params = new URLSearchParams({
      replay: this.replayLimit,
      from_start: this.fromStart ? 'true' : 'false',
    });
    const url = `/api/stream/${this.channel}?${params}`;
    this._es = new EventSource(url);

    this._es.onmessage = (evt) => {
      try {
        const envelope = JSON.parse(evt.data);
        this.onEvent(envelope);
      } catch (e) {
        console.warn('[MDKEventStream] bad JSON', evt.data, e);
      }
    };

    this._es.onerror = (_err) => {
      console.warn(`[MDKEventStream] error on channel=${this.channel}, reconnecting in ${this.reconnectMs}ms`);
      this._es.close();
      this._es = null;
      if (this._active) {
        this._reconnectTimer = setTimeout(() => this._open(), this.reconnectMs);
      }
    };
  }
}

// ── Shared utilities ─────────────────────────────────────────────────────────

/**
 * Format an ISO timestamp to HH:MM:SS UTC.
 * @param {string} ts
 * @returns {string}
 */
function fmtTime(ts) {
  if (!ts) return '--:--:--';
  try {
    const d = new Date(ts);
    return d.toISOString().slice(11, 19);
  } catch {
    return ts;
  }
}

/**
 * Format a float to N decimal places, or '—' if null/undefined.
 * @param {number|null|undefined} v
 * @param {number} decimals
 */
function fmtNum(v, decimals = 1) {
  if (v == null || isNaN(v)) return '—';
  return Number(v).toFixed(decimals);
}

/**
 * Clamp value to [0, 100] and return as integer percent string.
 * @param {number} v
 */
function fmtPct(v) {
  if (v == null || isNaN(v)) return '0';
  return Math.max(0, Math.min(100, Math.round(v))).toString();
}

/**
 * CSS class for a miner status string.
 * @param {string} status  'ok' | 'warn' | 'imm' | 'shut'
 */
function statusClass(status) {
  const map = { ok: 'text-ok', warn: 'text-warn', imm: 'text-crit', shut: 'text-shut' };
  return map[status] ?? 'text-mute';
}

/**
 * CSS class for a flag severity.
 * @param {string} severity  'info' | 'warn' | 'crit'
 */
function severityClass(severity) {
  const map = { info: 'text-dim', warn: 'text-warn', crit: 'text-crit' };
  return map[severity] ?? 'text-mute';
}

// ── Live timestamp ticker ─────────────────────────────────────────────────────

function startClockTicker(selector) {
  const el = document.querySelector(selector);
  if (!el) return;
  const update = () => { el.textContent = new Date().toISOString().slice(0, 19) + 'Z'; };
  update();
  setInterval(update, 1000);
}

// Export for module-style usage (ignored if plain <script>)
if (typeof module !== 'undefined') {
  module.exports = { MDKEventStream, fmtTime, fmtNum, fmtPct, statusClass, severityClass, startClockTicker };
}
