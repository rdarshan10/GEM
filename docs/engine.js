/* GemGraph — a small canvas engine for drawing GEM memory as a living graph.
   Nodes breathe around a home position, changes travel as pulses along edges, state flips ripple,
   rewritten labels cross-fade, and Jev decisions appear as dials. Themes decide the look; the
   scenario helpers replay recorded GEM runs (docs/data/scenarios.json) on top of it. */
(function () {
  "use strict";
  const TAU = Math.PI * 2;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const easeIO = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const easeOut = (t) => 1 - Math.pow(1 - t, 3);
  function rgb(h) {
    h = h.replace("#", "");
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const mix = (a, b, t) => a.map((v, i) => Math.round(lerp(v, b[i], t)));
  const css = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  class GemGraph {
    constructor(canvas, theme, opts = {}) {
      this.c = canvas;
      this.ctx = canvas.getContext("2d");
      this.theme = theme;
      this.opts = Object.assign({ drag: false, labels: true, offsetX: 0, offsetY: 0, labelWidth: 170 }, opts);
      this.nodes = new Map();
      this.edges = [];
      this.fx = [];
      this.notes = [];
      this.cam = { x: 0, y: 0, s: 0.6, ox: this.opts.offsetX, oy: this.opts.offsetY };
      this.camT = { x: 0, y: 0, s: 0.6, ox: this.opts.offsetX, oy: this.opts.offsetY };
      this.time = 0;
      this.last = 0;
      this.timers = [];
      this.wrapCache = new Map();
      this.hover = null;
      this.active = true;
      this.resize();
      if ("IntersectionObserver" in window)
        new IntersectionObserver((es) => { this.active = es.some((e) => e.isIntersecting); }, { rootMargin: "120px" }).observe(canvas);
      if ("ResizeObserver" in window) new ResizeObserver(() => this.resize()).observe(canvas);
      if (this.opts.drag) this._bindPointer();
      requestAnimationFrame((t) => this._frame(t));
    }

    resize() {
      const r = this.c.getBoundingClientRect();
      this.dpr = Math.min(2, window.devicePixelRatio || 1);
      this.w = Math.max(1, r.width);
      this.h = Math.max(1, r.height);
      this.c.width = Math.round(this.w * this.dpr);
      this.c.height = Math.round(this.h * this.dpr);
      if (this._framed) this.frame(...this._framed); // keep the framing right when the canvas changes size
    }

    // ---------------------------------------------------------------- model
    color(state) { return rgb(this.theme.states[state] || this.theme.states.active); }

    addNode(id, o) {
      const c = this.color(o.state || "active");
      const n = {
        id, label: o.label || "", x: o.x, y: o.y, hx: o.x, hy: o.y, bx: o.x, by: o.y, r: o.r || 6,
        state: o.state || "active", col: c, colFrom: c, colTo: c, tState: 1,
        alpha: o.alpha ?? 1, alphaT: o.alpha ?? 1, phase: Math.random() * TAU,
        labelOld: null, labelT: 1, badge: "", badgeT: 0, dial: null,
        kind: o.kind || "fact", group: o.group || null, showLabel: o.showLabel ?? true, labelA: 1, labelAT: 1,
        drop: 1,
      };
      this.nodes.set(id, n);
      return n;
    }
    note(x, y, text) { const n = { x, y, text, a: 0, aT: 0 }; this.notes.push(n); return n; }
    // a framed panel in world space with a title and a status line, drawn behind the graph
    card(x0, y0, x1, y1, title) { const c = { card: true, x0, y0, x1, y1, title, status: "", active: 0, activeT: 0, a: 0, aT: 0 }; this.notes.push(c); return c; }
    showNotes(on) { this.notes.forEach((n) => { n.aT = on ? 1 : 0; }); }
    clear() { this.notes = []; this.nodes.clear(); this.edges = []; this.fx = []; this.timers.forEach((t) => t.r()); this.timers = []; }
    addEdge(a, b, type = "derived") { this.edges.push({ a, b, type, alpha: 1 }); }
    removeNode(id) {
      this.nodes.delete(id);
      this.edges = this.edges.filter((e) => e.a !== id && e.b !== id);
    }

    setState(id, state, o = {}) {
      const n = this.nodes.get(id);
      if (!n) return;
      n.colFrom = n.col.slice();
      n.colTo = this.color(state);
      n.tState = 0;
      n.state = state;
      if (o.label != null && o.label !== n.label) { n.labelOld = n.label; n.label = o.label; n.labelT = 0; }
      if (o.badge != null) { n.badge = o.badge; n.badgeT = 0; }
      if (o.ripple !== false) this.fx.push({ type: "ripple", id, t: 0, dur: 1100, col: n.colTo });
    }

    setBadge(id, text) { const n = this.nodes.get(id); if (n) { n.badge = text; n.badgeT = 0; } }
    dial(id, value, label, state) {
      const n = this.nodes.get(id);
      if (n) n.dial = { v: clamp(value, 0, 1), t: 0, label, col: this.color(state || "accent") };
    }
    clearDials() { this.nodes.forEach((n) => { n.dial = null; }); }

    pulse(a, b, o = {}) {
      return new Promise((resolve) => {
        if (!this.nodes.get(a) || !this.nodes.get(b)) return resolve();
        this.fx.push({ type: "pulse", a, b, t: 0, dur: (o.dur || 750) / (o.speed || 1), col: rgb(o.color || this.theme.pulse), resolve });
      });
    }
    sonar(id, o = {}) {
      this.fx.push({ type: "sonar", id, t: 0, dur: o.dur || 1200, max: o.radius || 320, col: rgb(o.color || this.theme.pulse) });
    }
    dropIn(id) { const n = this.nodes.get(id); if (n) n.drop = 0; }

    // focus: dim everything outside `ids` (null = everything visible)
    setFocus(ids, dim = this.theme.dim ?? 0.18) {
      const set = ids ? new Set(ids) : null;
      this.nodes.forEach((n) => {
        n.alphaT = !set || set.has(n.id) ? 1 : dim;
        n.labelAT = !set || set.has(n.id) ? 1 : 0;
      });
    }
    showLabels(ids, on) { ids.forEach((id) => { const n = this.nodes.get(id); if (n) n.labelAT = on ? 1 : 0; }); }

    frame(ids, pad = 90, maxS = 1.3, minS = 0.18, extra = [], ox = this.opts.offsetX, oy = this.opts.offsetY) {
      this._framed = [ids, pad, maxS, minS, extra, ox, oy];
      const pts = ids.map((id) => this.nodes.get(id)).filter(Boolean).map((n) => [n.hx, n.hy]).concat(extra);
      if (!pts.length) return;
      const lw = this.opts.labels ? this.opts.labelWidth : 0;
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      pts.forEach(([x, y]) => { x0 = Math.min(x0, x); x1 = Math.max(x1, x + lw); y0 = Math.min(y0, y - 20); y1 = Math.max(y1, y + 40); });
      const availW = this.w - 2 * pad - Math.abs(ox) * 2;
      const availH = this.h - 2 * pad - Math.abs(oy) * 2;
      const s = clamp(Math.min(availW / Math.max(1, x1 - x0), availH / Math.max(1, y1 - y0)), minS, maxS);
      this.camT = { x: (x0 + x1) / 2, y: (y0 + y1) / 2, s, ox, oy };
    }
    snap() { this.cam = { ...this.camT }; }

    wait(ms) { return new Promise((r) => { this.timers.push({ at: this.time + ms / 1000, r }); }); }

    // ---------------------------------------------------------------- pointer (drag + hover)
    _bindPointer() {
      let drag = null;
      const pos = (e) => { const r = this.c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
      this.c.addEventListener("pointerdown", (e) => {
        const [x, y] = pos(e);
        const n = this.pick(x, y);
        if (n) { drag = n; this.hover = n; this.c.setPointerCapture(e.pointerId); }
      });
      this.c.addEventListener("pointermove", (e) => {
        const [x, y] = pos(e);
        if (drag) {
          const [wx, wy] = this.toWorld(x, y);
          drag.hx = drag.bx = wx; drag.hy = drag.by = wy;
        } else {
          this.hover = this.pick(x, y);
          this.c.style.cursor = this.hover ? "grab" : "default";
        }
      });
      const end = () => { drag = null; };
      this.c.addEventListener("pointerup", end);
      this.c.addEventListener("pointercancel", end);
      this.c.addEventListener("pointerleave", () => { this.hover = null; });
    }
    toScreen(x, y) {
      return [(x - this.cam.x) * this.cam.s + this.w / 2 + this.cam.ox, (y - this.cam.y) * this.cam.s + this.h / 2 + this.cam.oy];
    }
    toWorld(x, y) {
      return [(x - this.w / 2 - this.cam.ox) / this.cam.s + this.cam.x, (y - this.h / 2 - this.cam.oy) / this.cam.s + this.cam.y];
    }
    pick(x, y) {
      let best = null, bd = this.opts.compact ? 30 : 18;
      this.nodes.forEach((n) => {
        if (n.alpha < 0.4) return;
        const [sx, sy] = this.toScreen(n.x, n.y);
        const d = Math.hypot(sx - x, sy - y);
        if (d < bd) { bd = d; best = n; }
      });
      return best;
    }

    // ---------------------------------------------------------------- loop
    _frame(ts) {
      const dt = Math.min(0.05, (ts - (this.last || ts)) / 1000);
      this.last = ts;
      // off-screen or hidden: keep time and effects moving (so awaited pulses resolve) but skip drawing
      this.time += dt;
      this._update(dt);
      if (this.active && !document.hidden) this._draw();
      requestAnimationFrame((t) => this._frame(t));
    }

    _update(dt) {
      const k = 1 - Math.exp(-dt * (this.theme.camSpeed || 2.6));
      this.cam.x = lerp(this.cam.x, this.camT.x, k);
      this.cam.y = lerp(this.cam.y, this.camT.y, k);
      this.cam.s = lerp(this.cam.s, this.camT.s, k);
      this.cam.ox = lerp(this.cam.ox, this.camT.ox, k);
      this.cam.oy = lerp(this.cam.oy, this.camT.oy, k);
      const amp = reduceMotion ? 0 : this.theme.breath ?? 3;
      this.nodes.forEach((n) => {
        const kb = 1 - Math.exp(-dt * 3);   // ease toward home, so a new layout morphs in
        n.bx = lerp(n.bx, n.hx, kb);
        n.by = lerp(n.by, n.hy, kb);
        n.x = n.bx + Math.sin(this.time * 0.55 + n.phase) * amp;
        n.y = n.by + Math.cos(this.time * 0.47 + n.phase * 1.7) * amp;
        n.alpha = lerp(n.alpha, n.alphaT, 1 - Math.exp(-dt * 4));
        n.labelA = lerp(n.labelA, n.labelAT, 1 - Math.exp(-dt * 4));
        if (n.tState < 1) n.tState = Math.min(1, n.tState + dt / 0.55);
        n.col = mix(n.colFrom, n.colTo, easeIO(n.tState));
        if (n.labelT < 1) n.labelT = Math.min(1, n.labelT + dt / 0.7);
        if (n.badgeT < 1) n.badgeT = Math.min(1, n.badgeT + dt / 0.5);
        if (n.drop < 1) n.drop = Math.min(1, n.drop + dt / 0.9);
        if (n.dial && n.dial.t < 1) n.dial.t = Math.min(1, n.dial.t + dt / 0.8);
      });
      for (const n of this.notes) {
        n.a = lerp(n.a, n.aT, 1 - Math.exp(-dt * 3));
        if (n.card) n.active = lerp(n.active, n.activeT, 1 - Math.exp(-dt * 5));
      }
      for (const f of this.fx) f.t += (dt * 1000) / f.dur;
      this.fx = this.fx.filter((f) => {
        if (f.t < 1) return true;
        if (f.resolve) f.resolve();
        return false;
      });
      this.timers = this.timers.filter((tm) => { if (this.time >= tm.at) { tm.r(); return false; } return true; });
    }

    _curve(a, b) {
      const [ax, ay] = this.toScreen(a.x, a.y - (1 - easeOut(a.drop)) * 220);
      const [bx, by] = this.toScreen(b.x, b.y - (1 - easeOut(b.drop)) * 220);
      const mx = (ax + bx) / 2, my = (ay + by) / 2;
      const dx = bx - ax, dy = by - ay, len = Math.hypot(dx, dy) || 1;
      const bend = this.theme.bend ?? 0.12;
      return { ax, ay, bx, by, cx: mx - (dy / len) * len * bend, cy: my + (dx / len) * len * bend, len };
    }
    _pt(q, t) {
      const u = 1 - t;
      return [u * u * q.ax + 2 * u * t * q.cx + t * t * q.bx, u * u * q.ay + 2 * u * t * q.cy + t * t * q.by];
    }

    _draw() {
      const ctx = this.ctx, th = this.theme;
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      ctx.clearRect(0, 0, this.w, this.h);
      if (th.background) th.background(ctx, this);
      const sc = clamp(this.cam.s, 0.55, 1.35);

      // edges
      for (const e of this.edges) {
        const a = this.nodes.get(e.a), b = this.nodes.get(e.b);
        if (!a || !b) continue;
        const al = Math.min(a.alpha, b.alpha) * e.alpha * Math.min(a.drop, b.drop);
        if (al < 0.02) continue;
        const q = this._curve(a, b);
        ctx.beginPath();
        ctx.moveTo(q.ax, q.ay);
        ctx.quadraticCurveTo(q.cx, q.cy, q.bx, q.by);
        const col = rgb(e.type === "trigger" ? th.pulse : th.edge);
        ctx.strokeStyle = css(col, al * (e.type === "assoc" ? 0.6 : th.edgeAlpha ?? 0.55));
        ctx.lineWidth = (e.type === "trigger" ? 1.6 : 1.15) * sc;
        ctx.setLineDash(e.type === "assoc" ? [4, 5] : []);
        ctx.stroke();
        ctx.setLineDash([]);
        if (e.type !== "assoc") { // direction chevron at 58%
          const [px, py] = this._pt(q, 0.58), [qx, qy] = this._pt(q, 0.6);
          const ang = Math.atan2(qy - py, qx - px), s = 4.5 * sc;
          ctx.beginPath();
          ctx.moveTo(px - Math.cos(ang - 0.6) * s, py - Math.sin(ang - 0.6) * s);
          ctx.lineTo(px, py);
          ctx.lineTo(px - Math.cos(ang + 0.6) * s, py - Math.sin(ang + 0.6) * s);
          ctx.stroke();
        }
      }

      // cards and notes (e.g. memory-group panels), behind everything else
      for (const nt of this.notes) {
        if (nt.a < 0.02) continue;
        if (nt.card) {
          const [x0, y0] = this.toScreen(nt.x0, nt.y0), [x1, y1] = this.toScreen(nt.x1, nt.y1);
          ctx.globalAlpha = nt.a;
          ctx.beginPath();
          ctx.roundRect(x0, y0, x1 - x0, y1 - y0, 12);
          ctx.fillStyle = th.cardFill || "rgba(255,255,255,.03)";
          ctx.fill();
          ctx.lineWidth = 1;
          ctx.strokeStyle = th.cardLine || "rgba(255,255,255,.1)";
          ctx.stroke();
          if (nt.active > 0.02) {
            ctx.globalAlpha = nt.a * nt.active;
            ctx.strokeStyle = th.cardActive || th.pulse;
            ctx.lineWidth = 1.4;
            ctx.stroke();
          }
          ctx.globalAlpha = nt.a;
          ctx.font = th.cardTitleFont || "500 12px monospace";
          ctx.fillStyle = th.text;
          ctx.fillText(nt.title, x0 + 16, y0 + 26);
          if (nt.status) {
            ctx.font = th.cardStatusFont || "400 11.5px monospace";
            ctx.fillStyle = nt.active > 0.5 ? (th.cardActive || th.pulse) : (th.noteColor || th.text);
            let st = nt.status;
            const room = x1 - x0 - 32;
            while (st.length > 4 && ctx.measureText(st).width > room) st = st.slice(0, -2).trimEnd() + "…";
            ctx.fillText(st, x0 + 16, y1 - 16);
          }
          ctx.globalAlpha = 1;
          continue;
        }
        const [x, y] = this.toScreen(nt.x, nt.y);
        ctx.globalAlpha = nt.a;
        ctx.font = th.noteFont || "500 11px monospace";
        ctx.fillStyle = th.noteColor || th.text;
        ctx.textAlign = "center";
        ctx.fillText(nt.text, x, y);
        ctx.textAlign = "start";
        ctx.globalAlpha = 1;
      }

      // effects under nodes
      for (const f of this.fx) {
        if (f.type === "sonar") {
          const n = this.nodes.get(f.id);
          if (!n) continue;
          const [x, y] = this.toScreen(n.x, n.y);
          for (let i = 0; i < 2; i++) {
            const tt = clamp(f.t - i * 0.18, 0, 1);
            if (tt <= 0) continue;
            ctx.beginPath();
            ctx.arc(x, y, easeOut(tt) * f.max * this.cam.s, 0, TAU);
            ctx.strokeStyle = css(f.col, (1 - tt) * 0.5);
            ctx.lineWidth = 1.2;
            ctx.stroke();
          }
        } else if (f.type === "pulse") {
          const a = this.nodes.get(f.a), b = this.nodes.get(f.b);
          if (!a || !b) continue;
          const q = this._curve(a, b);
          const t = easeIO(clamp(f.t, 0, 1));
          const tail = 0.22;
          ctx.lineCap = "round";
          for (let i = 0; i < 10; i++) {
            const t0 = clamp(t - tail * (i / 10), 0, 1), t1 = clamp(t - tail * ((i + 1) / 10), 0, 1);
            const [x0, y0] = this._pt(q, t0), [x1, y1] = this._pt(q, t1);
            ctx.beginPath();
            ctx.moveTo(x0, y0);
            ctx.lineTo(x1, y1);
            ctx.strokeStyle = css(f.col, (1 - i / 10) * 0.95);
            ctx.lineWidth = (3.2 - i * 0.22) * sc;
            ctx.stroke();
          }
          const [hx, hy] = this._pt(q, t);
          ctx.save();
          ctx.shadowColor = css(f.col, 0.9);
          ctx.shadowBlur = th.glow ?? 14;
          ctx.beginPath();
          ctx.arc(hx, hy, 3.4 * sc, 0, TAU);
          ctx.fillStyle = css(f.col, 1);
          ctx.fill();
          ctx.restore();
          ctx.lineCap = "butt";
        }
      }

      // nodes
      // label type scales with zoom (within limits) so text and spacing stay in proportion
      const fz = clamp(this.cam.s, 0.9, 1.15);
      const font = th.fontSize ? `${th.fontWeight || 400} ${(th.fontSize * fz).toFixed(1)}px ${th.fontFamily}` : th.font || "13px sans-serif";
      for (const n of this.nodes.values()) {
        if (n.alpha < 0.02) continue;
        const dropY = (1 - easeOut(n.drop)) * 220;
        const [x, y] = this.toScreen(n.x, n.y - dropY);
        if (x < -300 || x > this.w + 300 || y < -120 || y > this.h + 120) continue;
        const r = n.r * sc;
        ctx.globalAlpha = n.alpha * (0.3 + 0.7 * easeOut(n.drop));
        if (th.drawNode) th.drawNode(ctx, n, x, y, r, this);
        else {
          ctx.beginPath();
          ctx.arc(x, y, r, 0, TAU);
          ctx.fillStyle = css(n.col);
          ctx.fill();
        }
        // ripples
        for (const f of this.fx) {
          if (f.type !== "ripple" || f.id !== n.id) continue;
          ctx.beginPath();
          ctx.arc(x, y, r + easeOut(f.t) * 34 * sc, 0, TAU);
          ctx.strokeStyle = css(f.col, (1 - f.t) * 0.8);
          ctx.lineWidth = 1.6;
          ctx.stroke();
        }
        // dial
        if (n.dial) {
          const R = r + 8 * sc, d = n.dial;
          ctx.beginPath();
          ctx.arc(x, y, R, 0, TAU);
          ctx.strokeStyle = css(rgb(th.edge), 0.25);
          ctx.lineWidth = 2.4 * sc;
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(x, y, R, -Math.PI / 2, -Math.PI / 2 + TAU * d.v * easeOut(d.t));
          ctx.strokeStyle = css(d.col);
          ctx.lineWidth = 2.4 * sc;
          ctx.stroke();
        }
        // label
        // compact: label only the fact being decided (the new fact until a decision starts) or a tapped one
        const spot = !this.opts.compact || this.hover === n || this.spotlight === n.id || (n.kind === "trigger" && !this.spotlight);
        if (this.opts.labels && n.showLabel && n.labelA > 0.03 && spot) {
          const lx = x + r + 10 * sc, lw = this.opts.labelWidth * sc;
          ctx.font = font;
          const lh = (th.lineHeight || 16) * fz;
          const drawLabel = (text, a) => {
            const lines = this._wrap(text, lw, font);
            const top = y - ((lines.length - 1) * lh) / 2 + 4;
            ctx.globalAlpha = n.alpha * n.labelA * a;
            lines.forEach((ln, i) => {
              if (th.labelHalo) {
                ctx.lineWidth = 4;
                ctx.strokeStyle = th.labelHalo;
                ctx.strokeText(ln, lx, top + i * lh);
              }
              ctx.fillStyle = n.kind === "trigger" ? th.triggerText || th.text : th.text;
              ctx.fillText(ln, lx, top + i * lh);
            });
            return lines.length;
          };
          let nl = drawLabel(n.label, n.labelT < 1 ? easeIO(n.labelT) : 1);
          if (n.labelOld && n.labelT < 1) drawLabel(n.labelOld, 1 - easeIO(n.labelT));
          if (n.badge) {
            ctx.font = th.badgeFont || "600 10px monospace";
            ctx.globalAlpha = n.alpha * n.labelA * easeOut(n.badgeT);
            ctx.fillStyle = css(n.col);
            const by = y + ((nl - 1) * lh) / 2 + 4 + lh + 1;
            if (th.labelHalo) { ctx.lineWidth = 4; ctx.strokeStyle = th.labelHalo; ctx.strokeText(n.badge, lx, by); }
            ctx.fillText(n.badge, lx, by);
          }
        } else if (this.hover === n && n.label) {
          ctx.font = font;
          ctx.globalAlpha = 1;
          ctx.fillStyle = th.text;
          ctx.fillText(n.label.length > 60 ? n.label.slice(0, 57) + "…" : n.label, x + r + 8, y + 4);
        }
        ctx.globalAlpha = 1;
      }
      if (th.foreground) th.foreground(ctx, this);
    }

    _wrap(text, maxW, font) {
      const key = font + "|" + Math.round(maxW) + "|" + text;
      if (this.wrapCache.has(key)) return this.wrapCache.get(key);
      const ctx = this.ctx;
      ctx.font = font;
      const words = String(text).split(/\s+/), lines = [];
      let cur = "";
      for (const w of words) {
        const t = cur ? cur + " " + w : w;
        if (ctx.measureText(t).width > maxW && cur) { lines.push(cur); cur = w; } else cur = t;
      }
      if (cur) lines.push(cur);
      this.wrapCache.set(key, lines);
      return lines;
    }
  }

  // ------------------------------------------------------------------ scenario helpers
  // layered layout: roots on top, children under their parents (barycentre), trigger above
  GemGraph.layout = function (sc, ox, oy, dx = 200, dy = 112) {
    const by = Object.fromEntries(sc.nodes.map((n) => [n.id, n]));
    const depth = {};
    const d = (id) => depth[id] ?? (depth[id] = by[id].parents.length ? 1 + Math.max(...by[id].parents.map(d)) : 0);
    sc.nodes.forEach((n) => d(n.id));
    const rows = [];
    sc.nodes.forEach((n) => (rows[depth[n.id]] = rows[depth[n.id]] || []).push(n.id));
    const assocA = new Set(sc.assoc.map(([a]) => a));
    const r0 = rows[0].filter((id) => !assocA.has(id));
    const ordered = [];
    r0.forEach((id) => { ordered.push(id); sc.assoc.filter(([, b]) => b === id).forEach(([a]) => ordered.push(a)); });
    rows[0] = ordered;
    const pos = {};
    rows[0].forEach((id, i) => { pos[id] = [i * dx, 0]; });
    for (let r = 1; r < rows.length; r++) {
      const want = rows[r].map((id) => [id, by[id].parents.reduce((s, p) => s + pos[p][0], 0) / by[id].parents.length]).sort((a, b) => a[1] - b[1]);
      let last = -Infinity;
      const placed = want.map(([id, w]) => { const v = Math.max(w, last + dx); last = v; return [id, v, w]; });
      const shift = placed.reduce((s, p) => s + p[2] - p[1], 0) / placed.length;
      placed.forEach(([id, v]) => { pos[id] = [v + shift, r * dy]; });
    }
    const direct = sc.steps.filter((s) => s.kind === "revise" && s.depth === 0).map((s) => s.id);
    const anchor = direct.length ? direct : rows[0];
    const tx = anchor.reduce((s, id) => s + pos[id][0], 0) / anchor.length;
    const xs = Object.values(pos).map((p) => p[0]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const out = {};
    Object.entries(pos).forEach(([id, [x, y]]) => { out[id] = [ox + x - cx, oy + y]; });
    out.__trigger = [ox + tx - cx, oy - dy * 1.05];
    out.__rows = rows.length;
    return out;
  };

  GemGraph.build = function (g, sc, prefix, ox, oy, o = {}) {
    const L = GemGraph.layout(sc, ox, oy, o.dx ?? (g.opts.labels ? g.opts.labelWidth + 85 : 200), o.dy);
    const roots = new Set(sc.nodes.filter((n) => !n.parents.length).map((n) => n.id));
    sc.nodes.forEach((n) => {
      const [x, y] = L[n.id];
      g.addNode(prefix + n.id, { label: n.text, x, y, r: roots.has(n.id) ? 7.5 : 5.5, group: sc.key, alpha: o.alpha ?? 1 });
    });
    sc.nodes.forEach((n) => n.parents.forEach((p) => g.addEdge(prefix + p, prefix + n.id, "derived")));
    sc.assoc.forEach(([a, b]) => g.addEdge(prefix + b, prefix + a, "assoc"));
    return { ids: sc.nodes.map((n) => prefix + n.id), trigger: prefix + "new", home: L.__trigger, layout: L };
  };

  GemGraph.reset = function (g, sc, prefix) {
    g.removeNode(prefix + "new");
    sc.nodes.forEach((n) => {
      const node = g.nodes.get(prefix + n.id);
      if (!node) return;
      node.label = n.text; node.labelOld = null; node.labelT = 1; node.badge = ""; node.dial = null;
      node.colFrom = node.colTo = node.col = g.color("active"); node.state = "active"; node.tState = 1;
    });
  };

  const parentOf = (sc, id, done) => {
    const ps = sc.nodes.find((n) => n.id === id).parents;
    const hit = ps.filter((p) => done.has(p));
    return hit.length ? hit[hit.length - 1] : ps[0];
  };

  // replay a recorded run. mode "gem" walks the cascade; "flat" applies flat memory's end state.
  // opts: speed, log(kind, text), count({jev, llm}), token {cancelled}, dials, badges
  GemGraph.play = async function (g, sc, prefix, home, opts = {}) {
    const sp = opts.speed || 1, tok = opts.token || {}, P = (id) => prefix + id;
    if (tok.cancelled) return;
    const log = opts.log || (() => {}), wait = (ms) => g.wait(ms / sp);
    const live = () => !tok.cancelled;
    const txt = (id) => sc.nodes.find((n) => n.id === id).text;
    let jev = 0, llm = 0;
    const count = () => opts.count && opts.count({ jev, llm });
    GemGraph.reset(g, sc, prefix);
    g.spotlight = null;
    count();
    const trig = g.addNode(P("new"), { label: sc.trigger, x: home[0], y: home[1], r: 8, kind: "trigger" });
    trig.colFrom = trig.colTo = trig.col = g.color("trigger");
    g.dropIn(P("new"));
    log("write", `add_memory("${sc.trigger}")`);
    await wait(900); if (!live()) return;

    // conflict scan (flat memory runs its scan on the LLM; only GEM's scan goes to Jev)
    const scan = sc.steps.find((s) => s.kind === "scan");
    g.sonar(P("new"), { radius: opts.sonarRadius || 420, dur: 1300 / sp });
    if (opts.mode === "flat") log("flat", "conflict scan · similar facts checked by the LLM");
    else if (scan) {
      jev += 1; count();
      log("jev", `conflict scan · ${scan.checked.length} similar facts in one call`);
      await wait(700); if (!live()) return;
      for (const c of scan.checked) {
        if (c.route === "skip") {
          g.setBadge(P(c.id), opts.badges === false ? "" : `no direct conflict · ${c.p_no_conflict.toFixed(2)}`);
          log("jev", `  ${txt(c.id)} → no conflict (${c.p_no_conflict.toFixed(2)})`);
        }
      }
    }
    await wait(500); if (!live()) return;

    if (opts.mode === "flat") {
      for (let i = 0; i < sc.nodes.length; i++) {
        const n = sc.nodes[i], f = sc.flat[i], gem = sc.gem[i];
        const changed = f.status !== "ACTIVE" || f.content !== n.text;
        const gemChanged = gem.status !== "ACTIVE" || gem.review || gem.content !== n.text;
        if (changed || gemChanged) g.spotlight = P(n.id);
        if (changed) {
          g.addEdge(P("new"), P(n.id), "trigger");
          await g.pulse(P("new"), P(n.id), { speed: sp }); if (!live()) return;
          if (f.status === "ACTIVE") g.setState(P(n.id), "updated", { label: f.content, badge: "updated" });
          else g.setState(P(n.id), "stale", { badge: f.status.toLowerCase() });
          log("flat", `${n.text} → ${f.status === "ACTIVE" ? f.content : f.status}`);
        } else if (gemChanged) {
          await wait(260); if (!live()) return;
          g.setState(P(n.id), "wrong", { badge: "outdated · still served as true" });
          log("wrong", `${n.text} (still served as true)`);
        }
      }
      return;
    }

    const done = new Set(), batched = new Set();
    const hop = (from) => { if (!batched.has(from)) { batched.add(from); jev += 1; } };
    for (const s of sc.steps) {
      if (!live()) return;
      if (s.kind === "revise" || s.kind === "stop") g.spotlight = P(s.id);
      if (s.kind === "revise") {
        const from = s.depth === 0 ? "new" : parentOf(sc, s.id, done);
        if (s.depth === 0) g.addEdge(P("new"), P(s.id), "trigger");
        await g.pulse(P(from), P(s.id), { speed: sp }); if (!live()) return;
        if (s.jev && s.depth > 0 && opts.dials !== false) {
          g.dial(P(s.id), s.jev.p_affected, "affected", s.status === "ACTIVE" ? "updated" : "stale");
          await wait(650); if (!live()) return;
        }
        if (s.depth > 0) hop(from);
        if (s.via !== "jev") llm += 1;
        const state = s.status === "ACTIVE" ? "updated" : s.status === "SUPERSEDED" ? "superseded" : "stale";
        const who = opts.who === false ? "" : ` · ${s.via === "jev" ? "Jev" : "LLM"}`;
        const badge = state === "updated" ? `updated${who}` : `stale · review${who}`;
        g.setState(P(s.id), state, { label: state === "updated" ? s.content : undefined, badge: opts.badges === false ? "" : badge });
        if (opts.onStep) opts.onStep({ id: s.id, state, text: state === "updated" ? s.content : txt(s.id), via: s.via });
        log(s.via, `${s.depth === 0 ? "conflict" : "hop " + s.depth} · ${txt(s.id)} → ${state === "updated" ? s.content : "STALE"}${s.jev && s.jev.p_affected != null ? ` (P affected ${s.jev.p_affected.toFixed(2)})` : ""}`);
        done.add(s.id);
        count();
        if (opts.until === "conflict" && s.depth === 0) return;
        await wait(420);
      } else if (s.kind === "stop") {
        const from = parentOf(sc, s.id, done);
        hop(from);
        await g.pulse(P(from), P(s.id), { speed: sp }); if (!live()) return;
        if (s.jev && opts.dials !== false) { g.dial(P(s.id), s.jev.p_unaffected, "unaffected", "kept"); await wait(650); if (!live()) return; }
        g.setState(P(s.id), "kept", { badge: opts.badges === false ? "" : opts.who === false ? "unaffected" : `unaffected · ${s.via === "jev" ? "Jev" : "LLM"}${s.jev ? " " + s.jev.p_unaffected.toFixed(2) : ""}` });
        if (opts.onStep) opts.onStep({ id: s.id, state: "kept", text: txt(s.id), via: s.via });
        if (s.via !== "jev") llm += 1;
        log(s.via, `stop · ${txt(s.id)} unaffected${s.jev ? ` (P unaffected ${s.jev.p_unaffected.toFixed(2)})` : ""}`);
        done.add(s.id);
        count();
        await wait(420);
      }
    }
    // the recorded totals are the honest numbers (scan + one call per hop)
    jev = sc.calls.jev; llm = sc.calls.llm; count();
    log("done", `${sc.calls.jev} Jev calls · ${sc.calls.llm} LLM call${sc.calls.llm === 1 ? "" : "s"}`);
  };

  GemGraph.loadData = (url) => fetch(url).then((r) => r.json());
  window.GemGraph = GemGraph;
})();
