/* GEM site: memory explorer (replays recorded GEM runs) + results charts. No dependencies. */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const $ = (s, el = document) => el.querySelector(s);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ------------------------------------------------------------------ theme
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem("gem-theme");
    if (saved === "light" || saved === "dark") root.dataset.theme = saved;
  } catch (e) { /* storage unavailable: follow the OS */ }
  $(".theme-toggle").addEventListener("click", () => {
    const dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("gem-theme", root.dataset.theme); } catch (e) { /* ignore */ }
  });

  // ------------------------------------------------------------------ helpers
  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs || {}) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const pct = (p) => (p == null ? "–" : p.toFixed(2));

  function wrap(text, max) {
    const words = String(text).split(/\s+/);
    const lines = [];
    let cur = "";
    for (const w of words) {
      if (!cur) cur = w;
      else if ((cur + " " + w).length <= max) cur += " " + w;
      else { lines.push(cur); cur = w; }
    }
    if (cur) lines.push(cur);
    return lines;
  }

  const tip = $("#tooltip");
  function showTip(evt, html) {
    tip.innerHTML = html;
    tip.hidden = false;
    const pad = 14;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    const r = tip.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }
  const hideTip = () => { tip.hidden = true; };

  // ------------------------------------------------------------------ explorer
  const NODE_W = 206, CHARS = 28, GAP_X = 16, GAP_Y = 52, PAD = 14, LINE = 16;
  const state = { data: null, si: 0, step: 0, view: "gem", timer: null };

  function buildSteps(sc) {
    const steps = [{ kind: "before" }, { kind: "arrive" }];
    const already = [];
    for (const s of sc.steps) {
      if (s.kind === "already") already.push(s.id);
      else steps.push(s);
    }
    steps.push({ kind: "done", already });
    return steps;
  }

  const byId = (sc) => Object.fromEntries(sc.nodes.map((n, i) => [n.id, { ...n, i }]));

  function gemChanged(sc, i) {
    const g = sc.gem[i];
    return g.status !== "ACTIVE" || g.review || g.content !== sc.nodes[i].text;
  }

  // replay recorded steps up to k -> what each node shows
  function stateAt(sc, steps, k) {
    const nodes = {};
    sc.nodes.forEach((n) => { nodes[n.id] = { st: "active", text: n.text, via: null }; });
    const out = { nodes, trig: k >= 1, trigEdges: [], hot: null, current: [], step: steps[k] };
    const processed = new Set();
    for (let j = 2; j <= k; j++) {
      const s = steps[j];
      if (s.kind === "scan") {
        s.checked.forEach((c) => {
          if (c.route === "skip") { nodes[c.id].st = "checked"; nodes[c.id].via = "JEV"; }
        });
        if (j === k) out.current = s.checked.map((c) => c.id);
      } else if (s.kind === "revise") {
        const n = nodes[s.id];
        n.st = s.status === "STALE" ? "stale" : s.status === "SUPERSEDED" ? "superseded" : "revised";
        if (n.st === "revised") n.text = s.content;
        n.via = s.via.toUpperCase();
        if (s.depth === 0) out.trigEdges.push(s.id);
        if (j === k) {
          out.current = [s.id];
          out.hot = s.depth === 0 ? ["new", s.id] : [parentOf(sc, s.id, processed), s.id];
        }
        processed.add(s.id);
      } else if (s.kind === "stop") {
        nodes[s.id].st = "kept";
        nodes[s.id].via = s.via.toUpperCase();
        if (j === k) { out.current = [s.id]; out.hot = [parentOf(sc, s.id, processed), s.id]; }
        processed.add(s.id);
      }
    }
    return out;
  }

  function parentOf(sc, id, processed) {
    const ps = byId(sc)[id].parents;
    const done = ps.filter((p) => processed.has(p));
    return (done.length ? done[done.length - 1] : ps[0]) || null;
  }

  function flatState(sc) {
    const nodes = {};
    const trigEdges = [];
    sc.nodes.forEach((n, i) => {
      const f = sc.flat[i];
      let st = "active", text = n.text;
      if (f.status === "STALE") st = "stale";
      else if (f.status === "SUPERSEDED") st = "superseded";
      else if (f.content !== n.text) { st = "revised"; text = f.content; }
      else if (gemChanged(sc, i)) st = "wrong";
      if (st !== "active" && st !== "wrong") trigEdges.push(n.id);
      nodes[n.id] = { st, text, via: null };
    });
    return { nodes, trig: true, trigEdges, hot: null, current: [] };
  }

  // top-down layered layout: trigger on top, then depth rows; children centred under parents
  function layout(sc) {
    const map = byId(sc);
    const depth = {};
    const d = (id) => {
      if (depth[id] != null) return depth[id];
      const ps = map[id].parents;
      return (depth[id] = ps.length ? 1 + Math.max(...ps.map(d)) : 0);
    };
    sc.nodes.forEach((n) => d(n.id));
    const assocOf = {};
    sc.assoc.forEach(([a, b]) => { assocOf[a] = b; });
    const hasKids = new Set(sc.nodes.flatMap((n) => n.parents));

    // node heights: room for the longest text it will ever show
    const lines = {};
    sc.nodes.forEach((n, i) => {
      const variants = [n.text, sc.gem[i].content, sc.flat[i].content];
      lines[n.id] = Math.max(...variants.map((t) => wrap(t, CHARS).length));
    });
    const h = (id) => 30 + lines[id] * LINE + 8;

    const rows = [];
    sc.nodes.forEach((n) => { (rows[depth[n.id]] = rows[depth[n.id]] || []).push(n.id); });
    // row 0: roots in fact order, associated facts beside their partner, isolated facts last
    const r0 = rows[0];
    const isolated = r0.filter((id) => !hasKids.has(id) && !assocOf[id] && !sc.assoc.some(([, b]) => b === id));
    const linked = r0.filter((id) => !isolated.includes(id) && !assocOf[id]);
    const ordered = [];
    linked.forEach((id) => {
      ordered.push(id);
      sc.assoc.filter(([, b]) => b === id).forEach(([a]) => ordered.push(a));
    });
    rows[0] = ordered.concat(isolated);

    const x = {};
    rows[0].forEach((id, i) => { x[id] = i * (NODE_W + GAP_X); });
    for (let r = 1; r < rows.length; r++) {
      const want = rows[r].map((id) => {
        const ps = map[id].parents;
        return [id, ps.reduce((s, p) => s + x[p], 0) / ps.length];
      }).sort((a, b) => a[1] - b[1]);
      let last = -Infinity;
      const placed = want.map(([id, w]) => { const v = Math.max(w, last + NODE_W + GAP_X); last = v; return [id, v, w]; });
      const shift = placed.reduce((s, p) => s + (p[2] - p[1]), 0) / placed.length;
      placed.forEach(([id, v]) => { x[id] = v + shift; });
      rows[r] = placed.map((p) => p[0]);
    }
    const ids = sc.nodes.map((n) => n.id);
    const minX = Math.min(...ids.map((id) => x[id]));
    ids.forEach((id) => { x[id] += PAD - minX; });

    // trigger above the facts it directly updates
    const direct = sc.steps.filter((s) => s.kind === "revise" && s.depth === 0).map((s) => s.id);
    const trigLines = wrap(sc.trigger, CHARS).length;
    const trigH = 30 + trigLines * LINE + 8;
    const anchor = direct.length ? direct : rows[0];
    x.new = anchor.reduce((s, id) => s + x[id], 0) / anchor.length;

    const y = { new: PAD };
    let top = PAD + trigH + GAP_Y;
    rows.forEach((row) => {
      const rh = Math.max(...row.map(h));
      row.forEach((id) => { y[id] = top; });
      top += rh + GAP_Y;
    });
    const width = Math.max(...ids.concat("new").map((id) => x[id])) + NODE_W + PAD;
    const minAll = Math.min(x.new, PAD);
    if (minAll < PAD) Object.keys(x).forEach((k) => { x[k] += PAD - minAll; });
    return { x, y, h: (id) => (id === "new" ? trigH : h(id)), width: width + Math.max(0, PAD - minAll), height: top - GAP_Y + PAD };
  }

  const LABEL = {
    active: "MEMORY", new: "NEW FACT", revised: "UPDATED", stale: "STALE · REVIEW",
    kept: "UNAFFECTED", checked: "NO DIRECT CONFLICT", wrong: "OUTDATED · STILL SERVED", superseded: "SUPERSEDED",
  };

  function drawGraph(sc, L, view) {
    const svg = $("#graph");
    svg.innerHTML = "";
    svg.setAttribute("viewBox", `0 0 ${L.width} ${L.height}`);
    // scale down to fit the panel, but never below ~80% (text stays legible; wider graphs scroll)
    svg.setAttribute("width", "100%");
    svg.removeAttribute("height");
    svg.style.maxWidth = L.width + "px";
    svg.style.minWidth = Math.round(L.width * (window.innerWidth < 700 ? 1 : 0.8)) + "px";
    svg.setAttribute("aria-label", `${sc.title}: memory graph of ${sc.nodes.length} facts and the new fact "${sc.trigger}"`);
    const gEdges = el("g", {}, svg);
    const gNodes = el("g", {}, svg);
    const refs = { edges: {}, nodes: {} };
    const cx = (id) => L.x[id] + NODE_W / 2;

    function arrowAt(x2, y2, cls) { return el("path", { d: `M${x2 - 5} ${y2 - 8} L${x2} ${y2} L${x2 + 5} ${y2 - 8} Z`, class: "g-arrow " + cls }, gEdges); }

    sc.nodes.forEach((n) => {
      n.parents.forEach((p) => {
        const x1 = cx(p), y1 = L.y[p] + L.h(p), x2 = cx(n.id), y2 = L.y[n.id];
        const path = el("path", { d: `M${x1} ${y1} C ${x1} ${y1 + GAP_Y * 0.55}, ${x2} ${y2 - GAP_Y * 0.55}, ${x2} ${y2 - 1}`, class: "g-edge" }, gEdges);
        refs.edges[p + ">" + n.id] = [path, arrowAt(x2, y2 - 1, "")];
      });
    });
    sc.assoc.forEach(([a, b]) => {
      const left = L.x[a] < L.x[b] ? a : b, right = left === a ? b : a;
      const yy = Math.min(L.y[a] + L.h(a), L.y[b] + L.h(b)) / 2 + Math.max(L.y[a], L.y[b]) / 2;
      el("path", { d: `M${L.x[left] + NODE_W} ${yy} L${L.x[right]} ${yy}`, class: "g-edge assoc" }, gEdges);
    });
    // trigger -> directly updated facts
    const direct = new Set(view.trigEdges.concat(sc.steps.filter((s) => s.kind === "revise" && s.depth === 0).map((s) => s.id)));
    direct.forEach((id) => {
      const x1 = cx("new"), y1 = L.y.new + L.h("new"), x2 = cx(id), y2 = L.y[id];
      const path = el("path", { d: `M${x1} ${y1} C ${x1} ${y1 + GAP_Y * 0.55}, ${x2} ${y2 - GAP_Y * 0.55}, ${x2} ${y2 - 1}`, class: "g-edge trigger" }, gEdges);
      const arr = arrowAt(x2, y2 - 1, "trigger");
      const lbl = el("text", { x: (x1 + x2) / 2 + 8, y: (y1 + y2) / 2 + 4, class: "g-edge-label" }, gEdges);
      lbl.textContent = "updates";
      refs.edges["new>" + id] = [path, arr, lbl];
    });

    function node(id, text, cls) {
      const g = el("g", { class: "g-node " + cls, transform: `translate(${L.x[id]},${L.y[id]})` }, gNodes);
      el("rect", { class: "ring", x: -4, y: -4, width: NODE_W + 8, height: L.h(id) + 8, rx: 9 }, g);
      el("rect", { class: "box", width: NODE_W, height: L.h(id), rx: 6 }, g);
      const label = el("text", { class: "label", x: 12, y: 19 }, g);
      const via = el("text", { class: "via", x: NODE_W - 12, y: 19, "text-anchor": "end" }, g);
      const body = el("text", { class: "body", x: 12, y: 38 }, g);
      refs.nodes[id] = { g, label, via, body };
      return g;
    }
    node("new", sc.trigger, "s-new");
    sc.nodes.forEach((n) => node(n.id, n.text, "s-active"));
    return refs;
  }

  function setBody(t, text) {
    t.textContent = "";
    wrap(text, CHARS).forEach((line, i) => {
      const s = document.createElementNS(NS, "tspan");
      s.setAttribute("x", 12);
      s.setAttribute("dy", i ? LINE : 0);
      s.textContent = line;
      t.appendChild(s);
    });
  }

  function paint(sc, refs, view) {
    // nodes
    const tr = refs.nodes.new;
    tr.g.setAttribute("class", "g-node s-new" + (view.trig ? "" : " dim"));
    tr.label.textContent = view.trig ? LABEL.new : "NEXT WRITE";
    setBody(tr.body, sc.trigger);
    sc.nodes.forEach((n) => {
      const v = view.nodes[n.id], r = refs.nodes[n.id];
      const cur = view.current.includes(n.id);
      r.g.setAttribute("class", `g-node s-${v.st}${cur ? " current" : ""}`);
      r.label.textContent = LABEL[v.st];
      r.via.textContent = v.via ? "VIA " + v.via : "";
      setBody(r.body, v.text);
    });
    // edges
    Object.entries(refs.edges).forEach(([key, parts]) => {
      const [from, to] = key.split(">");
      const isTrig = from === "new";
      const shown = isTrig ? view.trigEdges.includes(to) : true;
      const hot = view.hot && view.hot[0] === from && view.hot[1] === to;
      parts.forEach((p) => {
        const base = p.tagName === "text" ? "g-edge-label" : p.tagName === "path" && p.classList.contains("g-arrow") ? "g-arrow" : "g-edge";
        let cls = base + (isTrig ? " trigger" : "");
        if (hot && !isTrig) cls += " hot";
        if (!shown) cls += " hidden";
        p.setAttribute("class", cls);
      });
    });
  }

  function gauge(title, value, zones) {
    const on = (z) => value != null && value >= z[0] && (value < z[1] || (z[1] === 1 && value <= 1));
    const segs = zones.map((z) => `<div class="gauge-seg${on(z) ? " on" : ""}" style="width:${(z[1] - z[0]) * 100}%"></div>`).join("");
    const names = zones.map((z) => `<div class="gauge-zone${on(z) ? " on" : ""}" style="width:${(z[1] - z[0]) * 100}%">${esc(z[2])}</div>`).join("");
    const mark = value == null ? "" : `<div class="gauge-mark" style="left:${Math.min(99.5, Math.max(0.5, value * 100))}%"></div>`;
    return `<div class="gauge"><div class="gauge-label"><span>${esc(title)}</span><span>${pct(value)}</span></div><div class="gauge-track">${segs}${mark}</div><div class="gauge-names">${names}</div></div>`;
  }
  const gUnaffected = (j) => gauge("Jev · P(unaffected)", j && j.p_unaffected, [[0, 0.35, "affected"], [0.35, 0.9, "unsure → LLM"], [0.9, 1, "stop"]]);
  const gKnown = (j) => gauge("Jev · P(new value follows)", j && j.p_known, [[0, 0.5, "unknown → stale"], [0.5, 1, "follows → LLM rewrite"]]);
  const viaTag = (via) => via === "jev" ? `<span class="tag tag-jev">decided by Jev</span><span class="tag tag-llm">no LLM call</span>` : `<span class="tag tag-llm">LLM call</span>`;

  function panelHTML(sc, steps, k) {
    const s = steps[k];
    const map = byId(sc);
    const text = (id) => esc(map[id].text);
    if (s.kind === "before") {
      const links = sc.nodes.reduce((a, n) => a + n.parents.length, 0);
      return `<h3>${esc(sc.title)}: memory before the change</h3>
        <div class="meta-row"><span class="tag tag-llm">${esc(sc.domain)}</span></div>
        <p>${esc(sc.blurb)}</p>
        <p>${sc.nodes.length} facts, ${links} derived-from link${links === 1 ? "" : "s"}${sc.assoc.length ? `, ${sc.assoc.length} associated link` : ""}. Arrows point from a fact to the facts worked out from it.</p>`;
    }
    if (s.kind === "arrive") {
      return `<h3>A new fact arrives</h3><div class="fact">${esc(sc.trigger)}</div>
        <p>Saved with <span class="mono">add_memory</span>. Before it's stored, GEM checks whether it changes anything already in memory.</p>`;
    }
    if (s.kind === "scan") {
      const rows = s.checked.map((c) => {
        const r = c.route === "skip" ? `no conflict · Jev ${pct(c.p_no_conflict)}` : c.route === "covered" ? "fully replaced · Jev" : "unsure · to LLM";
        return `<li><span>${text(c.id)}</span><span>${r}</span></li>`;
      }).join("");
      const settled = s.checked.filter((c) => c.route !== "llm").length;
      return `<h3>Conflict scan: ${s.checked.length} similar facts, one Jev call</h3>
        <ul class="checks">${rows}</ul>
        <p>Jev answers with probabilities. It settled ${settled} of ${s.checked.length} on its own; the rest wait for the LLM. Facts the cascade reaches first skip that LLM call.</p>`;
    }
    if (s.kind === "revise" && s.depth === 0) {
      const old = text(s.id);
      const j = s.jev || {};
      const why = s.via === "jev"
        ? `Jev judged that the new fact fully replaces this one (P = ${pct(j.p_covered)}), so it was replaced without an LLM call.`
        : `Jev saw a conflict (most likely ${esc((j.top || "").toLowerCase())}, P = ${pct(j.p_top)}) but wasn't sure of the details, so the LLM wrote the update.`;
      const after = s.status === "ACTIVE" ? `<s>${old}</s><span class="arrow">→</span>${esc(s.content)}` : old;
      return `<h3>Direct conflict: ${s.status === "ACTIVE" ? "fact updated" : "fact " + s.status.toLowerCase()}</h3>
        <div class="fact">${after}</div><div class="meta-row">${viaTag(s.via)}</div><p>${why}</p>
        <p>The cascade now starts from this fact.</p>`;
    }
    if (s.kind === "revise") {
      const p = parentOf(sc, s.id, new Set(steps.slice(2, k).filter((x) => x.id).map((x) => x.id)));
      const from = p ? `<p>Derived from “${text(p)}”, which just changed.</p>` : "";
      const j = s.jev;
      let head, why, fact;
      if (s.status === "ACTIVE") {
        head = `Step ${s.depth}: rewritten`;
        fact = `<s>${text(s.id)}</s><span class="arrow">→</span>${esc(s.content)}`;
        why = `Affected, and Jev expects the new value to follow from the change, so the LLM wrote it.`;
      } else if (s.via === "jev") {
        head = `Step ${s.depth}: marked stale`;
        fact = text(s.id);
        why = `Affected, and the new value can't be worked out from the change, so the fact is marked stale and flagged for review. No LLM call.`;
      } else {
        head = `Step ${s.depth}: marked stale`;
        fact = text(s.id);
        why = `Jev expected the new value to follow, so the LLM was asked to rewrite it. The LLM judged it affected but couldn't state the new value, so it was marked stale.`;
      }
      return `<h3>${head}</h3><div class="fact">${fact}</div>${from}${j ? gUnaffected(j) + gKnown(j) : ""}
        <div class="meta-row">${viaTag(s.via)}</div><p>${why}</p>`;
    }
    if (s.kind === "stop") {
      const p = parentOf(sc, s.id, new Set(steps.slice(2, k).filter((x) => x.id).map((x) => x.id)));
      return `<h3>Step ${s.depth}: unaffected, the cascade stops here</h3><div class="fact">${text(s.id)}</div>
        ${p ? `<p>Derived from “${text(p)}”.</p>` : ""}${s.jev ? gUnaffected(s.jev) : ""}
        <div class="meta-row">${viaTag(s.via)}</div>
        <p>${s.via === "jev" ? "Jev is confident the change doesn't touch the property this fact depends on, so it stays as it is and nothing below it is checked." : "The LLM judged that the change doesn't affect it."}</p>`;
    }
    // done
    let upd = 0, stale = 0, kept = 0;
    sc.nodes.forEach((n, i) => {
      const g = sc.gem[i];
      if (g.status !== "ACTIVE" || g.review) stale++;
      else if (g.content !== n.text) upd++;
      else kept++;
    });
    const wrong = sc.nodes.filter((n, i) => gemChanged(sc, i) && sc.flat[i].status === "ACTIVE" && sc.flat[i].content === n.text).length;
    const skipped = s.already.length;
    return `<h3>Result</h3>
      <div class="sumgrid"><div><b>${upd}</b><span>updated</span></div><div><b>${stale}</b><span>stale</span></div><div><b>${kept}</b><span>unchanged</span></div></div>
      <p>This write cost <b>${sc.calls.jev}</b> Jev call${sc.calls.jev === 1 ? "" : "s"} and <b>${sc.calls.llm}</b> LLM call${sc.calls.llm === 1 ? "" : "s"}.${skipped ? ` ${skipped} fact${skipped === 1 ? "" : "s"} the scan had sent to the LLM ${skipped === 1 ? "was" : "were"} already handled by the cascade, so ${skipped === 1 ? "that call was" : "those calls were"} skipped.` : ""}</p>
      <p>${wrong ? `Flat memory, given the same write, still serves <b>${wrong}</b> of these facts as true.` : "On this example flat memory also caught every affected fact."} <a href="#" data-goto-flat>Show flat memory</a></p>`;
  }

  function flatPanelHTML(sc) {
    const rows = [];
    let wrong = 0;
    sc.nodes.forEach((n, i) => {
      const f = sc.flat[i];
      if (gemChanged(sc, i) && f.status === "ACTIVE" && f.content === n.text) {
        wrong++;
        const g = sc.gem[i];
        const gemSays = g.status !== "ACTIVE" ? "GEM: stale" : "GEM: updated";
        rows.push(`<li><span>${esc(n.text)}</span><span>${gemSays}</span></li>`);
      }
    });
    return `<h3>Flat memory after the same write</h3>
      <p>Flat memory compares the new fact with similar facts and updates the ones that directly conflict. It doesn't follow derived-from links, so facts built on the old value aren't checked.</p>
      ${wrong ? `<p><b>${wrong}</b> fact${wrong === 1 ? " is" : "s are"} still served as true:</p><ul class="checks">${rows.join("")}</ul>`
        : `<p>Here flat memory caught every affected fact. They all mention the changed term, so a similarity search finds them. The difference shows when dependents share no words with the change.</p>`}
      <p><a href="#" data-goto-gem>Back to the GEM run</a></p>`;
  }

  let refs = null, layoutCache = null, steps = null;

  function loadScenario(i) {
    stop();
    state.si = i;
    state.step = 0;
    const sc = state.data.scenarios[i];
    steps = buildSteps(sc);
    layoutCache = layout(sc);
    document.querySelectorAll(".scenario-tab").forEach((t, j) => t.setAttribute("aria-selected", String(j === i)));
    render();
  }

  function render() {
    const sc = state.data.scenarios[state.si];
    const flat = state.view === "flat";
    const view = flat ? flatState(sc) : stateAt(sc, steps, state.step);
    refs = drawGraph(sc, layoutCache, view);
    paint(sc, refs, view);
    const links = sc.nodes.reduce((a, n) => a + n.parents.length, 0);
    $("#canvas-note").textContent = `${sc.nodes.length} facts · ${links} derived-from links`;
    $("#step-body").innerHTML = flat ? flatPanelHTML(sc) : panelHTML(sc, steps, state.step);
    $("#step-count").textContent = flat ? "Flat memory · end state" : `Step ${state.step + 1} of ${steps.length}`;
    $("#progress").style.width = flat ? "100%" : `${(state.step / (steps.length - 1)) * 100}%`;
    $("#btn-back").disabled = flat || state.step === 0;
    $("#btn-next").disabled = flat || state.step === steps.length - 1;
    $("#btn-play").disabled = flat;
    document.querySelectorAll(".seg button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.view === state.view)));
    const fl = $("[data-goto-flat]");
    if (fl) fl.addEventListener("click", (e) => { e.preventDefault(); setView("flat"); });
    const gl = $("[data-goto-gem]");
    if (gl) gl.addEventListener("click", (e) => { e.preventDefault(); setView("gem"); });
  }

  function go(k) { state.step = Math.max(0, Math.min(steps.length - 1, k)); render(); }
  function setView(v) { stop(); state.view = v; render(); }
  function stop() {
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
    $("#btn-play").classList.remove("playing");
    $("#btn-play").setAttribute("aria-label", "Play");
  }
  function play() {
    if (state.timer) return stop();
    if (state.step === steps.length - 1) go(0);
    $("#btn-play").classList.add("playing");
    $("#btn-play").setAttribute("aria-label", "Pause");
    state.timer = setInterval(() => {
      if (state.step >= steps.length - 1) return stop();
      go(state.step + 1);
    }, reduceMotion ? 2600 : 1900);
  }

  function initExplorer(data) {
    state.data = data;
    const tabs = $(".scenario-tabs");
    data.scenarios.forEach((sc, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "scenario-tab";
      b.setAttribute("role", "tab");
      b.innerHTML = `<span class="st-title">${esc(sc.title)}</span><span class="st-domain">${esc(sc.domain)}</span>`;
      b.addEventListener("click", () => { state.view = "gem"; loadScenario(i); });
      tabs.appendChild(b);
    });
    $("#btn-next").addEventListener("click", () => { stop(); go(state.step + 1); });
    $("#btn-back").addEventListener("click", () => { stop(); go(state.step - 1); });
    $("#btn-reset").addEventListener("click", () => { stop(); state.view = "gem"; go(0); });
    $("#btn-play").addEventListener("click", play);
    document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
    $(".explorer").addEventListener("keydown", (e) => {
      if (state.view !== "gem" || e.target.closest("a")) return;
      if (e.key === "ArrowRight") { stop(); go(state.step + 1); e.preventDefault(); }
      if (e.key === "ArrowLeft") { stop(); go(state.step - 1); e.preventDefault(); }
    });
    $("#panel-foot").textContent = `Recorded ${data.generated} · LLM ${data.llm} · Jev decider`;
    $("#data-note").innerHTML = `Recorded from real runs by <span class="mono">docs/build_scenarios.py</span>; flat memory is the same engine with the cascade turned off. Model decisions can vary between runs, and re-running the script re-records them.`;
    // deep links: ?ex=<key>&step=<n>&view=flat
    const q = new URLSearchParams(location.search);
    const si = Math.max(0, data.scenarios.findIndex((s) => s.key === q.get("ex")));
    if (q.get("view") === "flat") state.view = "flat";
    loadScenario(si);
    if (q.get("step")) go(parseInt(q.get("step"), 10) - 1 || 0);
  }

  fetch("data/scenarios.json")
    .then((r) => r.json())
    .then(initExplorer)
    .catch(() => {
      $("#step-body").innerHTML = `<p>The recorded runs couldn't be loaded. Serve this folder over HTTP (for example <span class="mono">python -m http.server</span> in <span class="mono">docs/</span>).</p>`;
    });

  // ------------------------------------------------------------------ results charts
  const LIFT = [
    ["Deep chains", 42], ["Off-domain distractors", 60], ["Messy phrasing", 67], ["Multi-hop chains", 68],
    ["Wide fan-out", 75], ["Noise amid signal", 80], ["Several changes at once", 80],
  ];

  function dumbbell() {
    const host = $("#dumbbell");
    const W = 600, left = 176, right = 540, rowH = 34, top = 26;
    const H = top + LIFT.length * rowH + 6;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Flat memory versus GEM by scenario type; GEM scores 100% in every type shown, flat memory between 42% and 80%." }, host);
    const sx = (v) => left + (v / 100) * (right - left);
    [0, 25, 50, 75, 100].forEach((t) => {
      el("line", { x1: sx(t), x2: sx(t), y1: top - 6, y2: H - 4, class: "viz-grid" }, svg);
      const tx = el("text", { x: sx(t), y: 12, "text-anchor": "middle", class: "viz-axis" }, svg);
      tx.textContent = t + "%";
    });
    LIFT.forEach(([name, flat], i) => {
      const y = top + i * rowH + rowH / 2;
      const g = el("g", { class: "viz-row" }, svg);
      el("rect", { x: 0, y: y - rowH / 2, width: W, height: rowH, class: "viz-hit" }, g);
      const t = el("text", { x: left - 14, y: y + 4, "text-anchor": "end", class: "viz-cat" }, g);
      t.textContent = name;
      el("line", { x1: sx(flat), x2: sx(100), y1: y, y2: y, class: "viz-link" }, g);
      el("circle", { cx: sx(flat), cy: y, r: 6, class: "viz-dot-flat" }, g);
      el("circle", { cx: sx(100), cy: y, r: 6, class: "viz-dot-gem" }, g);
      const v = el("text", { x: sx(flat) - 11, y: y + 4, "text-anchor": "end", class: "viz-val" }, g);
      v.textContent = flat + "%";
      const d = el("text", { x: W - 4, y: y + 4, "text-anchor": "end", class: "viz-val" }, g);
      d.textContent = "+" + (100 - flat);
      g.addEventListener("mousemove", (e) => showTip(e, `<b>${esc(name)}</b><br>Flat memory ${flat}% · GEM 100% · +${100 - flat} points`));
      g.addEventListener("mouseleave", hideTip);
    });
    const tb = $("#dumbbell-table tbody");
    tb.innerHTML = LIFT.map(([n, f]) => `<tr><td>${esc(n)}</td><td>${f}%</td><td>100%</td><td>+${100 - f}</td></tr>`).join("");
  }

  function callbars() {
    const host = $("#callbars");
    const rows = [
      ["Before: the LLM makes every decision", 58, true],
      ["Skip facts the cascade already reached", 37, true],
      ["Jev decides, LLM only when needed", 12, false],
    ];
    const W = 600, left = 0, right = 548, barH = 16, rowH = 54, top = 4;
    const H = top + rows.length * rowH;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "LLM calls on the 12-scenario suite: 58 before, 37 after skipping facts the cascade already reached, 12 with the Jev decider." }, host);
    const sx = (v) => left + (v / 60) * (right - left);
    rows.forEach(([name, v, muted], i) => {
      const y = top + i * rowH;
      const g = el("g", { class: "viz-row" }, svg);
      el("rect", { x: 0, y: y, width: W, height: rowH - 6, class: "viz-hit" }, g);
      const t = el("text", { x: 0, y: y + 15, class: "viz-cat" }, g);
      t.textContent = name;
      const w = sx(v) - left, by = y + 24, r = 4;
      el("path", { d: `M${left} ${by} H${left + w - r} Q${left + w} ${by} ${left + w} ${by + r} V${by + barH - r} Q${left + w} ${by + barH} ${left + w - r} ${by + barH} H${left} Z`, class: "viz-bar" + (muted ? " muted" : "") }, g);
      const vt = el("text", { x: left + w + 8, y: by + 12.5, class: "viz-val" }, g);
      vt.textContent = v + " calls";
      g.addEventListener("mousemove", (e) => showTip(e, `<b>${esc(name)}</b><br>${v} LLM calls · 37/37 correct`));
      g.addEventListener("mouseleave", hideTip);
    });
  }

  dumbbell();
  callbars();
})();
