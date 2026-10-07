/* GEM site: the pinned story graph, the playground, and page chrome. Everything on the graphs replays
   recorded GEM runs from data/scenarios.json (made by docs/build_scenarios.py). */
(async function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const RM = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  // wide: text column on the left, graph in the space to its right. Below BP the graph sits on top
  // and the text card below it (the two would collide otherwise). Matches the CSS breakpoint.
  const BP = 1180;
  const wide = window.innerWidth > BP, compact = !wide;
  // phones: the graph keeps the top of the screen to itself and the step text scrolls up beneath it
  const PHONE = 760, phone = window.innerWidth <= PHONE;
  const W0 = window.innerWidth;
  const textRight = W0 * 0.09 + Math.min(460, W0 * 0.34);       // .step padding + .card width
  const OX = wide ? (textRight + 56 - 28) / 2 : 0;                // centre the graph in what's left
  const mode = () => (window.innerWidth > BP ? "wide" : window.innerWidth <= PHONE ? "phone" : "compact");
  let lastMode = mode();
  window.addEventListener("resize", () => {                       // the story is laid out for one mode
    if (mode() !== lastMode) { lastMode = mode(); location.reload(); }
  });
  document.documentElement.classList.add("js");

  // ------------------------------------------------------------------ film grain (generated once)
  (function grain() {
    const c = $("#grain"), x = c.getContext("2d");
    c.width = c.height = 256;
    const img = x.createImageData(256, 256);
    for (let i = 0; i < img.data.length; i += 4) { const v = Math.random() * 255; img.data[i] = img.data[i + 1] = img.data[i + 2] = v; img.data[i + 3] = 255; }
    x.putImageData(img, 0, 0);
    c.style.backgroundImage = `url(${c.toDataURL()})`;
    c.width = c.height = 1;
  })();

  // ------------------------------------------------------------------ chrome: solid bar after the hero, active nav link
  const bar = $("#top-bar");
  const onScroll = () => bar.classList.toggle("solid", window.scrollY > window.innerHeight * 0.6);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
  const navLinks = [...document.querySelectorAll(".top nav a[href^='#']")];
  const sections = navLinks.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  const navIO = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) navLinks.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id));
  }), { rootMargin: "-45% 0px -50% 0px" });
  sections.forEach((s) => navIO.observe(s));

  // sections after the story ease in as they arrive
  const revealIO = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    e.target.classList.add("in");
    revealIO.unobserve(e.target);
  }), { rootMargin: "0px 0px -8% 0px" });
  document.querySelectorAll(".after .head, .picks li, .play-main, .numbers > div, .method, .install > div, .limits > div, footer .big, footer ul")
    .forEach((n) => {
      n.classList.add("rv");
      n.style.setProperty("--rd", ([...n.parentElement.children].indexOf(n) % 4) * 70 + "ms");   // siblings stagger
      revealIO.observe(n);
    });

  // result numbers count up once, from the value in the markup (so no-JS readers see the real ones)
  const countIO = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    countIO.unobserve(e.target);
    const node = e.target.firstChild, final = node.textContent.trim(), to = parseFloat(final);
    if (RM || !isFinite(to)) return;
    const dec = (final.split(".")[1] || "").length, t0 = performance.now(), dur = 1100;
    const tick = (now) => {
      const k = Math.min(1, (now - t0) / dur), v = to * (1 - Math.pow(1 - k, 3));
      node.textContent = k < 1 ? v.toFixed(dec) : final;
      if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }), { threshold: 0.6 });
  document.querySelectorAll(".numbers b").forEach((b) => countIO.observe(b));

  // ------------------------------------------------------------------ live background
  // Neurons behind the page: organic cells with branching dendrites, meshed to their neighbours by axons.
  // They float, each on its own slow path, so the page is alive but calm to read over. Now and then one
  // fires and the change hops cell to cell: blue where the new value follows, amber where it goes stale,
  // green where a cell is unaffected and passes nothing on. Everything stays behind content; cells near
  // the pointer brighten, and a click fires the nearest one with a longer reach.
  const fieldHooks = {};                   // the story graph reports where it is, so the field keeps clear of it
  (function field() {
    const c = $("#field"), x = c.getContext("2d");
    const COL = { write: "236,232,223", rw: "134,168,255", stale: "244,180,67", keep: "107,213,154" };
    const KEEP = ".card, .callpanel, .after .head, .picks, .play-bar, .play-stage, .numbers > div, .method, .install .code, .tools, .tools-note, .limits > div, footer .big, footer ul, .legal";
    const PAR = 0.12, SPEED = 170, DRIFT = 7;    // field scroll vs page; signal px/s; how far a cell floats
    const rnd = (a, b) => a + Math.random() * (b - a);
    let W = 0, H = 0, FH = 0, dpr = 1, mouse = null, last = 0, now = 0, nextFire = 3, storyEnd = Infinity;
    let cells = [], axons = [], signals = [], lit = [], keepOut = [], blocks = [];
    const ripples = [];

    const measureKeepOut = () => {
      keepOut = [...document.querySelectorAll(KEEP)].map((el) => { const r = el.getBoundingClientRect(); return [r.left, r.top + window.scrollY, r.width, r.height]; })
        .filter((r) => r[2] > 0 && r[3] > 0);
    };
    // a dendrite: a wandering line that sometimes forks, drawn around the cell's own origin
    function grow(path, px, py, ang, steps, depth) {
      path.moveTo(px, py);
      for (let i = 0; i < steps; i++) {
        ang += rnd(-0.32, 0.32);
        px += Math.cos(ang) * 5; py += Math.sin(ang) * 5;
        path.lineTo(px, py);
        if (depth < 2 && i > 3 && Math.random() < 0.07) {
          grow(path, px, py, ang + (Math.random() < 0.5 ? -1 : 1) * rnd(0.4, 0.9), Math.floor(steps * 0.55), depth + 1);
          path.moveTo(px, py);
        }
      }
    }
    function build() {
      // the field is as tall as the screen plus the page's slow scroll, so it never wraps
      FH = H + Math.max(0, document.documentElement.scrollHeight - H) * PAR + 60;
      const n = Math.round((W * FH) / (W < 700 ? 17000 : 24000)), gap = W < 700 ? 46 : 64;
      cells = [];
      for (let tries = 0; cells.length < n && tries < n * 30; tries++) {        // keep cells apart
        const cx = rnd(10, W - 10), cy = rnd(0, FH);
        if (cells.every((q) => Math.hypot(q.x - cx, q.y - cy) > gap)) {
          const q = { x: cx, y: cy, px: cx, py: cy, r: rnd(2, 3.4), links: [], flash: null, dend: new Path2D(),
            f: [rnd(0.02, 0.045), rnd(0.025, 0.05), rnd(0.055, 0.085), rnd(0.05, 0.08)], ph: [rnd(0, 6.3), rnd(0, 6.3), rnd(0, 6.3), rnd(0, 6.3)] };
          const k = Math.round(rnd(3, 6)), base = rnd(0, Math.PI * 2);
          for (let i = 0; i < k; i++) grow(q.dend, 0, 0, base + (i / k) * Math.PI * 2 + rnd(-0.3, 0.3), Math.round(rnd(4, 10)), 0);
          cells.push(q);
        }
      }
      // axons: each cell to its three to five nearest neighbours; the bend is stored, the ends follow the cells
      axons = [];
      const seen = new Set();
      cells.forEach((a, i) => {
        cells.map((b, j) => [j, Math.hypot(a.x - b.x, a.y - b.y)]).filter(([j, d]) => j !== i && d < 190).sort((u, v) => u[1] - v[1]).slice(0, Math.round(rnd(3, 5)))
          .forEach(([j, d]) => {
            const key = Math.min(i, j) + "-" + Math.max(i, j);
            if (seen.has(key)) return;
            seen.add(key);
            const ax = { i, j, len: d, w1: rnd(-20, 20), w2: rnd(-20, 20) };
            axons.push(ax); a.links.push(ax); cells[j].links.push(ax);
          });
      });
      signals = []; lit = [];
    }
    function size() {
      dpr = Math.min(1.5, window.devicePixelRatio || 1);
      W = window.innerWidth; H = window.innerHeight;
      c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
      const after = document.querySelector(".after");
      storyEnd = after ? after.offsetTop : Infinity;
      measureKeepOut();
      build();
    }
    size();
    setInterval(measureKeepOut, 2000);      // reveals and late fonts shift blocks a little
    window.addEventListener("resize", size);
    window.addEventListener("pointermove", (e) => { mouse = [e.clientX, e.clientY]; }, { passive: true });
    document.addEventListener("pointerleave", () => { mouse = null; });

    const shift = () => window.scrollY * PAR;
    const float = (q) => {                  // two slow sine pairs per axis: a gentle wander that never repeats quickly
      q.px = q.x + DRIFT * (Math.sin(now * q.f[0] + q.ph[0]) + 0.45 * Math.sin(now * q.f[2] + q.ph[2]));
      q.py = q.y + DRIFT * (Math.cos(now * q.f[1] + q.ph[1]) + 0.45 * Math.cos(now * q.f[3] + q.ph[3]));
    };
    const onScreen = (q) => { const y = q.py - shift(); return q.px > 20 && q.px < W - 20 && y > 80 && y < H - 30; };
    const open = (q) => {                   // on screen and clear of content
      const y = q.py - shift();
      return onScreen(q) && !blocks.some(([l, t, w, h]) => q.px > l - 30 && q.px < l + w + 30 && y > t - 30 && y < t + h + 30);
    };
    const curve = (ax) => {                 // the axon's current bezier, from where its two cells are now
      const a = cells[ax.i], b = cells[ax.j], dx = b.px - a.px, dy = b.py - a.py, d = Math.hypot(dx, dy) || 1, nx = -dy / d, ny = dx / d;
      return [a.px, a.py, a.px + dx / 3 + nx * ax.w1, a.py + dy / 3 + ny * ax.w1, a.px + (2 * dx) / 3 + nx * ax.w2, a.py + (2 * dy) / 3 + ny * ax.w2, b.px, b.py];
    };
    const bez = (p, t) => {
      const u = 1 - t;
      return [u * u * u * p[0] + 3 * u * u * t * p[2] + 3 * u * t * t * p[4] + t * t * t * p[6], u * u * u * p[1] + 3 * u * u * t * p[3] + 3 * u * t * t * p[5] + t * t * t * p[7]];
    };

    // how many hops a cascade runs through the mesh: the background fires rarely and briefly, a click runs
    // long. Each cell fires once per cascade (a shared visited set, like GEM's cascade frontier).
    const AUTO = { max: 4, root: 2, branch: 2, carry: 0.7, keep: 0.14 }, CLICK = { max: 10, root: 99, branch: 3, carry: 0.9, keep: 0.06, click: true };
    function fire(i, state, depth, reach = AUTO, seen = new Set([i])) {   // a cell lights, then (unless unaffected) passes the change on
      cells[i].flash = { t: now, state };
      if (state === "keep" || depth >= reach.max) return;
      cells[i].links.filter((ax) => !seen.has(ax.i === i ? ax.j : ax.i)).sort(() => Math.random() - 0.5).slice(0, depth === 0 ? reach.root : reach.branch).forEach((ax) => {
        if (depth > 0 && Math.random() > reach.carry) return;
        const to = ax.i === i ? ax.j : ax.i, r = Math.random();
        let next = r < reach.keep ? "keep" : r < reach.keep + 0.27 ? "rw" : "stale";
        if (reach.click && depth === 0 && next === "keep") next = "stale";   // a clicked cascade always gets going
        seen.add(to);
        signals.push({ ax, fwd: ax.i === i, t0: now, dur: ax.len / SPEED, to, state: next, depth: depth + 1, reach, seen, k: 0 });
      });
    }
    function fireSomewhere() {
      const choices = cells.map((q, i) => i).filter((i) => open(cells[i]) && !cells[i].flash);
      if (choices.length) fire(choices[Math.floor(Math.random() * choices.length)], "write", 0);
    }
    // a click almost anywhere (not on a control) fires the nearest cell, preferring one in open space
    window.addEventListener("click", (e) => {
      if (RM || e.target.closest("a, button, input, select, textarea, pre, code, .picks, .play-stage, .play-bar")) return;
      if (String(window.getSelection && window.getSelection()).length) return;
      let best = -1, bd = Infinity;
      cells.forEach((q, i) => {
        if (!onScreen(q)) return;
        const dist = Math.hypot(q.px - e.clientX, q.py - shift() - e.clientY), rank = dist + (open(q) ? 0 : 120);   // open cells win close calls
        if (dist < 200 && rank < bd) { bd = rank; best = i; }
      });
      ripples.push({ x: e.clientX, y: e.clientY, t: now });
      if (best >= 0) fire(best, "write", 0, CLICK);
    });

    function draw() {
      x.setTransform(dpr, 0, 0, dpr, 0, 0);
      x.clearRect(0, 0, W, H);
      const dim = window.scrollY + H * 0.5 < storyEnd ? 0.6 : 1;   // quieter behind the story
      const stage = dim < 1 && fieldHooks.stageBox ? fieldHooks.stageBox() : null;
      blocks = keepOut.map(([l, t, w, h]) => [l, t - window.scrollY, w, h]).filter(([, t, , h]) => t < H + 40 && t + h > -40);
      if (stage) blocks.push(stage);
      const sh = shift(), visible = (q) => q.py - sh > -80 && q.py - sh < H + 80;
      cells.forEach(float);
      if (!RM && now > nextFire) { fireSomewhere(); nextFire = now + rnd(6, 10); }

      // tissue: each cell's dendrites around its floating centre, then the axons between them
      x.lineCap = "round";
      x.lineWidth = 0.7; x.strokeStyle = `rgba(170,180,210,${(0.065 * dim).toFixed(3)})`;
      for (const q of cells) {
        if (!visible(q)) continue;
        x.setTransform(dpr, 0, 0, dpr, q.px * dpr, (q.py - sh) * dpr);
        x.stroke(q.dend);
      }
      x.setTransform(dpr, 0, 0, dpr, 0, 0);
      x.lineWidth = 1; x.strokeStyle = `rgba(170,180,210,${(0.06 * dim).toFixed(3)})`;
      x.beginPath();
      for (const ax of axons) {
        if (!visible(cells[ax.i]) && !visible(cells[ax.j])) continue;
        const p = curve(ax);
        x.moveTo(p[0], p[1] - sh); x.bezierCurveTo(p[2], p[3] - sh, p[4], p[5] - sh, p[6], p[7] - sh);
      }
      x.stroke();

      // signals running along axons; arrivals fire after the filter (signals pushed while filtering
      // would never reach the new array, and cascades would end at one hop)
      const arrived = [];
      signals = signals.filter((sg) => {
        sg.k = (now - sg.t0) / sg.dur;
        if (sg.k < 1) return true;
        arrived.push(sg);
        return false;
      });
      arrived.forEach((sg) => {
        lit.push({ ax: sg.ax, col: COL[sg.state === "keep" ? "keep" : "rw"], t0: now });   // the axon stays lit a moment
        fire(sg.to, sg.state, sg.depth, sg.reach, sg.seen);
      });
      // conduction: an axon lights up from the firing cell toward the next, brightest where the change is now
      const span = (p, a, b, alpha, width, col) => {
        if (b - a < 0.002) return;
        x.strokeStyle = `rgba(${col},${(alpha * dim).toFixed(3)})`; x.lineWidth = width;
        x.beginPath();
        for (let s2 = 0; s2 <= 14; s2++) {
          const [qx, qy] = bez(p, a + ((b - a) * s2) / 14);
          if (s2 === 0) x.moveTo(qx, qy - sh); else x.lineTo(qx, qy - sh);
        }
        x.stroke();
      };
      lit = lit.filter((l) => {
        const age = (now - l.t0) / 1.8;
        if (age >= 1) return false;
        span(curve(l.ax), 0, 1, 0.26 * (1 - age), 1.1, l.col);
        return true;
      });
      for (const sg of signals) {
        const p = curve(sg.ax), t = sg.fwd ? sg.k : 1 - sg.k, col = COL[sg.state === "keep" ? "keep" : "rw"];
        if (sg.fwd) span(p, 0, t, 0.26, 1.1, col); else span(p, t, 1, 0.26, 1.1, col);       // the stretch already reached
        span(p, Math.max(0, t - 0.06), Math.min(1, t + 0.06), 0.42, 1.4, col);               // a soft brightening at the front
      }

      // cell bodies: a faint glow when they fire, brighter near the pointer
      for (const q of cells) {
        if (!visible(q)) continue;
        const y = q.py - sh;
        let col = "236,232,223", a = 0.24, glow = 0;
        if (mouse) a += Math.max(0, 1 - Math.hypot(q.px - mouse[0], y - mouse[1]) / 240) * 0.45;
        if (q.flash) {
          const k = Math.max(0, 1 - Math.max(0, now - q.flash.t - 0.5) / 3.4);
          if (k <= 0) q.flash = null; else { col = COL[q.flash.state]; glow = k; a = Math.max(a, 0.9 * k); }
        }
        if (glow) {
          const R = 9 + 11 * glow, g2 = x.createRadialGradient(q.px, y, 0, q.px, y, R);
          g2.addColorStop(0, `rgba(${col},${(0.13 * glow * dim).toFixed(3)})`); g2.addColorStop(1, `rgba(${col},0)`);
          x.fillStyle = g2; x.fillRect(q.px - R, y - R, R * 2, R * 2);
        }
        x.fillStyle = `rgba(${col},${(a * dim).toFixed(3)})`;
        x.beginPath(); x.arc(q.px, y, q.r * (1 + 0.2 * glow), 0, Math.PI * 2); x.fill();
      }

      // everything stays behind content: the tissue fades out there, with a soft edge
      x.save();
      x.globalCompositeOperation = "destination-out";
      x.shadowColor = "#000"; x.shadowBlur = 28; x.fillStyle = "#000";
      for (const [l, t, w, h] of blocks) x.fillRect(l - 10, t - 10, w + 20, h + 20);
      x.restore();
      for (let i = ripples.length - 1; i >= 0; i--) {     // where you clicked
        const k = (now - ripples[i].t) / 0.8;
        if (k >= 1) { ripples.splice(i, 1); continue; }
        x.strokeStyle = `rgba(236,232,223,${(0.4 * (1 - k)).toFixed(3)})`; x.lineWidth = 1.2;
        x.beginPath(); x.arc(ripples[i].x, ripples[i].y, 6 + 30 * (1 - Math.pow(1 - k, 3)), 0, Math.PI * 2); x.stroke();
      }
    }
    function frame(t) {
      const dt = Math.min(0.05, (t - (last || t)) / 1000); last = t; now += dt;
      if (!document.hidden) draw();
      requestAnimationFrame(frame);
    }
    if (RM) {                              // still tissue, redrawn on scroll and resize
      draw();
      window.addEventListener("scroll", () => draw(), { passive: true });
      window.addEventListener("resize", () => draw());
    } else requestAnimationFrame(frame);
  })();

  // copy buttons
  document.querySelectorAll(".copy").forEach((b) => b.addEventListener("click", async () => {
    const text = document.getElementById(b.dataset.copy).innerText;
    try { await navigator.clipboard.writeText(text); b.textContent = "copied"; } catch (e) { b.textContent = "select + copy"; }
    setTimeout(() => { b.textContent = "copy"; }, 1600);
  }));

  // ------------------------------------------------------------------ graph theme
  const theme = {
    states: { active: "#d9d5ca", trigger: "#ffffff", updated: "#86a8ff", stale: "#f4b443", kept: "#6bd59a", wrong: "#ff6b5d", superseded: "#6f6e68", accent: "#86a8ff" },
    edge: "#8d8f98", edgeAlpha: 0.42, pulse: "#a9c2ff", text: "#ece8df", triggerText: "#ffffff",
    labelHalo: "rgba(10,11,13,.9)", fontSize: 14, fontFamily: "'Geist', sans-serif", badgeFont: "500 10px 'Geist Mono', monospace",
    lineHeight: 17, breath: 3.2, glow: 18, dim: 0.16, camSpeed: 2.2,
    noteFont: "500 12px 'Geist Mono', monospace", noteColor: "#8a8880",
    cardFill: "rgba(255,255,255,.025)", cardLine: "rgba(255,255,255,.09)", cardActive: "#86a8ff",
    cardTitleFont: `500 ${wide ? 12.5 : 9.5}px 'Geist Mono', monospace`, cardStatusFont: `400 ${wide ? 12 : 10.5}px 'Geist Mono', monospace`,
    background(ctx, g) {      // facts that changed warm the space under them in their state's colour
      g.nodes.forEach((n) => {
        if (n.state === "active" || n.state === "trigger" || n.alpha < 0.3) return;
        const [x, y] = g.toScreen(n.x, n.y), R = Math.max(46, 84 * g.cam.s);
        const a = 0.05 * n.alpha * n.tState;
        const grd = ctx.createRadialGradient(x, y, 0, x, y, R);
        grd.addColorStop(0, `rgba(${n.col.join(",")},${a.toFixed(3)})`);
        grd.addColorStop(1, `rgba(${n.col.join(",")},0)`);
        ctx.fillStyle = grd;
        ctx.fillRect(x - R, y - R, R * 2, R * 2);
      });
    },
    drawNode(ctx, n, x, y, r) {
      ctx.save();
      if (n.state !== "active") { ctx.shadowColor = `rgba(${n.col.join(",")},.65)`; ctx.shadowBlur = 16; }
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${n.col.join(",")})`; ctx.fill();
      ctx.restore();
      if (n.kind === "trigger") { ctx.beginPath(); ctx.arc(x, y, r + 5, 0, Math.PI * 2); ctx.strokeStyle = "rgba(255,255,255,.5)"; ctx.lineWidth = 1; ctx.stroke(); }
    },
  };
  const lay = compact ? { dx: 132, dy: 104 } : {};       // compact: a 104px label slot plus the node

  let data;
  try { data = await GemGraph.loadData("data/scenarios.json"); }
  catch (e) { $("#cp-call").textContent = "the recorded runs could not be loaded"; return; }
  const S = Object.fromEntries(data.scenarios.map((s) => [s.key, s]));
  $("#data-stamp").textContent = `graphs replay GEM runs recorded ${data.generated} · ${data.llm}`;
  const rel = S.relocation;
  const relDecided = rel.steps.filter((s) => s.kind === "revise" || s.kind === "stop");
  $("#m-cheap").textContent = `${relDecided.filter((s) => s.via === "jev").length} of ${relDecided.length}`;
  $("#m-llm").textContent = rel.calls.llm;
  // step 08 numbers: shown for whichever recorded change is running
  $("#s-total").textContent = data.scenarios.reduce((a, s) => a + s.nodes.length, 0);
  const scanCount = (sc) => { const scan = sc.steps.find((s) => s.kind === "scan"); return scan ? scan.checked.length : 0; };
  const linkCount = (sc) => sc.steps.filter((s) => (s.kind === "revise" || s.kind === "stop") && s.depth > 0).length;
  const writeStats = (sc) => `${linkCount(sc)} followed · ${sc.calls.llm} LLM`;
  function showWrite(sc) {
    $("#s-checked").textContent = scanCount(sc);
    $("#s-links").textContent = linkCount(sc);
    $("#s-llm").textContent = sc.calls.llm;
  }
  showWrite(S.launch);

  // ------------------------------------------------------------------ story
  const g = new GemGraph($("#stage"), theme, {
    offsetX: OX, offsetY: wide ? 0 : -window.innerHeight * 0.2,
    labelWidth: compact ? 104 : 180, compact,
    insetTop: phone ? 66 : 84,        // the top bar plus breathing room
  });
  fieldHooks.stageBox = () => {           // screen box around the visible facts and their labels
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    const lw = g.opts.labels ? g.opts.labelWidth * Math.max(0.55, Math.min(1.35, g.cam.s)) + 24 : 12;
    g.nodes.forEach((n) => {
      if (n.alpha < 0.3) return;
      const [sx, sy] = g.toScreen(n.x, n.y);
      x0 = Math.min(x0, sx - 20); x1 = Math.max(x1, sx + lw); y0 = Math.min(y0, sy - 30); y1 = Math.max(y1, sy + 44);
    });
    g.notes.forEach((nt) => {                   // step 08's memory cards
      if (!nt.card || nt.a < 0.3) return;
      const [ax, ay] = g.toScreen(nt.x0, nt.y0), [bx, by] = g.toScreen(nt.x1, nt.y1);
      x0 = Math.min(x0, ax); y0 = Math.min(y0, ay); x1 = Math.max(x1, bx); y1 = Math.max(y1, by);
    });
    if (!(x0 < x1)) return null;
    // the stage canvas scrolls away at the end of the story: move the box with it and clip to what's visible
    const r = g.c.getBoundingClientRect();
    const L = Math.max(r.left, r.left + x0), T = Math.max(r.top, r.top + y0), R = Math.min(r.right, r.left + x1), B = Math.min(r.bottom, r.top + y1);
    return R > L && B > T ? [L, T, R - L, B - T] : null;
  };
  const spots = { relocation: [0, 0], runtime: [1250, -60], launch: [-1250, -330], cloud: [1180, 760], reorg: [-1150, 800], email: [40, 900] };
  const C = {};
  Object.entries(spots).forEach(([k, [x, y]]) => { C[k] = GemGraph.build(g, S[k], k + ":", x, y, lay); });
  // step 08's map: the six memories as compact graphs in a 3x2 grid, titled, so the whole
  // memory is readable at once. Nodes glide between this and the story layout.
  const MAP = [["launch", 0, 0], ["relocation", 1, 0], ["runtime", 2, 0], ["reorg", 0, 1], ["email", 1, 1], ["cloud", 2, 1]];
  const CELL_W = 540, CELL_H = 600, CARD_W = 500, CARD_TOP = 150, CARD_H = 560;
  MAP.forEach(([k, col, row]) => {
    const L = GemGraph.layout(S[k], col * CELL_W, row * CELL_H, 110, 64);
    const c = C[k];
    c.story = Object.fromEntries(c.ids.map((id) => [id, [g.nodes.get(id).hx, g.nodes.get(id).hy]]));
    c.storyHome = c.home;
    c.map = Object.fromEntries(S[k].nodes.map((n) => [k + ":" + n.id, L[n.id]]));
    c.mapHome = L.__trigger;
    c.card = g.card(col * CELL_W - CARD_W / 2, row * CELL_H - CARD_TOP, col * CELL_W + CARD_W / 2, row * CELL_H - CARD_TOP + CARD_H, S[k].title.toUpperCase());
    // centre each graph vertically in its card
    const ys = Object.values(c.map).map((p) => p[1]).concat(c.mapHome[1]);
    const dy = (row * CELL_H - CARD_TOP + CARD_H / 2) - (Math.min(...ys) + Math.max(...ys)) / 2 + 6;
    Object.values(c.map).forEach((p) => { p[1] += dy; });
    c.mapHome = [c.mapHome[0], c.mapHome[1] + dy];
  });
  const placeAll = (which) => Object.values(C).forEach((c) => {
    Object.entries(c[which]).forEach(([id, [x, y]]) => { const n = g.nodes.get(id); n.hx = x; n.hy = y; });
    c.home = which === "map" ? c.mapHome : c.storyHome;
  });
  const all = [].concat(...Object.values(C).map((c) => c.ids));
  const R = C.relocation;
  const ZOOM_OX = OX, WIDE_OX = OX;
  const speed = RM ? 40 : 1;

  let token = { cancelled: true }, current = null;
  const fresh = () => { token.cancelled = true; token = { cancelled: false }; g.clearDials(); g.spotlight = null; return token; };
  const resetAll = () => Object.keys(C).forEach((k) => GemGraph.reset(g, S[k], k + ":"));
  // hero: graph sits higher so the tool-call panel fits under it
  const HERO_OY = wide ? -window.innerHeight * 0.06 : undefined;
  // stacked layout: centre the graph between the top bar and the active step's text card
  const stackOy = () => {
    if (phone) return (g.opts.insetTop - 20) / 2;                 // the stage is the graph's own: fill it below the bar
    const card = document.querySelector(".step.on .card"), H = window.innerHeight;
    if (!card) return g.opts.offsetY;
    const pad = parseFloat(getComputedStyle(card.closest(".step")).paddingBottom) || 0;
    const cardTop = H - pad - card.offsetHeight;                  // .step aligns the card to its bottom
    return (g.opts.insetTop + cardTop - 16) / 2 - H / 2;
  };
  const focusOn = (key, hero = false) => {
    const c = C[key];
    placeAll("story");
    g.opts.compact = compact;
    g.opts.labels = true;
    MAP.forEach(([k]) => { C[k].card.activeT = 0; });
    g.showNotes(false);
    g.setFocus(c.ids.concat(key + ":new"), 0);
    g.frame(c.ids, compact ? 14 : 40, 1.15, 0.18, [c.home], ZOOM_OX, wide ? (hero ? HERO_OY : 0) : stackOy());
  };

  // ---------------- hero tool-call panel: the call types itself, results fill in as the cascade runs
  const cpCall = $("#cp-call"), cpRows = $("#cp-rows"), cpFoot = $("#cp-foot");
  const ROW = { updated: "updated", stale: "stale", kept: "unaffected", superseded: "superseded" };
  async function typeCall(t, text) {
    cpCall.classList.add("typing");
    for (let i = 1; i <= text.length; i++) {
      if (t.cancelled) return;
      cpCall.textContent = text.slice(0, i);
      await g.wait(RM ? 0 : 26);
    }
    cpCall.classList.remove("typing");
  }
  const addRow = ({ state, text }) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="t-${state}">${ROW[state] || state}</span><span></span>`;
    li.lastChild.textContent = text;
    cpRows.appendChild(li);
  };
  const overview = () => {
    placeAll("map");
    g.opts.labels = false;            // the map shows shape and state; titles say what's running
    g.setFocus(all, 1);
    g.showNotes(true);
    const xs = MAP.map(([, c]) => c * CELL_W), ys = MAP.map(([, , r]) => r * CELL_H);
    g.frame(all, 40, 0.95, 0.1, [[Math.min(...xs) - CARD_W / 2, Math.min(...ys) - CARD_TOP], [Math.max(...xs) + CARD_W / 2, Math.max(...ys) - CARD_TOP + CARD_H]], WIDE_OX, wide ? 0 : stackOy());
  };
  const play = (key, t, o = {}) => t.cancelled ? Promise.resolve() :
    GemGraph.play(g, S[key], key + ":", C[key].home, Object.assign({ token: t, dials: false, who: false }, o, { speed: (o.speed || 1) * speed }));
  const settled = (key) => g.nodes.get(key + ":" + S[key].steps.find((s) => s.kind === "revise" && s.depth > 0)?.id)?.state !== "active";

  const beats = {
    async hero() { // the relocation write on a loop, with the tool call and what it returns
      const t = fresh(); resetAll(); focusOn("relocation", true);
      await g.wait(500);
      do {
        cpRows.innerHTML = ""; cpFoot.textContent = "running…"; cpCall.textContent = "";
        await typeCall(t, `add_memory("${rel.trigger}")`);
        if (t.cancelled) return;
        await play("relocation", t, { onStep: (st) => { if (!t.cancelled) addRow(st); } });
        if (t.cancelled) return;
        cpFoot.innerHTML = `<b>${rel.calls.llm}</b> LLM call${rel.calls.llm === 1 ? "" : "s"}`;
        if (RM) return;
        await g.wait(4200); if (t.cancelled) return;
        GemGraph.reset(g, rel, "relocation:"); g.spotlight = null;
        await g.wait(700);
      } while (!t.cancelled);
    },
    async built() { // the links, traced from the root down
      const t = fresh(); resetAll(); focusOn("relocation");
      await g.wait(500);
      const order = g.edges.filter((e) => e.type === "derived" && e.a.startsWith("relocation:"))
        .sort((a, b) => g.nodes.get(a.b).hy - g.nodes.get(b.b).hy);
      for (const e of order) {
        if (t.cancelled) return;
        g.spotlight = e.b;
        g.pulse(e.a, e.b, { dur: 900, color: "#9a9ca6" });
        await g.wait(RM ? 0 : 520);
      }
    },
    async change() { const t = fresh(); resetAll(); focusOn("relocation"); await play("relocation", t, { until: "conflict" }); },
    async flat() {
      const t = fresh(); resetAll(); focusOn("relocation");
      await play("relocation", t, { mode: "flat", speed: 1.4 });
      if (!t.cancelled) g.spotlight = "relocation:" + rel.nodes[1].id; // the commute: what the agent reads back
    },
    async cascade() { const t = fresh(); resetAll(); focusOn("relocation"); await play("relocation", t, { speed: 1.05 }); },
    async rewrite() { const t = fresh(); resetAll(); focusOn("reorg"); await g.wait(600); await play("reorg", t); },
    async stale() {
      const t = fresh(); resetAll(); focusOn("relocation");
      await play("relocation", t, { speed: 8, badges: true });
      if (t.cancelled) return;
      const commute = "relocation:" + rel.nodes[1].id;
      g.spotlight = commute;
      g.setState(commute, "stale", { badge: "stale · ask the user" });
    },
    async cheap() {
      const t = fresh(); focusOn("relocation");
      if (!settled("relocation")) { resetAll(); await play("relocation", t, { speed: 8 }); }
      if (t.cancelled) return;
      rel.steps.forEach((s) => {
        if (s.jev && s.depth > 0) g.dial("relocation:" + s.id, s.kind === "stop" ? s.jev.p_unaffected : s.jev.p_affected, "", s.kind === "stop" ? "kept" : "stale");
      });
    },
    async scale() { // six changes in turn; each lights up only its own group
      const t = fresh(); resetAll(); overview();
      MAP.forEach(([k]) => { C[k].card.activeT = 0; C[k].card.status = `${S[k].nodes.length} facts`; });
      await g.wait(RM ? 0 : 1100); if (t.cancelled) return;
      for (const [k] of MAP) {
        if (t.cancelled) return;
        const sc = S[k], c = C[k];
        showWrite(sc);
        c.card.activeT = 1;
        c.card.status = "updating…";
        g.setFocus(c.ids.concat(k + ":new"), 0.3);
        await play(k, t, { badges: false, speed: 1.5, sonarRadius: 170 });
        if (t.cancelled) return;
        c.card.activeT = 0;
        c.card.status = writeStats(sc);
        await g.wait(RM ? 0 : 600);
      }
      if (t.cancelled) return;
      g.setFocus(all, 1);
      $("#s-checked").textContent = data.scenarios.reduce((a, s) => a + scanCount(s), 0);
      $("#s-links").textContent = data.scenarios.reduce((a, s) => a + linkCount(s), 0);
      $("#s-llm").textContent = data.scenarios.reduce((a, s) => a + s.calls.llm, 0);
    },
  };

  const steps = [...document.querySelectorAll(".step")];
  const rail = $(".rail");
  steps.forEach((s, i) => {
    if (!s.id) s.id = "step-" + s.dataset.beat;
    const a = document.createElement("a");
    a.href = "#" + s.id;
    a.setAttribute("aria-label", s.querySelector("h1, h2").textContent);
    a.dataset.label = (s.querySelector(".kicker b")?.textContent || "") + " " + s.querySelector("h1, h2").textContent;
    rail.appendChild(a);
  });
  const panel = $("#callpanel");
  // The story moves only when a step's text card is fully on screen below the top bar. Between two cards
  // neither is fully visible and nothing changes, so slow scrolling can't flip a step back and forth.
  // A card taller than the screen counts once it fills nearly all of it.
  const activate = (step) => {
    const beat = step.dataset.beat;
    if (beat === current) return;
    current = beat;
    [...rail.children].forEach((a, i) => a.classList.toggle("on", steps[i] === step));
    steps.forEach((s) => s.classList.toggle("on", s === step));
    panel.classList.toggle("off", beat !== "hero");
    beats[beat]();
  };
  const storyIO = new IntersectionObserver((es) => es.forEach((e) => {
    const room = window.innerHeight - 64;
    if (e.intersectionRatio >= 0.995 || e.intersectionRect.height >= room * 0.9) activate(e.target.closest(".step"));
  }), { rootMargin: "-64px 0px 0px 0px", threshold: [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 1] });
  // phones: a step leads once its text has come up into the band under the graph
  if (phone) {
    let raf = 0;
    const pick = () => {
      raf = 0;
      const line = window.innerHeight * 0.78;
      let on = steps[0];
      steps.forEach((s) => { if (s.getBoundingClientRect().top < line) on = s; });
      activate(on);
    };
    window.addEventListener("scroll", () => { if (!raf) raf = requestAnimationFrame(pick); }, { passive: true });
  } else steps.forEach((s) => storyIO.observe(s.querySelector(".card")));
  new IntersectionObserver((es) => es.forEach((e) => rail.classList.toggle("off", e.isIntersecting)), { threshold: 0.02 })
    .observe($(".after"));
  current = "hero"; steps[0].classList.add("on"); beats.hero(); g.snap();

  // ------------------------------------------------------------------ playground: use cases are the picker
  const PICKS = [
    ["runtime", "Coding agents", "Lambda → containers", "The limits built on the platform never name it, so text search can't find them."],
    ["relocation", "Personal assistants", "I moved to Mumbai", "Routines are rewritten when the new value is known and flagged when it isn't."],
    ["reorg", "Workplace", "Bob is my new manager", "Three routines hang off one fact. Each gets the new name."],
    ["launch", "Planning", "Launch slipped to Dec 1", "Every date is set from the one before it. The change travels six steps."],
    ["cloud", "Infrastructure", "m5.large → m5.xlarge", "The budget is built on two facts, and only one of them changed."],
    ["email", "Change impact", "Gmail → Fastmail", "The dependency tree beside the change is left alone."],
  ];
  const pg = new GemGraph($("#play-canvas"), theme, { drag: true, labelWidth: compact ? 104 : 175, offsetY: 6, compact });
  let pKey = "runtime", pMode = "gem", pTok = { cancelled: true }, pIv = null, started = false;
  const picks = $("#picks");
  PICKS.forEach(([key, cat, title, desc]) => {
    const li = document.createElement("li"), b = document.createElement("button");
    b.type = "button"; b.dataset.key = key;
    b.innerHTML = '<span class="cat"></span><span class="ttl"></span><span class="desc"></span>';
    b.children[0].textContent = cat; b.children[1].textContent = title; b.children[2].textContent = desc;
    b.addEventListener("click", () => { pKey = key; runPlay(); });
    li.appendChild(b); picks.appendChild(li);
  });
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => { pMode = b.dataset.m; runPlay(); }));
  $("#replay").addEventListener("click", () => runPlay());
  const setR = (id, v) => { document.getElementById(id).textContent = v; };

  async function runPlay() {
    pTok.cancelled = true; pTok = { cancelled: false }; const my = pTok;
    picks.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.key === pKey)));
    document.querySelectorAll(".seg button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.m === pMode)));
    const sc = S[pKey];
    $("#play-q").textContent = (pMode === "flat" ? "flat memory · " : "") + `add_memory("${sc.trigger}")`;
    pg.clear();
    const built = GemGraph.build(pg, sc, "p:", 0, 0, compact ? { dx: 132, dy: 128 } : lay);   // the stage is tall: rows get more room
    pg.frame(built.ids, compact ? 28 : 60, 1.1, 0.18, [built.home]);
    if (!started) { pg.snap(); started = true; }
    const flat = pMode === "flat";
    setR("r-llm", flat ? "–" : 0); setR("r-up", 0); setR("r-st", 0);
    const tick = () => {
      let up = 0, st = 0, wr = 0;
      built.ids.forEach((id) => { const n = pg.nodes.get(id); if (!n) return; if (n.state === "updated") up++; if (n.state === "stale" || n.state === "superseded") st++; if (n.state === "wrong") wr++; });
      setR("r-up", up); setR("r-st", flat ? `${st} · ${wr} still served as true` : st);
    };
    clearInterval(pIv); pIv = setInterval(tick, 150);
    await GemGraph.play(pg, sc, "p:", built.home, {
      token: my, mode: pMode, speed, who: false,
      count: flat ? null : ({ llm }) => { setR("r-llm", llm); },
    });
    if (!my.cancelled) { tick(); clearInterval(pIv); }
  }
  picks.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.key === pKey)));
  new IntersectionObserver((es, o) => { if (es.some((e) => e.isIntersecting)) { o.disconnect(); runPlay(); } }, { threshold: 0.35 })
    .observe($(".play-stage"));
})();
