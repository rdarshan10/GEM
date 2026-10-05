/* GEM site: the pinned story graph, the playground, and page chrome. Everything on the graphs replays
   recorded GEM runs from data/scenarios.json (made by docs/build_scenarios.py). */
(async function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const RM = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const wide = window.innerWidth > 760, compact = !wide;

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
    drawNode(ctx, n, x, y, r) {
      ctx.save();
      if (n.state !== "active") { ctx.shadowColor = `rgba(${n.col.join(",")},.65)`; ctx.shadowBlur = 16; }
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${n.col.join(",")})`; ctx.fill();
      ctx.restore();
      if (n.kind === "trigger") { ctx.beginPath(); ctx.arc(x, y, r + 5, 0, Math.PI * 2); ctx.strokeStyle = "rgba(255,255,255,.5)"; ctx.lineWidth = 1; ctx.stroke(); }
    },
  };
  const lay = compact ? { dx: 120, dy: 96 } : {};

  let data;
  try { data = await GemGraph.loadData("data/scenarios.json"); }
  catch (e) { $("#cp-call").textContent = "the recorded runs could not be loaded"; return; }
  const S = Object.fromEntries(data.scenarios.map((s) => [s.key, s]));
  $("#data-stamp").textContent = `graphs replay GEM runs recorded ${data.generated} · ${data.llm}`;
  const rel = S.relocation;
  const relDecided = rel.steps.filter((s) => s.kind === "revise" || s.kind === "stop");
  $("#m-cheap").textContent = `${relDecided.filter((s) => s.via === "jev").length} of ${relDecided.length}`;
  $("#m-llm").textContent = rel.calls.llm;
  // step 08 numbers, from the recorded launch run
  const la = S.launch, laScan = la.steps.find((s) => s.kind === "scan");
  $("#s-checked").textContent = laScan ? laScan.checked.length : 0;
  $("#s-links").textContent = la.steps.filter((s) => (s.kind === "revise" || s.kind === "stop") && s.depth > 0).length;
  $("#s-llm").textContent = la.calls.llm;
  $("#s-total").textContent = data.scenarios.reduce((a, s) => a + s.nodes.length, 0);

  // ------------------------------------------------------------------ story
  const g = new GemGraph($("#stage"), theme, {
    offsetX: wide ? window.innerWidth * 0.17 : 0, offsetY: wide ? 0 : -window.innerHeight * 0.2,
    labelWidth: compact ? 150 : 180, compact,
  });
  const spots = { relocation: [0, 0], runtime: [1250, -60], launch: [-1250, -330], cloud: [1180, 760], reorg: [-1150, 800], email: [40, 900] };
  const C = {};
  Object.entries(spots).forEach(([k, [x, y]]) => { C[k] = GemGraph.build(g, S[k], k + ":", x, y, lay); });
  // step 08's map: the six memories as compact graphs in a 3x2 grid, titled, so the whole
  // memory is readable at once. Nodes glide between this and the story layout.
  const MAP = [["launch", 0, 0], ["relocation", 1, 0], ["runtime", 2, 0], ["reorg", 0, 1], ["email", 1, 1], ["cloud", 2, 1]];
  const CELL_W = 560, CELL_H = 520;
  MAP.forEach(([k, col, row]) => {
    const L = GemGraph.layout(S[k], col * CELL_W, row * CELL_H, 110, 64);
    const c = C[k];
    c.story = Object.fromEntries(c.ids.map((id) => [id, [g.nodes.get(id).hx, g.nodes.get(id).hy]]));
    c.storyHome = c.home;
    c.map = Object.fromEntries(S[k].nodes.map((n) => [k + ":" + n.id, L[n.id]]));
    c.mapHome = L.__trigger;
    g.note(col * CELL_W, row * CELL_H - 135, S[k].title.toUpperCase());
  });
  const placeAll = (which) => Object.values(C).forEach((c) => {
    Object.entries(c[which]).forEach(([id, [x, y]]) => { const n = g.nodes.get(id); n.hx = x; n.hy = y; });
    c.home = which === "map" ? c.mapHome : c.storyHome;
  });
  const all = [].concat(...Object.values(C).map((c) => c.ids));
  const R = C.relocation;
  const ZOOM_OX = wide ? window.innerWidth * 0.21 : 0, WIDE_OX = wide ? window.innerWidth * 0.24 : 0;
  const speed = RM ? 40 : 1;

  let token = { cancelled: true }, current = null;
  const fresh = () => { token.cancelled = true; token = { cancelled: false }; g.clearDials(); g.spotlight = null; return token; };
  const resetAll = () => Object.keys(C).forEach((k) => GemGraph.reset(g, S[k], k + ":"));
  // hero: graph sits higher so the tool-call panel fits under it
  const HERO_OY = wide ? -window.innerHeight * 0.13 : undefined;
  const focusOn = (key, hero = false) => {
    const c = C[key];
    placeAll("story");
    g.opts.compact = compact;
    g.showNotes(false);
    g.setFocus(c.ids.concat(key + ":new"), 0);
    g.frame(c.ids, 40, 1.15, 0.18, [c.home], ZOOM_OX, hero ? HERO_OY : undefined);
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
    g.opts.compact = true;            // in the map only the fact being decided is labelled
    g.setFocus(all, 1);
    g.showNotes(true);
    const xs = MAP.map(([, c]) => c * CELL_W), ys = MAP.map(([, , r]) => r * CELL_H);
    g.frame(all, 50, 0.95, 0.18, [[Math.min(...xs) - 160, Math.min(...ys) - 150], [Math.max(...xs) + 40, Math.max(...ys) + 330]], WIDE_OX, 0);
  };
  const play = (key, t, o = {}) => t.cancelled ? Promise.resolve() :
    GemGraph.play(g, S[key], key + ":", C[key].home, Object.assign({ token: t, dials: false, who: false }, o, { speed: (o.speed || 1) * speed }));
  const settled = (key) => g.nodes.get(key + ":" + S[key].steps.find((s) => s.kind === "revise" && s.depth > 0)?.id)?.state !== "active";

  const beats = {
    async hero() { // the relocation write on a loop, with the tool call and what it returns
      const t = fresh(); resetAll(); focusOn("relocation", true);
      await g.wait(500);
      do {
        cpRows.innerHTML = ""; cpFoot.textContent = ""; cpCall.textContent = "";
        await typeCall(t, `add_memory("${rel.trigger}")`);
        if (t.cancelled) return;
        await play("relocation", t, { onStep: (st) => { if (!t.cancelled) addRow(st); } });
        if (t.cancelled) return;
        cpFoot.innerHTML = `<b>${rel.calls.llm}</b> LLM call${rel.calls.llm === 1 ? "" : "s"} · everything else left alone`;
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
    async scale() { // the whole memory at once; one write lights up only what it touches
      const t = fresh(); resetAll(); overview();
      await g.wait(RM ? 0 : 1300); if (t.cancelled) return;
      g.setFocus(C.launch.ids.concat("launch:new"), 0.32);
      await play("launch", t, { badges: false });
    },
  };

  const steps = [...document.querySelectorAll(".step")];
  const rail = $(".rail");
  steps.forEach((s, i) => {
    if (!s.id) s.id = "step-" + s.dataset.beat;
    const a = document.createElement("a");
    a.href = "#" + s.id;
    a.setAttribute("aria-label", s.querySelector("h1, h2").textContent);
    rail.appendChild(a);
  });
  const panel = $("#callpanel");
  const storyIO = new IntersectionObserver((es) => {
    es.forEach((e) => {
      if (!e.isIntersecting) return;
      const beat = e.target.dataset.beat;
      if (beat === current) return;
      current = beat;
      [...rail.children].forEach((a, i) => a.classList.toggle("on", steps[i] === e.target));
      panel.classList.toggle("off", beat !== "hero");
      beats[beat]();
    });
  }, { threshold: 0.55 });
  steps.forEach((s) => storyIO.observe(s));
  new IntersectionObserver((es) => es.forEach((e) => rail.classList.toggle("off", e.isIntersecting)), { threshold: 0.02 })
    .observe($(".after"));
  current = "hero"; beats.hero(); g.snap();

  // ------------------------------------------------------------------ playground
  const SHORT = { relocation: "I moved to Mumbai", runtime: "Lambda → containers", launch: "Launch slipped to Dec 1", cloud: "m5.large → m5.xlarge", reorg: "Bob is my new manager", email: "Gmail → Fastmail" };
  const pg = new GemGraph($("#play-canvas"), theme, { drag: true, labelWidth: compact ? 150 : 175, offsetY: 6, compact });
  let pKey = "relocation", pMode = "gem", pTok = { cancelled: true }, pIv = null, started = false;
  const chips = $("#chips");
  data.scenarios.forEach((sc) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip"; b.textContent = SHORT[sc.key] || sc.title;
    b.addEventListener("click", () => { pKey = sc.key; runPlay(); });
    chips.appendChild(b);
  });
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => { pMode = b.dataset.m; runPlay(); }));
  $("#replay").addEventListener("click", () => runPlay());
  const setR = (id, v) => { document.getElementById(id).textContent = v; };

  async function runPlay() {
    pTok.cancelled = true; pTok = { cancelled: false }; const my = pTok;
    [...chips.children].forEach((b, i) => b.setAttribute("aria-pressed", String(data.scenarios[i].key === pKey)));
    document.querySelectorAll(".seg button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.m === pMode)));
    const sc = S[pKey];
    $("#play-q").textContent = (pMode === "flat" ? "flat memory · " : "") + `add_memory("${sc.trigger}")`;
    pg.clear();
    const built = GemGraph.build(pg, sc, "p:", 0, 0, lay);
    pg.frame(built.ids, 60, 1.1, 0.18, [built.home]);
    if (!started) { pg.snap(); started = true; }
    const flat = pMode === "flat";
    setR("r-llm", flat ? "–" : 0); setR("r-up", 0); setR("r-st", 0); setR("r-last", "");
    const tick = () => {
      let up = 0, st = 0, wr = 0;
      built.ids.forEach((id) => { const n = pg.nodes.get(id); if (!n) return; if (n.state === "updated") up++; if (n.state === "stale" || n.state === "superseded") st++; if (n.state === "wrong") wr++; });
      setR("r-up", up); setR("r-st", flat ? `${st} · ${wr} still served as true` : st);
    };
    clearInterval(pIv); pIv = setInterval(tick, 150);
    await GemGraph.play(pg, sc, "p:", built.home, {
      token: my, mode: pMode, speed, who: false,
      log: (k, x) => { if (!my.cancelled && k !== "write") setR("r-last", x.trim().replace(/ \(P (un)?affected [0-9.]+\)/, "")); },
      count: flat ? null : ({ llm }) => { setR("r-llm", llm); },
    });
    if (!my.cancelled) { tick(); clearInterval(pIv); }
  }
  if (compact) $("#play-q").style.display = "none";
  new IntersectionObserver((es, o) => { if (es.some((e) => e.isIntersecting)) { o.disconnect(); runPlay(); } }, { threshold: 0.35 })
    .observe($(".play-stage"));

  // use cases: "try it" loads that example into the playground
  document.querySelectorAll("[data-try]").forEach((b) => b.addEventListener("click", () => {
    pKey = b.dataset.try; pMode = "gem";
    $("#play").scrollIntoView({ behavior: RM ? "auto" : "smooth" });
    setTimeout(runPlay, RM ? 0 : 500);
  }));
})();
