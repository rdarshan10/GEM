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
  catch (e) { $("#stage-cap").textContent = "the recorded runs could not be loaded"; return; }
  const S = Object.fromEntries(data.scenarios.map((s) => [s.key, s]));
  $("#data-stamp").textContent = `graphs replay GEM runs recorded ${data.generated} · Jev + ${data.llm}`;
  const rel = S.relocation;
  $("#m-jev").textContent = rel.calls.jev;
  $("#m-llm").textContent = rel.calls.llm;
  $("#stage-cap").innerHTML = `replaying a recorded run · <b>${rel.calls.jev} Jev calls · ${rel.calls.llm} LLM call${rel.calls.llm === 1 ? "" : "s"}</b>`;

  // ------------------------------------------------------------------ story
  const g = new GemGraph($("#stage"), theme, {
    offsetX: wide ? window.innerWidth * 0.17 : 0, offsetY: wide ? 0 : -window.innerHeight * 0.2,
    labelWidth: compact ? 150 : 180, compact,
  });
  const spots = { relocation: [0, 0], runtime: [1250, -60], launch: [-1250, -330], cloud: [1180, 760], reorg: [-1150, 800], email: [40, 900] };
  const C = {};
  Object.entries(spots).forEach(([k, [x, y]]) => { C[k] = GemGraph.build(g, S[k], k + ":", x, y, lay); });
  Object.entries(C).forEach(([k, c]) => {
    const xs = c.ids.map((id) => g.nodes.get(id).hx), ys = c.ids.map((id) => g.nodes.get(id).hy);
    g.note((Math.min(...xs) + Math.max(...xs)) / 2, Math.max(...ys) + 70, S[k].title.toUpperCase());
  });
  const all = [].concat(...Object.values(C).map((c) => c.ids));
  const R = C.relocation;
  const ZOOM_OX = wide ? window.innerWidth * 0.21 : 0, WIDE_OX = wide ? window.innerWidth * 0.24 : 0;
  const speed = RM ? 40 : 1;

  let token = { cancelled: true }, current = null;
  const fresh = () => { token.cancelled = true; token = { cancelled: false }; g.clearDials(); g.spotlight = null; return token; };
  const resetAll = () => Object.keys(C).forEach((k) => GemGraph.reset(g, S[k], k + ":"));
  const focusOn = (key) => {
    const c = C[key];
    g.showNotes(false);
    g.setFocus(c.ids.concat(key + ":new"), 0);
    g.frame(c.ids, 40, 1.15, 0.18, [c.home], ZOOM_OX);
  };
  const overview = () => {
    g.setFocus(all, 1);
    all.forEach((id) => (g.nodes.get(id).labelAT = 0));
    g.showNotes(true);
    g.frame(all, 60, 0.5, 0.18, Object.values(C).map((c) => [c.home[0], c.home[1] - 40]), WIDE_OX);
  };
  const play = (key, t, o = {}) =>
    GemGraph.play(g, S[key], key + ":", C[key].home, Object.assign({ token: t, dials: false }, o, { speed: (o.speed || 1) * speed }));
  const settled = (key) => g.nodes.get(key + ":" + S[key].steps.find((s) => s.kind === "revise" && s.depth > 0)?.id)?.state !== "active";

  const beats = {
    async hero() { // the relocation write, on a loop: the first screen shows what GEM does
      const t = fresh(); resetAll(); focusOn("relocation");
      await g.wait(700);
      do {
        await play("relocation", t);
        if (t.cancelled || RM) return;
        await g.wait(3400); if (t.cancelled) return;
        GemGraph.reset(g, rel, "relocation:"); g.spotlight = null;
        await g.wait(900);
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
    async scale() {
      const t = fresh(); resetAll(); overview();
      await g.wait(RM ? 0 : 900); if (t.cancelled) return;
      g.setFocus(C.launch.ids.concat("launch:new"), 0.3);
      all.forEach((id) => (g.nodes.get(id).labelAT = 0));
      await play("launch", t, { speed: 1.3, badges: false });
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
  const stageCap = $("#stage-cap");
  const storyIO = new IntersectionObserver((es) => {
    es.forEach((e) => {
      if (!e.isIntersecting) return;
      const beat = e.target.dataset.beat;
      if (beat === current) return;
      current = beat;
      [...rail.children].forEach((a, i) => a.classList.toggle("on", steps[i] === e.target));
      stageCap.style.opacity = beat === "hero" ? 1 : 0;
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
    setR("r-jev", flat ? "–" : 0); setR("r-llm", flat ? "–" : 0); setR("r-up", 0); setR("r-st", 0); setR("r-last", "");
    const tick = () => {
      let up = 0, st = 0, wr = 0;
      built.ids.forEach((id) => { const n = pg.nodes.get(id); if (!n) return; if (n.state === "updated") up++; if (n.state === "stale" || n.state === "superseded") st++; if (n.state === "wrong") wr++; });
      setR("r-up", up); setR("r-st", flat ? `${st} · ${wr} still served as true` : st);
    };
    clearInterval(pIv); pIv = setInterval(tick, 150);
    await GemGraph.play(pg, sc, "p:", built.home, {
      token: my, mode: pMode, speed,
      log: (k, x) => { if (!my.cancelled && k !== "write") setR("r-last", x.trim()); },
      count: flat ? null : ({ jev, llm }) => { setR("r-jev", jev); setR("r-llm", llm); },
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
