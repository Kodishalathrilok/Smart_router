"use strict";

// All data comes from the SmartRouter API; nothing here reads internal state.
const TIERS = ["cheap", "mid", "strong"];
const SVG_NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);
let threshold = null;

// ---------- formatting ----------
const fmt = (v, d = 2) => (v == null ? "—" : Number(v).toFixed(d));
const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const ms = (v) => (v == null ? "—" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`);
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : "—");
const reduction = (p) => (p == null ? "—" : p >= 0 ? `${p.toFixed(1)}% lower` : `${(-p).toFixed(1)}% higher`);
const STATUS_TEXT = { accepted: "Accepted", best_effort: "Best effort (below threshold)", failed: "Failed" };

// ---------- DOM helpers ----------
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  for (const c of children) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

function svgEl(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text != null) node.textContent = String(text);
  return node;
}

function svgRoot(width, height, label) {
  return svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": label });
}

function empty(container, message) {
  container.replaceChildren(el("div", { class: "empty" }, message));
}

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || body.error || response.statusText);
  return body;
}

// ---------- query ----------
function showResult(r) {
  $("result").hidden = false;
  $("answer").textContent = r.answer ?? "(no answer: every attempt failed)";
  $("r-initial").textContent = `${cap(r.initial_tier)} · ${r.initial_model}`;
  $("r-final").textContent = r.final_tier ? `${cap(r.final_tier)} · ${r.final_model}` : "—";
  $("r-quality").textContent = `${fmt(r.quality_score)} (threshold ${fmt(r.quality_threshold)})`;
  $("r-status").textContent = STATUS_TEXT[r.final_status] ?? r.final_status;
  $("r-status").className = `status-${r.final_status}`;
  $("r-escalated").textContent = r.escalated ? "Yes" : "No";
  $("r-attempts").textContent = r.attempts;
  $("r-latency").textContent = ms(r.latency_ms);
  $("r-cost").textContent = fmt(r.normalized_cost, 1);
  $("trace").replaceChildren(...r.trace.map((s) => el("li", { class: `k-${s.kind}` }, s.label)));
}

// GET /api/query/{id} returns the full record; map it to the POST response shape.
function recordToResult(rec) {
  return {
    ...rec,
    quality_score: rec.final_quality_score,
    attempts: rec.attempts.length,
    latency_ms: rec.total_latency_ms,
  };
}

async function onSubmit(event) {
  event.preventDefault();
  const query = $("query").value.trim();
  if (!query) return;
  const button = $("run-btn");
  button.disabled = true;
  $("ask-status").textContent = "Routing…";
  try {
    showResult(await api("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    }));
    $("ask-status").textContent = "";
    $("trace-box").open = true;
    await refresh();
  } catch (err) {
    $("ask-status").textContent = `Error: ${err.message}`;
  } finally {
    button.disabled = false;
  }
}

async function openQuery(queryId) {
  try {
    showResult(recordToResult(await api(`/api/query/${encodeURIComponent(queryId)}`)));
    $("trace-box").open = true;
    $("result").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  } catch (err) {
    $("ask-status").textContent = `Error: ${err.message}`;
  }
}

// ---------- overview ----------
function renderCards(stats) {
  const n = stats.n || 0;
  const cards = [
    ["Queries processed", n, ""],
    ["Average quality", fmt(stats.quality?.mean), "judge score, 0–1"],
    ["Normalized cost", fmt(stats.cost?.mean), n ? `per query · total ${fmt(stats.cost?.total, 1)}` : ""],
    ["Cost vs. all-cheap", stats.cost_vs_all_cheap_x == null ? "—" : `${fmt(stats.cost_vs_all_cheap_x, 2)}×`,
      `${reduction(stats.cost_reduction_vs_all_strong_pct)} vs. all-strong · generation only, judge excluded`],
    ["Escalation rate", pct(stats.escalation?.rate), ""],
    ["Average latency", ms(stats.latency_ms?.mean), n ? `p95 ${ms(stats.latency_ms?.p95)}` : ""],
  ];
  $("cards").replaceChildren(...cards.map(([label, value, hint]) =>
    el("div", { class: "card" }, el("div", { class: "label" }, label), el("div", { class: "value" }, value), el("div", { class: "hint" }, hint))));
  $("cards-note").textContent = n
    ? "Live-traffic numbers. For controlled comparisons against baselines, see Benchmark below."
    : "No queries yet. Ask something above.";
}

function renderUtilization(stats) {
  const box = $("util-chart");
  const share = stats.utilization?.calls_pct;
  if (!share) return empty(box, "No generation calls yet.");
  const counts = stats.utilization.calls;
  const W = 420, rowH = 36, labelW = 70, barMax = W - labelW - 90;
  const svg = svgRoot(W, rowH * 3, "Model utilization by tier");
  TIERS.forEach((tier, i) => {
    const y = i * rowH;
    const width = Math.max(2, (share[tier] ?? 0) * barMax);
    svg.append(
      svgEl("text", { x: 0, y: y + 22, class: "strong-label" }, cap(tier)),
      svgEl("rect", { x: labelW, y: y + 8, width: barMax, height: 20, rx: 4, class: "gridline", fill: "none" }),
      svgEl("rect", { x: labelW, y: y + 8, width, height: 20, rx: 4, class: `fill-${tier}` }),
      svgEl("text", { x: labelW + width + 8, y: y + 22 }, `${pct(share[tier])} (${counts[tier]})`),
    );
  });
  box.replaceChildren(svg);
}

function renderScatter(routes) {
  const box = $("scatter-chart");
  const points = routes.filter((r) => r.quality_score != null);
  if (!points.length) return empty(box, "No scored queries yet.");
  const W = 460, H = 260, m = { l: 40, r: 12, t: 10, b: 36 };
  const maxCost = Math.max(1, ...points.map((p) => p.normalized_cost));
  const x = (c) => m.l + (c / maxCost) * (W - m.l - m.r);
  const y = (q) => H - m.b - q * (H - m.t - m.b);
  const svg = svgRoot(W, H, "Quality versus normalized cost per query");
  for (const q of [0, 0.25, 0.5, 0.75, 1]) {
    svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: y(q), y2: y(q), class: "gridline" }), svgEl("text", { x: 4, y: y(q) + 4 }, q.toFixed(2)));
  }
  const ticks = [...new Set([0, ...points.map((p) => p.normalized_cost)])].sort((a, b) => a - b);
  for (const c of ticks) svg.append(svgEl("text", { x: x(c), y: H - m.b + 16, "text-anchor": "middle" }, c));
  svg.append(
    svgEl("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }),
    svgEl("text", { x: (W + m.l) / 2, y: H - 4, "text-anchor": "middle" }, "Normalized cost"),
  );
  if (threshold != null) svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: y(threshold), y2: y(threshold), class: "threshold" }));
  points.forEach((p, i) => {
    const jitter = ((i * 37) % 11 - 5) * 1.2; // spread identical costs horizontally
    const dot = svgEl("circle", { cx: x(p.normalized_cost) + jitter, cy: y(p.quality_score), r: 5, class: `fill-${p.final_tier}`, "fill-opacity": 0.75 });
    dot.append(svgEl("title", {}, `${p.query}\nquality ${fmt(p.quality_score)} · cost ${p.normalized_cost} · ${p.final_tier}`));
    svg.append(dot);
  });
  const legend = el("div", { class: "legend" }, ...TIERS.map((t) => el("span", {}, el("i", { class: `fill-${t}`, style: `background: var(--${t})` }), `final: ${t}`)));
  box.replaceChildren(svg, legend);
}

function renderFlow(stats) {
  const box = $("flow-chart");
  const flow = stats.routing_flow;
  if (!flow?.length) return empty(box, "No routing decisions yet.");
  const count = (from, to) => flow.find((e) => e.from === from && e.to === to)?.count ?? 0;
  const W = 520, H = 250, nodeW = 110, nodeH = 40, top = 30, accY = 150;
  const gap = (W - 3 * nodeW) / 2;
  const nx = (i) => i * (nodeW + gap);
  const mid = top + nodeH / 2;
  const svg = svgRoot(W, H, "Routing flow between tiers with counts");
  const defs = svgEl("defs");
  defs.append(arrowMarker());
  svg.append(defs);

  const label = (x, y, n) => svg.append(svgEl("text", { x, y, "text-anchor": "middle", class: "strong-label" }, n));
  const line = (x1, y1, x2, y2, cls) => svg.append(svgEl("path", { d: `M${x1},${y1} L${x2},${y2}`, class: `edge ${cls}`, "marker-end": "url(#arrow)" }));

  TIERS.forEach((tier, i) => {
    const cx = nx(i) + nodeW / 2;
    svg.append(
      svgEl("rect", { x: nx(i), y: top, width: nodeW, height: nodeH, rx: 8, class: "node" }),
      svgEl("rect", { x: nx(i), y: top, width: 6, height: nodeH, rx: 3, class: `fill-${tier}` }),
      svgEl("text", { x: cx, y: top + 25, "text-anchor": "middle", class: "strong-label" }, cap(tier)),
      svgEl("rect", { x: nx(i), y: accY, width: nodeW, height: 34, rx: 8, class: "node" }),
      svgEl("text", { x: cx, y: accY + 22, "text-anchor": "middle" }, "Accepted"),
    );
    const accepted = count(tier, "accepted");
    if (accepted) {
      line(cx, top + nodeH, cx, accY - 2, "edge-ok");
      label(cx + 16, (top + nodeH + accY) / 2 + 4, accepted);
    }
    const stopped = [["best effort", count(tier, "best_effort")], ["failed", count(tier, "failed")]].filter(([, n]) => n);
    if (stopped.length) {
      svg.append(svgEl("text", { x: cx, y: accY + 58, "text-anchor": "middle", class: "fill-bad" }, `stopped: ${stopped.map(([k, n]) => `${k} ${n}`).join(" · ")}`));
    }
  });
  for (const [i, from, to] of [[0, "cheap", "mid"], [1, "mid", "strong"]]) {
    const n = count(from, to);
    if (!n) continue;
    line(nx(i) + nodeW, mid, nx(i + 1) - 2, mid, "edge-bad");
    label(nx(i) + nodeW + gap / 2, mid - 6, n);
  }
  const skip = count("cheap", "strong");
  if (skip) {
    const x1 = nx(0) + nodeW / 2, x2 = nx(2) + nodeW / 2;
    svg.append(svgEl("path", { d: `M${x1},${top} C${x1},2 ${x2},2 ${x2},${top - 2}`, class: "edge edge-bad", "marker-end": "url(#arrow)" }));
    label(W / 2, 12, skip);
  }
  box.replaceChildren(svg);
}

function arrowMarker() {
  const marker = svgEl("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" });
  marker.append(svgEl("path", { d: "M0,0 L10,5 L0,10 z", style: "fill: var(--muted)" }));
  return marker;
}

function renderFailures(failures) {
  const box = $("failure-chart");
  const entries = Object.entries(failures.groups || {});
  if (!entries.length) return empty(box, "No failed attempts recorded.");
  const W = 420, rowH = 30, labelW = 170, barMax = W - labelW - 40;
  const max = Math.max(...entries.map(([, n]) => n));
  const svg = svgRoot(W, rowH * entries.length, "Failed attempts by cause");
  entries.forEach(([label, n], i) => {
    const y = i * rowH;
    const width = Math.max(2, (n / max) * barMax);
    svg.append(
      svgEl("text", { x: 0, y: y + 19 }, label),
      svgEl("rect", { x: labelW, y: y + 6, width, height: 18, rx: 4, class: "fill-bad", "fill-opacity": 0.8 }),
      svgEl("text", { x: labelW + width + 6, y: y + 19, class: "strong-label" }, n),
    );
  });
  box.replaceChildren(svg);
}

function renderRoutes(routes) {
  const body = $("routes-body");
  if (!routes.length) {
    body.replaceChildren(el("tr", {}, el("td", { colspan: 7, class: "empty" }, "No queries yet.")));
    return;
  }
  body.replaceChildren(...routes.map((r) => {
    const row = el("tr", { tabindex: 0, title: "Show routing trace" },
      el("td", { class: "q" }, r.query),
      el("td", {}, el("span", { class: `pill ${r.initial_tier}` }, r.initial_tier), " ", r.initial_model),
      el("td", {}, r.final_tier ? el("span", { class: `pill ${r.final_tier}` }, r.final_tier) : "—", " ", r.final_model ?? ""),
      el("td", { class: "num" }, fmt(r.quality_score)),
      el("td", {}, r.escalated ? `Yes (${r.attempts - 1})` : "No"),
      el("td", { class: "num" }, ms(r.latency_ms)),
      el("td", { class: `status-${r.final_status}` }, STATUS_TEXT[r.final_status] ?? r.final_status),
    );
    row.addEventListener("click", () => openQuery(r.query_id));
    row.addEventListener("keydown", (e) => { if (e.key === "Enter") openQuery(r.query_id); });
    return row;
  }));
}

// ---------- benchmark ----------
function renderBenchmark(b) {
  const box = $("benchmark");
  if (b.status !== "complete") {
    box.replaceChildren(el("p", { class: "notice" }, "Not evaluated yet. Run ", el("code", {}, "python -m benchmark.run"), " to generate measured results."));
    return;
  }
  const parts = [];
  if (b.mock_mode) parts.push(el("p", { class: "notice" }, "Mock run: synthetic answers and scores. These numbers only exercise the pipeline and do not measure real models."));
  parts.push(el("p", { class: "muted small" }, `Run ${b.run_id} · ${b.dataset.n} items · ${b.created_at}`));
  if (b.claim) parts.push(el("p", { class: "notice" }, el("strong", {}, `Claim verdict: ${b.claim.verdict.replaceAll("_", " ")}. `), b.claim.explanation));
  const headers = ["System", "Strict accuracy", "95% CI", "Cost incl. judge", "vs. cheap", "Calls/query", "Escalation", "p50 latency"];
  const rows = Object.entries(b.systems).map(([name, s]) => {
    const c = b.comparisons?.[name] ?? {};
    const ci = s.strict?.ci95;
    return el("tr", {},
      el("td", {}, s.label),
      el("td", { class: "num" }, pct(s.strict?.accuracy)),
      el("td", { class: "num" }, ci ? `${pct(ci[0])}–${pct(ci[1])}` : "—"),
      el("td", { class: "num" }, fmt(s.cost_incl_judge?.mean)),
      el("td", { class: "num" }, c.cost_incl_judge_vs_cheap_x == null ? "—" : `${fmt(c.cost_incl_judge_vs_cheap_x, 2)}×`),
      el("td", { class: "num" }, fmt(s.serving_calls_mean)),
      el("td", { class: "num" }, pct(s.escalation?.rate)),
      el("td", { class: "num" }, ms(s.latency_ms?.median)),
    );
  });
  const table = el("table", {}, el("thead", {}, el("tr", {}, ...headers.map((h) => el("th", {}, h)))), el("tbody", {}, ...rows));
  parts.push(el("div", { class: "table-scroll" }, table));
  parts.push(el("p", { class: "muted small" }, "Full report: benchmark/results/latest.md"));
  box.replaceChildren(...parts);
}

// ---------- boot ----------
async function loadModels() {
  const m = await api("/api/models");
  threshold = m.quality_threshold;
  $("mode-badge").hidden = !m.mock_mode;
  const byRole = Object.fromEntries(m.models.map((s) => [s.role, s.id]));
  $("config-line").textContent = `cheap ${byRole.cheap} · mid ${byRole.mid} · strong ${byRole.strong} · judge ${byRole.judge} · threshold ${threshold}`;
}

async function refresh() {
  const [stats, routes, failures, bench] = await Promise.all([
    api("/api/stats"), api("/api/routes?limit=200"), api("/api/failures"), api("/api/benchmarks/latest"),
  ]);
  renderCards(stats);
  renderUtilization(stats);
  renderScatter(routes);
  renderFlow(stats);
  renderFailures(failures);
  renderRoutes(routes);
  renderBenchmark(bench);
}

$("ask-form").addEventListener("submit", onSubmit);
$("query").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("ask-form").requestSubmit();
});
loadModels().then(refresh).catch((err) => { $("ask-status").textContent = `Could not reach the API: ${err.message}`; });
