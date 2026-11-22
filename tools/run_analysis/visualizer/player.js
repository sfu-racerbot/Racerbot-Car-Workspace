"use strict";

const runs = JSON.parse(document.getElementById("runs").textContent);
const $ = id => document.getElementById(id);
const panels = $("panels"), scrub = $("scrub");
const selected = [0, Math.min(1, runs.length - 1)];
let compare = runs.length > 1, views = [], duration = 0, time = 0, frame = null, lastTick = 0;

function svg(parent, tag, attrs, label) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (label !== undefined) el.textContent = label;
  parent.append(el);
  return el;
}

function extent(series, col, initial = [Infinity, -Infinity]) {
  let [min, max] = initial;
  for (const row of series) {
    min = Math.min(min, row[col]);
    max = Math.max(max, row[col]);
  }
  return [min, max];
}

function clock(t) {
  const tenths = Math.floor(t * 10 + 1e-7);
  return Math.floor(tenths / 600) + ":" + String(Math.floor(tenths / 10) % 60).padStart(2, "0") + "." + tenths % 10;
}

function title(run) {
  const stamp = run.name.replace(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$/, "$1-$2-$3 · $4:$5:$6");
  return [run.car.toUpperCase(), stamp].filter(Boolean).join(" / ");
}

// Topics have independent receive times. Do not borrow a future sample or
// carry a value past the topic's last observation.
function sample(series, t) {
  if (!series.length || t < series[0][0] || t > series.at(-1)[0]) return null;
  let lo = 0, hi = series.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (series[mid][0] <= t) lo = mid + 1;
    else hi = mid;
  }
  return series[lo - 1];
}

function drawOdometry(parent, rows, point, attrs) {
  if (!rows.length) return;
  svg(parent, "path", {d: odometryPath(rows, point), fill: "none", "stroke-linejoin": "round", ...attrs});
}

function trajectory(plot, run, ext, scale) {
  const cx = (ext.x[0] + ext.x[1]) / 2, cy = (ext.y[0] + ext.y[1]) / 2;
  const x = m => 250 + (m - cx) * scale, y = m => 94 - (m - cy) * scale;
  const step = 2 ** Math.ceil(Math.log2(38 / scale));
  for (let m = Math.ceil((cx - 250 / scale) / step) * step; x(m) < 500; m += step) {
    svg(plot, "line", {x1: x(m), x2: x(m), y1: 0, y2: 188, stroke: "#304b43"});
  }
  for (let m = Math.ceil((cy - 94 / scale) / step) * step; y(m) > 0; m += step) {
    svg(plot, "line", {x1: 0, x2: 500, y1: y(m), y2: y(m), stroke: "#304b43"});
  }
  drawOdometry(plot, run.odom, r => [x(r[1]), y(r[2])], {stroke: "var(--accent)", "stroke-width": 1.7});
  svg(plot, "circle", {cx: x(0), cy: y(0), r: 5.5, fill: "#152d28", stroke: "#fff", "stroke-width": 1.5});
  svg(plot, "path", {d: "M9 158v15h15m-15 -15l-2 4m17 11l-4 -2", fill: "none", stroke: "#b5cabe"});
  svg(plot, "text", {x: 5, y: 155, fill: "#b5cabe"}, "Y");
  svg(plot, "text", {x: 28, y: 177, fill: "#b5cabe"}, "X");
  const len = step * scale;
  svg(plot, "path", {d: "M" + (485-len) + " 173v4h" + len + "v-4", fill: "none", stroke: "#c3d3cb"});
  svg(plot, "text", {x: 485, y: 170, fill: "#c3d3cb", "text-anchor": "end"}, step + " m");
  const dot = svg(plot, "circle", {r: 4.2, stroke: "#142e29", "stroke-width": 1.3, fill: "white"});
  return row => {
    dot.style.display = row ? "" : "none";
    if (row) {
      dot.setAttribute("cx", x(row[1]));
      dot.setAttribute("cy", y(row[2]));
    }
  };
}

function chart(plot, series, bounds) {
  const left = 36, right = 493, top = 9, bottom = 78;
  const x = t => left + t / (duration || 1) * (right - left);
  const y = v => bottom - (v - bounds[0]) / (bounds[1]-bounds[0]) * (bottom-top);
  plot.setAttribute("preserveAspectRatio", "none");
  for (const v of [bounds[0], (bounds[0] + bounds[1])/2, bounds[1]]) {
    svg(plot, "line", {x1: left, x2: right, y1: y(v), y2: y(v), stroke: "#dbe3dd"});
    svg(plot, "text", {x: left - 5, y: y(v)+3, "text-anchor": "end", fill: "#566861"}, String(+v.toFixed(1)));
  }
  for (const f of [0, .5, 1]) {
    svg(plot, "text", {x: x(duration*f), y: 95, fill: "#566861", "text-anchor": f === 0 ? "start" : f === 1 ? "end" : "middle"},
        (duration*f).toFixed(1) + " s");
  }
  for (const [rows, col, kind] of series) {
    if (kind === "odom") {
      drawOdometry(plot, rows, r => [x(r[0]), y(r[col])], {stroke: "var(--trace)", "stroke-width": 1.8});
    } else {
      const attrs = kind === "target" ? {stroke: "#657382", "stroke-dasharray": "4 3"} : {stroke: "var(--trace)"};
      svg(plot, "path", {d: commandPath(rows, col, x, y), fill: "none", "stroke-width": 1.5, ...attrs});
    }
  }
  if (!series.some(([rows]) => rows.length)) {
    svg(plot, "text", {x: 260, y: 46, "text-anchor": "middle", fill: "#516760"}, "No command samples");
  }
  const cursor = svg(plot, "line", {y1: top, y2: bottom, stroke: "#263d34", "stroke-dasharray": "2 2"});
  plot.addEventListener("click", e => {
    const box = plot.getBoundingClientRect();
    seek(duration * ((e.clientX - box.left) / box.width * 500 - left) / (right - left));
  });
  return t => {
    cursor.setAttribute("x1", x(t)); cursor.setAttribute("x2", x(t));
  };
}

function panel(run, i, ext, scale, speedRange, steerRange) {
  const root = $("panel").content.firstElementChild.cloneNode(true);
  const q = sel => root.querySelector(sel);
  const letter = i ? "B" : "A";
  if (i) root.classList.add("b");
  q(".badge").textContent = letter;
  q(".source").textContent = run.source;
  if (!compare) q(".map-note").textContent = "Wheel odometry relative to the first recorded pose.";
  q(".late").hidden = !run.startedLate;
  q(".meta").textContent = run.algorithms.length ? run.algorithms.join(", ") : run.name;
  const menu = q(".run-select");
  menu.setAttribute("aria-label", "Recording " + letter);
  runs.forEach((r, n) => menu.add(new Option(title(r), n)));
  menu.value = selected[i];
  menu.onchange = () => {
    selected[i] = Number(menu.value); refresh(time);
    panels.querySelectorAll(".run-select")[i].focus();
  };
  for (const [cls, value, unit] of [["duration", run.duration, "s"], ["distance", completeDistance(run), "m"], ["peak", run.peak, "m/s"]]) {
    const strong = q("." + cls), small = document.createElement("small");
    strong.textContent = (value === null ? "—" : value.toFixed(1)) + " ";
    small.textContent = value === null ? "Incomplete · odometry gap" : unit;
    strong.append(small);
  }
  const map = trajectory(q(".map"), run, ext, scale);
  q(".cmd-key").hidden = !run.cmdTopic;
  q(".topic").textContent = run.cmdTopic ? run.cmdTopic + " · blank after " + COMMAND_MAX_AGE + " s without a sample" : "No /drive or /ackermann_cmd";
  q(".speed").setAttribute("aria-label", "Speed (m/s) over elapsed time");
  const speed = chart(q(".speed"), [[run.cmd, 1, "target"], [run.odom, 3, "odom"]], speedRange);
  const steer = chart(q(".steer"), [[run.cmd, 2, "angle"]], steerRange);
  const speedVal = q(".live-speed"), steerVal = q(".live-steer"), state = q(".state"), poseState = q(".pose-state");
  const cmdSpeedVal = q(".live-command");
  panels.append(root);
  return t => {
    const ended = t > run.duration;
    const prev = sample(run.odom, t), odom = prev && t - prev[0] <= ODOM_MAX_AGE ? prev : null;
    const cmd = sample(run.cmd, t), joy = sample(run.joy, t);
    map(odom); speed(t); steer(t);
    poseState.textContent = odom ? "● Cursor" : t < run.odom[0][0] ? "Waiting for odometry" :
      t > run.odom.at(-1)[0] ? "Odometry ended" : "Odometry gap";
    speedVal.textContent = (odom ? odom[3].toFixed(2) : "—") + " m/s";
    cmdSpeedVal.textContent = commandLabel(cmd, t, 1, " m/s");
    steerVal.textContent = commandLabel(cmd, t, 2, "°");
    state.textContent = ended ? "Run ended" : joystickLabel(joy, t);
  };
}

function refresh(at = 0) {
  pause();
  const active = selected.slice(0, compare ? 2 : 1).map(n => runs[n]);
  duration = Math.max(...active.map(r => r.duration));
  const box = active.map(r => ({x: extent(r.odom, 1), y: extent(r.odom, 2)}));
  const scale = Math.min(...box.map(({x, y}) => Math.min(456 / Math.max(1, x[1]-x[0]), 156 / Math.max(1, y[1]-y[0]))));
  let speed = [0, 1], steer = [0, 0];
  for (const r of active) {
    speed = extent(r.cmd, 1, extent(r.odom, 3, speed));
    steer = extent(r.cmd, 2, steer);
  }
  speed = [Math.floor(speed[0]), Math.ceil(speed[1])];
  const deg = Math.max(1, Math.ceil(Math.max(-steer[0], steer[1])));
  panels.replaceChildren();
  panels.classList.toggle("single", !compare);
  views = active.map((r, i) => panel(r, i, box[i], scale, speed, [-deg, deg]));
  $("compare").disabled = runs.length === 1;
  $("single").setAttribute("aria-pressed", String(!compare));
  $("compare").setAttribute("aria-pressed", String(compare));
  $("end").textContent = "/ " + clock(duration);
  scrub.max = duration; scrub.disabled = !duration; $("play").disabled = !duration;
  seek(at);
}

function seek(t) {
  time = Math.max(0, Math.min(duration, t));
  scrub.value = time;
  scrub.setAttribute("aria-valuetext", time.toFixed(2) + " seconds");
  $("elapsed").textContent = clock(time);
  views.forEach(update => update(time));
}

function pause() {
  cancelAnimationFrame(frame); frame = null; $("play").textContent = "▶ Play";
}

function tick(now) {
  seek(time + (now-lastTick) / 1000 * parseFloat($("rate").value));
  lastTick = now;
  if (time >= duration) pause(); else frame = requestAnimationFrame(tick);
}

$("play").onclick = () => {
  if (frame !== null) return pause();
  if (time >= duration) seek(0);
  $("play").textContent = "Ⅱ Pause"; lastTick = performance.now(); frame = requestAnimationFrame(tick);
};
scrub.oninput = () => seek(Number(scrub.value));
scrub.onkeydown = e => {
  const direction = {ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1}[e.key];
  if (!direction) return;
  e.preventDefault();
  seek((Math.round(time * 10) + direction) / 10);
};
$("single").onclick = () => { compare = false; refresh(time); };
$("compare").onclick = () => { compare = true; refresh(time); };
refresh();
