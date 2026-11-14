// DBSCAN — Discovering Music Communities
// Consumes a pre-computed trace (ANIM_DATA, injected by the template) and
// drives the same play/pause/step/reset UX as the reference animation, but
// themed around real Spotify tracks: each dot is a track, the epsilon ring
// pulses around the "active" track being visited, core tracks get a solid
// ring, border tracks get a thin ring, and noise tracks turn grey.

(function () {
  const palette = ["#1DB954", "#7F77DD", "#D85A30", "#D4537E", "#EF9F27", "#3FA7D6", "#E4B363"];
  const paletteDark = ["#0E6B31", "#3C3489", "#712B13", "#72243E", "#8A5D0E", "#1E5F7A", "#8C6A2E"];
  const NOISE_COLOR = "#4B564F";
  const ACTIVE_RING = "#1ED760";

  let data = null;
  let idx = -1;
  let playing = false;
  let timer = null;

  function init(animData) {
    data = animData;
    const svg = document.getElementById("dbscanSvg");
    svg.setAttribute("viewBox", `0 0 ${data.canvas.w} ${data.canvas.h}`);

    const pointsG = document.getElementById("dbPoints");
    data.points.forEach((p, i) => {
      const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("cx", p.x);
      c.setAttribute("cy", p.y);
      c.setAttribute("r", 7);
      c.id = "dbpt" + i;
      c.style.transition = "fill 0.25s, stroke 0.25s, r 0.25s";
      pointsG.appendChild(c);

      c.addEventListener("mouseenter", () => showHoverInfo(i));
    });

    document.getElementById("playBtn").addEventListener("click", () => (playing ? pause() : play()));
    document.getElementById("stepBtn").addEventListener("click", () => { pause(); step(); });
    document.getElementById("resetBtn").addEventListener("click", () => { pause(); idx = -1; render(); });

    render();
  }

  function showHoverInfo(i) {
    if (!data) return;
    const p = data.points[i];
    document.getElementById("infoTrack").textContent = p.track_name;
    document.getElementById("infoArtist").textContent = p.artists;
  }

  function render() {
    const epsG = document.getElementById("dbEpsRing");
    epsG.innerHTML = "";
    const counter = document.getElementById("dbCounter");
    const msg = document.getElementById("dbMsg");

    if (idx < 0) {
      data.points.forEach((p, i) => {
        const c = document.getElementById("dbpt" + i);
        c.setAttribute("fill", "#6C7D74");
        c.setAttribute("stroke", "#39433D");
        c.setAttribute("stroke-width", "1");
        c.setAttribute("r", 6);
      });
      counter.textContent = `Step 0 / ${data.history.length}`;
      msg.textContent = "Press play to start discovering music communities.";
      resetInfoPanel();
      return;
    }

    const f = data.history[idx];
    data.points.forEach((p, i) => {
      const c = document.getElementById("dbpt" + i);
      const lab = f.labels[i];
      if (lab === null || lab === undefined) {
        c.setAttribute("fill", "#39433D");
        c.setAttribute("stroke", "#232B26");
        c.setAttribute("stroke-width", "1");
        c.setAttribute("r", 6);
      } else if (lab === -1) {
        c.setAttribute("fill", NOISE_COLOR);
        c.setAttribute("stroke", "#232B26");
        c.setAttribute("stroke-width", "1");
        c.setAttribute("stroke-dasharray", "2 1.5");
        c.setAttribute("r", 6);
      } else {
        const idxColor = lab % palette.length;
        c.setAttribute("fill", palette[idxColor]);
        c.setAttribute("stroke", paletteDark[idxColor]);
        c.setAttribute("stroke-width", f.core[i] ? "2.5" : "1");
        c.setAttribute("stroke-dasharray", "");
        c.setAttribute("r", f.core[i] ? 8 : 6.5);
      }
    });

    if (f.active >= 0) {
      const ap = data.points[f.active];
      const ring = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      ring.setAttribute("cx", ap.x);
      ring.setAttribute("cy", ap.y);
      ring.setAttribute("r", data.eps_px);
      ring.setAttribute("fill", "none");
      ring.setAttribute("stroke", ACTIVE_RING);
      ring.setAttribute("stroke-width", "1.4");
      ring.setAttribute("stroke-dasharray", "5 4");
      ring.setAttribute("opacity", "0.65");
      ring.classList.add("eps-pulse");
      epsG.appendChild(ring);

      const active = document.getElementById("dbpt" + f.active);
      active.setAttribute("stroke", ACTIVE_RING);
      active.setAttribute("stroke-width", "3");

      const p = data.points[f.active];
      const lab = f.labels[f.active];
      document.getElementById("infoTrack").textContent = p.track_name;
      document.getElementById("infoArtist").textContent = p.artists;
      document.getElementById("infoCluster").textContent = (lab === null || lab === undefined) ? "—" : (lab === -1 ? "Noise" : "Community " + (lab + 1));
      document.getElementById("infoType").textContent = f.core[f.active] ? "Core track" : (lab === -1 ? "Noise" : "Border track");
    }

    counter.textContent = `Step ${idx + 1} / ${data.history.length}`;
    msg.textContent = f.msg;
  }

  function resetInfoPanel() {
    document.getElementById("infoTrack").textContent = "—";
    document.getElementById("infoArtist").textContent = "—";
    document.getElementById("infoCluster").textContent = "—";
    document.getElementById("infoType").textContent = "—";
  }

  function step() {
    if (idx >= data.history.length - 1) { pause(); return; }
    idx++;
    render();
  }

  function play() {
    playing = true;
    document.getElementById("playBtn").innerHTML = "&#10074;&#10074; Pause";
    timer = setInterval(() => { step(); if (idx >= data.history.length - 1) pause(); }, 550);
  }

  function pause() {
    playing = false;
    clearInterval(timer);
    document.getElementById("playBtn").innerHTML = "&#9658; Play";
  }

  window.initDbscanAnimation = init;
})();
