// Spotify ML Lab — shared frontend helpers

/**
 * Renders a Plotly figure (already serialized with fig.to_json() server-side)
 * into the element with the given id.
 */
function renderPlot(elId, figJsonString, extraConfig) {
  const el = document.getElementById(elId);
  if (!el || !figJsonString) return;
  const fig = JSON.parse(figJsonString);
  const config = Object.assign({ responsive: true, displaylogo: false }, extraConfig || {});
  Plotly.newPlot(el, fig.data, fig.layout, config);
}

// Generate a small set of equalizer bars with randomized heights/durations
function buildEqualizer(containerId, barCount) {
  const el = document.getElementById(containerId);
  if (!el) return;
  barCount = barCount || 18;
  for (let i = 0; i < barCount; i++) {
    const bar = document.createElement("div");
    bar.className = "bar";
    const duration = (0.7 + Math.random() * 0.9).toFixed(2);
    const delay = (Math.random() * 0.8).toFixed(2);
    bar.style.animationDuration = duration + "s";
    bar.style.animationDelay = delay + "s";
    bar.style.height = (15 + Math.random() * 30) + "%";
    el.appendChild(bar);
  }
}

// Floating music-note particles for hero sections
function buildNoteParticles(containerId, count) {
  const el = document.getElementById(containerId);
  if (!el) return;
  const glyphs = ["\u266A", "\u266B", "\u2669"];
  count = count || 10;
  for (let i = 0; i < count; i++) {
    const span = document.createElement("span");
    span.textContent = glyphs[Math.floor(Math.random() * glyphs.length)];
    span.style.left = Math.random() * 100 + "%";
    span.style.animationDuration = (6 + Math.random() * 6) + "s";
    span.style.animationDelay = (Math.random() * 6) + "s";
    span.style.fontSize = (12 + Math.random() * 14) + "px";
    el.appendChild(span);
  }
}

document.addEventListener("DOMContentLoaded", function () {
  buildEqualizer("heroEqualizer", 22);
  buildNoteParticles("heroNotes", 12);
});
