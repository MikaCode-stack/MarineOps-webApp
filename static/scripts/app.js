const API = "http://localhost:8000";

// Particles
const pc = document.getElementById("particles");
for (let i = 0; i < 25; i++) {
  const p = document.createElement("div");
  p.className = "particle";
  p.style.cssText = `
      left: ${Math.random() * 100}%;
      width: ${Math.random() * 3 + 1}px;
      height: ${Math.random() * 3 + 1}px;
      animation-duration: ${Math.random() * 15 + 8}s;
      animation-delay: ${Math.random() * 10}s;
      opacity: ${Math.random() * 0.5};
    `;
  pc.appendChild(p);
}

// Page navigation
function showPage(name) {
  document
    .querySelectorAll(".page")
    .forEach((p) => p.classList.remove("active"));
  document
    .querySelectorAll(".nav-link")
    .forEach((n) => n.classList.remove("active"));
  document.getElementById("page-" + name).classList.add("active");
  event.target.classList.add("active");
  if (name === "history") loadHistory();
}

// Toast
function toast(msg, type = "success") {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = "show " + type;
  setTimeout(() => (t.className = ""), 3000);
}

// File preview
const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) showPreview(fileInput.files[0]);
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () =>
  dropZone.classList.remove("dragover"),
);
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const f = e.dataTransfer.files[0];
  if (f) {
    fileInput.files = e.dataTransfer.files;
    showPreview(f);
  }
});

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("preview-img").src = e.target.result;
    document.getElementById("preview-container").style.display = "block";
  };
  reader.readAsDataURL(file);
}

// Validate then Detect
async function runDetect() {
  const file = fileInput.files[0];
  const location = document.getElementById("location-input").value.trim();
  const conf = document.getElementById("conf-input").value.trim() || "0.25";

  if (!file) return toast("Please select an image", "error");
  if (!location) return toast("Please enter a location name", "error");

  const btn = document.getElementById("detect-btn");
  const btnText = document.getElementById("detect-btn-text");
  btn.disabled = true;
  btnText.innerHTML = '<span class="spinner"></span> Validating...';

  try {
    // Step 1: Validate
    const vForm = new FormData();
    vForm.append("file", file);
    const vRes = await fetch(`${API}/validate`, {
      method: "POST",
      body: vForm,
    });
    if (!vRes.ok) {
      const err = await vRes.json();
      throw new Error(
        Array.isArray(err.detail) ? err.detail.join(" ") : err.detail,
      );
    }

    // Step 2: Detect
    btnText.innerHTML = '<span class="spinner"></span> Analysing...';
    const dForm = new FormData();
    dForm.append("file", file);
    dForm.append("location", location);
    const dRes = await fetch(`${API}/detect`, {
      method: "POST",
      body: dForm,
    });
    if (!dRes.ok) throw new Error("Detection failed");

    const data = await dRes.json();
    renderResults(data);
    toast("Detection complete!", "success");
  } catch (e) {
    toast(e.message || "Something went wrong", "error");
  } finally {
    btn.disabled = false;
    btnText.innerHTML = "🔍 Run Detection";
  }
}

// Render results
function renderResults(data) {
  document.getElementById("results-panel").style.display = "block";
  document
    .getElementById("results-panel")
    .scrollIntoView({ behavior: "smooth", block: "start" });

  document.getElementById("res-psi").textContent =
    data.image_psi ?? data.psi_score ?? "—";
  document.getElementById("res-location-psi").textContent =
    data.location_psi ?? "—";
  document.getElementById("res-objects").textContent = data.total_objects;
  document.getElementById("res-location").textContent = data.location;
  document.getElementById("res-img-count").textContent =
    (data.total_images_at_location ?? 1) + " image(s)";

  const sev = document.getElementById("res-severity");
  sev.textContent = data.severity;
  sev.className = `severity-badge severity-${data.severity}`;

  // Class breakdown bars
  const bars = document.getElementById("class-bars");
  const counts = data.class_counts;
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  bars.innerHTML = Object.entries(counts)
    .map(
      ([cls, cnt]) => `
      <div class="class-bar-item">
        <div class="class-bar-label">
          <span>${cls.replace(/_/g, " ")}</span>
          <span style="color:var(--accent-cyan);font-family:'Space Mono',monospace;font-size:12px;">${cnt}</span>
        </div>
        <div class="class-bar-track">
          <div class="class-bar-fill" style="width:${((cnt / total) * 100).toFixed(1)}%"></div>
        </div>
      </div>
    `,
    )
    .join("");

  // Detection list
  const list = document.getElementById("detection-list");
  list.innerHTML = data.detections
    .map(
      (d) => `
      <div class="detection-item">
        <span class="detection-class">${d.class.replace(/_/g, " ")}</span>
        <span class="detection-meta">
          <span class="detection-conf">conf ${d.confidence}</span>
          <span class="detection-psi">PSI ${d.psi_contribution}</span>
        </span>
      </div>
    `,
    )
    .join("");
}

//severity helper
function getSeverity(psi) {
  if (psi === null || psi === undefined) return "unknown";
  if (psi >= 15) return "critical";
  if (psi >= 10) return "high";
  if (psi >= 5) return "medium";
  return "low";
}

let filterDebounce = null;
function onFilterInput() {
  clearTimeout(filterDebounce);
  filterDebounce = setTimeout(() => loadHistory(), 350); //350 ms between each type
}
// History containers loader
async function loadHistory() {
  const loc = document.getElementById("location-filter").value.trim();
  const url = loc
    ? `${API}/analytics?location=${encodeURIComponent(loc)}`
    : `${API}/analytics`;

  const photoTbody = document.getElementById("photo-tbody");
  const locationTbody = document.getElementById("location-tbody");
  const photoCount = document.getElementById("photo-count");
  const locationCount = document.getElementById("location-count");

  const loadingCell = (cols) =>
    `<tr><td colspan="${cols}" style="text-align:center;color:var(--text-muted);padding:40px;">Loading...</td></tr>`;
  const errorCell = (cols) =>
    `<tr><td colspan="${cols}" style="text-align:center;color:var(--accent-danger);padding:40px;">Failed to load</td></tr>`;
  const emptyCell = (cols) =>
    `<tr><td colspan="${cols}" style="text-align:center;color:var(--text-muted);padding:40px;">No records found</td></tr>`;

  photoTbody.innerHTML = loadingCell(5);
  locationTbody.innerHTML = loadingCell(5);

  try {
    const res = await fetch(url);
    const data = await res.json();

    if (data.message) {
      photoTbody.innerHTML = emptyCell(5);
      locationTbody.innerHTML = emptyCell(5);
      return;
    }

    // TABLE 1: Photo history from psi_trend
    const photoRows = [...(data.psi_trend || [])].reverse();
    photoCount.textContent = `${photoRows.length} record${photoRows.length !== 1 ? "s" : ""}`;

    if (!photoRows.length) {
      photoTbody.innerHTML = emptyCell(5);
    } else {
      photoTbody.innerHTML = photoRows
        .map((r) => {
          const conf =
            r.avg_confidence != null
              ? `${(r.avg_confidence * 100).toFixed(1)}%`
              : "—";
          return `
          <tr>
            <td>${r.location ?? "—"}</td>
            <td style="font-family:'Space Mono',monospace; color:var(--accent-cyan);">
              ${r.psi_score?.toFixed(4) ?? "—"}
            </td>
            <td style="font-family:'Space Mono',monospace;">
              ${r.total_objects ?? 0}
            </td>
            <td style="color:var(--text-muted); font-size:13px;">
              ${r.timestamp ? new Date(r.timestamp).toLocaleString() : "—"}
            </td>
            <td style="font-family:'Space Mono',monospace; color:var(--accent-cyan);">
              ${conf}
            </td>
          </tr>
        `;
        })
        .join("");
    }

    // TABLE 2: Location history from location_stats
    const locationRows = data.location_stats || [];
    locationCount.textContent = `${locationRows.length} location${locationRows.length !== 1 ? "s" : ""}`;

    if (!locationRows.length) {
      locationTbody.innerHTML = emptyCell(5);
    } else {
      locationTbody.innerHTML = locationRows
        .map((s) => {
          const sev = getSeverity(s.avg_psi);
          return `
          <tr>
            <td>${s.location ?? "—"}</td>
            <td style="font-family:'Space Mono',monospace; color:var(--accent-cyan);">
              ${s.avg_psi?.toFixed(4) ?? "—"}
            </td>
            <td style="font-family:'Space Mono',monospace;">
              ${s.total_objects ?? 0}
            </td>
            <td>
              <span class="severity-badge severity-${sev}">${sev}</span>
            </td>
            <td style="font-family:'Space Mono',monospace;">
              ${s.total_scans ?? 0}
            </td>
          </tr>
        `;
        })
        .join("");
    }
  } catch (e) {
    console.error("loadHistory error:", e);
    photoTbody.innerHTML = errorCell(5);
    locationTbody.innerHTML = errorCell(5);
  }
}

// Dashboard iframe
function loadDash() {
  document.getElementById("dash-placeholder").style.display = "none";
  const frame = document.getElementById("dash-frame");
  frame.src = `${API}/dashboard/`;
  frame.style.display = "block";
}
