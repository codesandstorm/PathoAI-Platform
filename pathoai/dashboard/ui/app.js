/* app.js - Clinical Digital Pathology Platform Single Page Application Router & Views */

window.appState = {
  activeRoute: "dashboard",
  searchQuery: "",
  selectedCaseId: "CASE-2026-8891",
  overlays: {
    tumorBed: true,
    stroma: true,
    lymphocytes: true,
    rois: true,
    heatmap: false,
    confidence: true,
    opacity: 0.75,
  },
  cases: [
    { id: "CASE-2026-8891", patient: "PT-90412", hospital: "Mayo Clinic", scanner: "Aperio AT2", diagnosis: "Invasive Ductal Carcinoma", stil: 28.5, ci: "[24.1%, 32.9%]", category: "Intermediate", pathologist: "Dr. E. Vance, MD", status: "Completed" },
    { id: "CASE-2026-8892", patient: "PT-90413", hospital: "Johns Hopkins", scanner: "Hamamatsu NanoZoomer", diagnosis: "Triple-Negative Breast Cancer", stil: 64.2, ci: "[59.8%, 68.6%]", category: "High", pathologist: "Dr. M. Sterling, MD", status: "Completed" },
    { id: "CASE-2026-8893", patient: "PT-90414", hospital: "Memorial Sloan Kettering", scanner: "Leica GT450", diagnosis: "HER2+ Breast Carcinoma", stil: 8.4, ci: "[5.2%, 11.6%]", category: "Low", pathologist: "Dr. K. Aris, MD", status: "Completed" },
    { id: "CASE-2026-8894", patient: "PT-90415", hospital: "Cleveland Clinic", scanner: "Aperio AT2", diagnosis: "Lobular Carcinoma", stil: 18.0, ci: "[14.2%, 21.8%]", category: "Intermediate", pathologist: "Dr. E. Vance, MD", status: "Processing" },
  ],
};

window.navigate = function(route) {
  window.appState.activeRoute = route;
  window.renderApp();
};

window.onGlobalSearch = function(query) {
  window.appState.searchQuery = query;
  window.navigate("cases");
};

window.toggleOverlay = function(key) {
  window.appState.overlays[key] = !window.appState.overlays[key];
  window.renderApp();
};

window.setOpacity = function(val) {
  window.appState.overlays.opacity = parseFloat(val);
  const canvas = document.getElementById("wsi-overlay-canvas");
  if (canvas) canvas.style.opacity = val;
};

// Render Views
function renderDashboard() {
  return `
    <div class="view-container">
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px;">
        <div>
          <h1 class="h1">Command Center</h1>
          <p class="body-muted">Real-time enterprise clinical operations & AI inference tracking</p>
        </div>
        <button class="btn btn-primary" onclick="window.navigate('viewer')">⚡ Open WSI Viewer</button>
      </div>

      <!-- Key Metrics Row -->
      <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 16px; margin-bottom: 24px;">
        ${PathoCard({ contentHtml: `<div class="caption">Total Cases</div><div class="h2">1,248</div><div style="color:var(--success); font-size:12px;">↑ +12% this month</div>` })}
        ${PathoCard({ contentHtml: `<div class="caption">WSIs Processed</div><div class="h2">4,890</div><div style="color:var(--text-secondary); font-size:12px;">Avg 1.42s/slide</div>` })}
        ${PathoCard({ contentHtml: `<div class="caption">Mean sTIL Score</div><div class="h2">24.6%</div><div style="color:var(--text-secondary); font-size:12px;">n=1,248 cohort</div>` })}
        ${PathoCard({ contentHtml: `<div class="caption">GPU Cluster Load</div><div class="h2">16%</div><div style="color:var(--success); font-size:12px;">2 Active / 0 Queued</div>` })}
        ${PathoCard({ contentHtml: `<div class="caption">Clinical ICC</div><div class="h2">0.941</div><div style="color:var(--success); font-size:12px;">95% LoA [-6.5, +7.5]</div>` })}
        ${PathoCard({ contentHtml: `<div class="caption">Pipeline Status</div><div class="h2" style="color:var(--success);">Verified</div><div style="color:var(--text-secondary); font-size:12px;">609 Unit Tests</div>` })}
      </div>

      <!-- Main Section: Recent Cases & Active Jobs -->
      <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 24px;">
        ${PathoCard({
          title: "Recent Clinical Cases",
          subtitle: "Latest cases submitted for computational sTIL evaluation",
          contentHtml: `
            <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 14px;">
              <thead>
                <tr style="border-bottom: 1px solid var(--border-color); color: var(--text-secondary);">
                  <th style="padding: 10px;">Case ID</th>
                  <th style="padding: 10px;">Hospital</th>
                  <th style="padding: 10px;">Diagnosis</th>
                  <th style="padding: 10px;">sTIL %</th>
                  <th style="padding: 10px;">Category</th>
                  <th style="padding: 10px;">Action</th>
                </tr>
              </thead>
              <tbody>
                ${window.appState.cases.map(c => `
                  <tr style="border-bottom: 1px solid var(--border-color);">
                    <td style="padding: 12px 10px; font-weight: 600;">${c.id}</td>
                    <td style="padding: 12px 10px; color: var(--text-secondary);">${c.hospital}</td>
                    <td style="padding: 12px 10px;">${c.diagnosis}</td>
                    <td style="padding: 12px 10px; font-weight: 700;">${c.stil}%</td>
                    <td style="padding: 12px 10px;">${PathoMetricBadge({ category: c.category })}</td>
                    <td style="padding: 12px 10px;">
                      <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 12px;" onclick="window.appState.selectedCaseId='${c.id}'; window.navigate('viewer');">Open Viewer</button>
                    </td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          `
        })}

        ${PathoCard({
          title: "Active Processing Pipeline Jobs",
          subtitle: "Real-time streaming WSI inference tasks",
          contentHtml: `
            <div style="display: flex; flex-direction: column; gap: 16px;">
              <div style="padding: 12px; border: 1px solid var(--border-color); border-radius: var(--radius-md); background: var(--bg-subtle);">
                <div style="display: flex; justify-content: space-between; font-weight: 600; font-size: 13px;">
                  <span>CASE-2026-8894 (PT-90415)</span>
                  <span style="color: var(--primary);">Step 4/6: Cell Detection</span>
                </div>
                <div style="width: 100%; height: 6px; background: var(--border-color); border-radius: 3px; margin: 8px 0; overflow: hidden;">
                  <div style="width: 68%; height: 100%; background: var(--primary);"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-secondary);">
                  <span>Tile 412 / 600</span>
                  <span>GPU VRAM: 3.4 GB</span>
                </div>
              </div>

              <div style="padding: 12px; border: 1px solid var(--border-color); border-radius: var(--radius-md);">
                <div style="display: flex; justify-content: space-between; font-weight: 600; font-size: 13px;">
                  <span>System Provenance Version</span>
                  <span style="color: var(--success);">v1.0 Frozen</span>
                </div>
                <div style="font-size: 12px; color: var(--text-secondary); margin-top: 6px;">
                  Segmentation: <code>DeepLabV3+ (v1.2)</code><br>
                  Detection: <code>YOLO-Cell (v0.9)</code><br>
                  Scoring: <code>TIGER WG (v1.0)</code>
                </div>
              </div>
            </div>
          `
        })}
      </div>
    </div>
  `;
}

function renderCases() {
  const query = window.appState.searchQuery.toLowerCase();
  const filtered = window.appState.cases.filter(c => 
    c.id.toLowerCase().includes(query) || 
    c.patient.toLowerCase().includes(query) || 
    c.hospital.toLowerCase().includes(query) ||
    c.diagnosis.toLowerCase().includes(query)
  );

  return `
    <div class="view-container">
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px;">
        <div>
          <h1 class="h1">Enterprise Case Management</h1>
          <p class="body-muted">Filter, inspect, and export clinical breast cancer pathology cases</p>
        </div>
        <button class="btn btn-primary" onclick="alert('New Case Upload Interface Active')">+ New Clinical Case</button>
      </div>

      ${PathoCard({
        title: "All Clinical Cases",
        subtitle: `Showing ${filtered.length} of ${window.appState.cases.length} cases`,
        contentHtml: `
          <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 14px;">
            <thead>
              <tr style="border-bottom: 1px solid var(--border-color); color: var(--text-secondary);">
                <th style="padding: 12px 10px;">Case ID</th>
                <th style="padding: 12px 10px;">Patient ID</th>
                <th style="padding: 12px 10px;">Hospital</th>
                <th style="padding: 12px 10px;">Scanner</th>
                <th style="padding: 12px 10px;">Diagnosis</th>
                <th style="padding: 12px 10px;">sTIL %</th>
                <th style="padding: 12px 10px;">95% CI</th>
                <th style="padding: 12px 10px;">Risk Category</th>
                <th style="padding: 12px 10px;">Pathologist</th>
                <th style="padding: 12px 10px;">Action</th>
              </tr>
            </thead>
            <tbody>
              ${filtered.map(c => `
                <tr style="border-bottom: 1px solid var(--border-color);">
                  <td style="padding: 14px 10px; font-weight: 600;">${c.id}</td>
                  <td style="padding: 14px 10px;">${c.patient}</td>
                  <td style="padding: 14px 10px; color: var(--text-secondary);">${c.hospital}</td>
                  <td style="padding: 14px 10px; color: var(--text-secondary);">${c.scanner}</td>
                  <td style="padding: 14px 10px;">${c.diagnosis}</td>
                  <td style="padding: 14px 10px; font-weight: 700; font-size: 15px;">${c.stil}%</td>
                  <td style="padding: 14px 10px; color: var(--text-secondary); font-size: 12px;">${c.ci}</td>
                  <td style="padding: 14px 10px;">${PathoMetricBadge({ category: c.category })}</td>
                  <td style="padding: 14px 10px; color: var(--text-secondary);">${c.pathologist}</td>
                  <td style="padding: 14px 10px;">
                    <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 12px;" onclick="window.appState.selectedCaseId='${c.id}'; window.navigate('viewer');">Inspect Slide</button>
                  </td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        `
      })}
    </div>
  `;
}

function renderViewer() {
  const caseItem = window.appState.cases.find(c => c.id === window.appState.selectedCaseId) || window.appState.cases[0];
  const o = window.appState.overlays;

  return `
    <div style="display: flex; flex-direction: column; height: calc(100vh - 60px); overflow: hidden;">
      <!-- Toolbar -->
      <div style="height: 48px; background: var(--bg-card); border-bottom: 1px solid var(--border-color); display: flex; align-items: center; justify-content: space-between; padding: 0 16px;">
        <div style="display: flex; align-items: center; gap: 12px;">
          <span style="font-weight: 700; font-size: 14px;">${caseItem.id}</span>
          <span style="font-size: 12px; color: var(--text-secondary);">(${caseItem.patient} - ${caseItem.diagnosis})</span>
          ${PathoMetricBadge({ category: caseItem.category })}
        </div>

        <div style="display: flex; align-items: center; gap: 8px;">
          <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 12px;">🔍 40x Zoom</button>
          <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 12px;">📏 Scale: 100μm</button>
          <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 12px;">📍 Minimap ON</button>
        </div>
      </div>

      <!-- Main Viewer Grid -->
      <div style="flex: 1; display: grid; grid-template-columns: 240px 1fr 320px; height: calc(100% - 48px); overflow: hidden;">
        <!-- Left Panel: Layer Controls -->
        <div style="background: var(--bg-card); border-right: 1px solid var(--border-color); padding: 16px; display: flex; flex-direction: column; gap: 16px; overflow-y: auto;">
          <h4 class="h4">AI Overlay Layers</h4>
          
          <div style="display: flex; flex-direction: column; gap: 10px; font-size: 13px;">
            <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
              <input type="checkbox" ${o.tumorBed ? 'checked' : ''} onchange="window.toggleOverlay('tumorBed')">
              <span style="display: inline-block; width: 10px; height: 10px; background: rgba(220, 38, 38, 0.6); border-radius: 2px;"></span>
              Tumor Bed Mask
            </label>

            <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
              <input type="checkbox" ${o.stroma ? 'checked' : ''} onchange="window.toggleOverlay('stroma')">
              <span style="display: inline-block; width: 10px; height: 10px; background: rgba(37, 99, 235, 0.6); border-radius: 2px;"></span>
              Stroma Region Mask
            </label>

            <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
              <input type="checkbox" ${o.lymphocytes ? 'checked' : ''} onchange="window.toggleOverlay('lymphocytes')">
              <span style="display: inline-block; width: 10px; height: 10px; background: rgba(22, 163, 74, 0.9); border-radius: 50%;"></span>
              Lymphocyte Detections
            </label>

            <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
              <input type="checkbox" ${o.rois ? 'checked' : ''} onchange="window.toggleOverlay('rois')">
              <span style="display: inline-block; width: 10px; height: 10px; border: 1px solid var(--primary); border-radius: 2px;"></span>
              Tumor ROIs & Boundaries
            </label>

            <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
              <input type="checkbox" ${o.heatmap ? 'checked' : ''} onchange="window.toggleOverlay('heatmap')">
              <span style="display: inline-block; width: 10px; height: 10px; background: linear-gradient(to right, blue, red); border-radius: 2px;"></span>
              sTIL Density Heatmap
            </label>
          </div>

          <div style="margin-top: 12px; border-top: 1px solid var(--border-color); padding-top: 12px;">
            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px;">
              <span>Layer Opacity</span>
              <span><strong>${Math.round(o.opacity * 100)}%</strong></span>
            </div>
            <input type="range" min="0.1" max="1.0" step="0.05" value="${o.opacity}" style="width: 100%;" oninput="window.setOpacity(this.value)">
          </div>
        </div>

        <!-- Center Viewport: Deep Zoom Canvas Simulation -->
        <div style="position: relative; background: #0F172A; display: flex; align-items: center; justify-content: center; overflow: hidden;">
          <!-- Synthetic WSI Tissue Canvas -->
          <div style="position: relative; width: 600px; height: 450px; background: #E2E8F0; border-radius: 4px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); overflow: hidden;">
            <div style="position: absolute; inset: 0; background: radial-gradient(circle at 40% 40%, #F87171 0%, #EF4444 35%, #94A3B8 70%); opacity: 0.85;"></div>
            
            <!-- Dynamic AI Overlay SVG Canvas -->
            <svg id="wsi-overlay-canvas" style="position: absolute; inset: 0; width: 100%; height: 100%; opacity: ${o.opacity}; transition: opacity 150ms ease-in-out;">
              ${o.tumorBed ? `<polygon points="120,80 480,90 440,360 150,340" fill="rgba(220, 38, 38, 0.35)" stroke="#DC2626" stroke-width="2"/>` : ''}
              ${o.stroma ? `<polygon points="180,120 420,130 390,310 200,290" fill="rgba(37, 99, 235, 0.35)" stroke="#2563EB" stroke-width="2" stroke-dasharray="4"/>` : ''}
              ${o.lymphocytes ? `
                <circle cx="210" cy="150" r="5" fill="#16A34A"/>
                <circle cx="230" cy="180" r="5" fill="#16A34A"/>
                <circle cx="280" cy="210" r="5" fill="#16A34A"/>
                <circle cx="310" cy="240" r="5" fill="#16A34A"/>
                <circle cx="340" cy="170" r="5" fill="#16A34A"/>
                <circle cx="360" cy="290" r="5" fill="#16A34A"/>
              ` : ''}
            </svg>
          </div>

          <!-- Scale Bar & Viewport Stats Overlay -->
          <div style="position: absolute; bottom: 16px; left: 16px; background: rgba(15, 23, 42, 0.85); color: white; padding: 6px 12px; border-radius: 4px; font-size: 11px; display: flex; gap: 12px;">
            <span>Mag: 40x (MPP: 0.25 μm/px)</span>
            <span>Coords: (14,210, 8,940)</span>
            <span>FOV: 1.25 mm²</span>
          </div>
        </div>

        <!-- Right Panel: Quantitative Clinical Results -->
        <div style="background: var(--bg-card); border-left: 1px solid var(--border-color); padding: 20px; display: flex; flex-direction: column; gap: 20px; overflow-y: auto;">
          <h3 class="h3">Quantitative sTIL Results</h3>

          <div style="padding: 16px; background: var(--bg-app); border: 1px solid var(--border-color); border-radius: var(--radius-md);">
            <div class="caption">Automated sTIL Score</div>
            <div style="font-size: 36px; font-weight: 700; color: var(--primary);">${caseItem.stil}%</div>
            <div style="font-size: 12px; color: var(--text-secondary);">95% Bootstrap CI: <strong>${caseItem.ci}</strong></div>
            <div style="margin-top: 8px;">${PathoMetricBadge({ category: caseItem.category })}</div>
          </div>

          <div style="display: flex; flex-direction: column; gap: 12px; font-size: 13px;">
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-color); padding-bottom: 6px;">
              <span style="color: var(--text-secondary);">Stromal Area:</span>
              <strong>4.12 mm²</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-color); padding-bottom: 6px;">
              <span style="color: var(--text-secondary);">Stromal Lymphocytes:</span>
              <strong>1,420 cells</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-color); padding-bottom: 6px;">
              <span style="color: var(--text-secondary);">Lymphocyte Density:</span>
              <strong>344.66 cells/mm²</strong>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid var(--border-color); padding-bottom: 6px;">
              <span style="color: var(--text-secondary);">Tumor ROIs Extracted:</span>
              <strong>14 ROIs</strong>
            </div>
          </div>

          <button class="btn btn-primary" style="width: 100%;" onclick="window.navigate('publication')">📄 Export Clinical Report</button>
        </div>
      </div>
    </div>
  `;
}

function renderAnalysis() {
  const stages = [
    { name: "1. WSI Pyramid Ingestion", time: "120 ms", status: "Passed", details: "Aperio SVS reader, MPP 0.25 μm/px, Level 0 loaded" },
    { name: "2. Tissue Segmentation", time: "340 ms", status: "Passed", details: "DeepLabV3+ backbone, Dice: 0.914, Tissue coverage: 64.2%" },
    { name: "3. Tumor Bulk Extraction", time: "180 ms", status: "Passed", details: "Morphological closing & disk expansion, 14 Tumor ROIs generated" },
    { name: "4. Cell Detection", time: "420 ms", status: "Passed", details: "YOLO-Cell detector, Tile size 640px, 1,420 lymphocytes detected" },
    { name: "5. Spatial Fusion", time: "110 ms", status: "Passed", details: "Point-in-polygon mapping, 99.8% spatial precision" },
    { name: "6. Clinical sTIL Scoring", time: "140 ms", status: "Passed", details: "TIGER Working Group formula, Score: 28.5%, 95% CI: [24.1%, 32.9%]" },
  ];

  return `
    <div class="view-container">
      <div style="margin-bottom: 24px;">
        <h1 class="h1">AI Pipeline Stage Inspector</h1>
        <p class="body-muted">Inspect stage-by-stage computational metrics, VRAM footprints, and intermediate pipeline outputs</p>
      </div>

      <div style="display: flex; flex-direction: column; gap: 16px;">
        ${stages.map(s => `
          ${PathoCard({
            contentHtml: `
              <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                  <h4 class="h4">${s.name}</h4>
                  <p style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">${s.details}</p>
                </div>
                <div style="display: flex; align-items: center; gap: 16px;">
                  <span style="font-size: 13px; color: var(--text-secondary);">Execution Time: <strong>${s.time}</strong></span>
                  ${PathoMetricBadge({ category: "success", label: s.status })}
                </div>
              </div>
            `
          })}
        `).join('')}
      </div>
    </div>
  `;
}

function renderValidation() {
  return `
    <div class="view-container">
      <div style="margin-bottom: 24px;">
        <h1 class="h1">Scientific Validation Dashboard</h1>
        <p class="body-muted">Statistical agreement analysis, Bland–Altman limits, ICC, and error evaluation</p>
      </div>

      <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px;">
        ${PathoCard({ contentHtml: `<div class="caption">Intraclass Correlation (ICC)</div><div class="h2">0.941</div><div style="color:var(--success); font-size:12px;">Excellent Clinical Agreement</div>` })}
        ${PathoCard({ contentHtml: `<div class="caption">Bland–Altman Mean Bias</div><div class="h2">0.52%</div><div style="color:var(--text-secondary); font-size:12px;">95% LoA: [-6.5%, +7.5%]</div>` })}
        ${PathoCard({ contentHtml: `<div class="caption">Pearson Correlation (r)</div><div class="h2">0.948</div><div style="color:var(--text-secondary); font-size:12px;">p < 0.0001</div>` })}
        ${PathoCard({ contentHtml: `<div class="caption">Mean Absolute Error (MAE)</div><div class="h2">3.42%</div><div style="color:var(--text-secondary); font-size:12px;">RMSE: 4.61%</div>` })}
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
        ${PathoCard({
          title: "Bland–Altman Agreement Plot",
          subtitle: "AI vs Pathologist sTIL Score Differences (%)",
          contentHtml: `
            <div style="height: 240px; background: var(--bg-subtle); border-radius: var(--radius-md); display: flex; align-items: center; justify-content: center; color: var(--text-secondary); font-size: 13px;">
              [Bland–Altman Agreement Rendered Canvas: Bias +0.52%, LoA -6.5% to +7.5%]
            </div>
          `
        })}

        ${PathoCard({
          title: "AI vs Pathologist Scatter Plot",
          subtitle: "Correlation & 1:1 Identity Line",
          contentHtml: `
            <div style="height: 240px; background: var(--bg-subtle); border-radius: var(--radius-md); display: flex; align-items: center; justify-content: center; color: var(--text-secondary); font-size: 13px;">
              [Scatter Plot Rendered Canvas: r = 0.948, R² = 0.898]
            </div>
          `
        })}
      </div>
    </div>
  `;
}

function renderExperiments() {
  return `
    <div class="view-container">
      <div style="margin-bottom: 24px;">
        <h1 class="h1">Experiment Center & ML Leaderboard</h1>
        <p class="body-muted">Weights & Biases style model run provenance, git commits, seeds, and leaderboard</p>
      </div>

      ${PathoCard({
        title: "Model Run Leaderboard",
        subtitle: "Ranked by Intraclass Correlation Coefficient (ICC)",
        contentHtml: `
          <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 14px;">
            <thead>
              <tr style="border-bottom: 1px solid var(--border-color); color: var(--text-secondary);">
                <th style="padding: 10px;">Experiment ID</th>
                <th style="padding: 10px;">Git Commit</th>
                <th style="padding: 10px;">Segmentation Backbone</th>
                <th style="padding: 10px;">Detection Model</th>
                <th style="padding: 10px;">Dice</th>
                <th style="padding: 10px;">ICC</th>
                <th style="padding: 10px;">MAE</th>
                <th style="padding: 10px;">Status</th>
              </tr>
            </thead>
            <tbody>
              <tr style="border-bottom: 1px solid var(--border-color);">
                <td style="padding: 12px 10px; font-weight: 600;">exp_nature_med_001</td>
                <td style="padding: 12px 10px; font-family: var(--font-mono);">dfbafd4</td>
                <td style="padding: 12px 10px;">DeepLabV3+ (v1.2)</td>
                <td style="padding: 12px 10px;">YOLO-Cell (v0.9)</td>
                <td style="padding: 12px 10px;">0.914</td>
                <td style="padding: 12px 10px; font-weight: 700; color: var(--primary);">0.941</td>
                <td style="padding: 12px 10px;">3.42%</td>
                <td style="padding: 12px 10px;">${PathoMetricBadge({ category: "success", label: "Completed" })}</td>
              </tr>
              <tr style="border-bottom: 1px solid var(--border-color);">
                <td style="padding: 12px 10px; font-weight: 600;">exp_baseline_002</td>
                <td style="padding: 12px 10px; font-family: var(--font-mono);">c617716</td>
                <td style="padding: 12px 10px;">UNet Standard</td>
                <td style="padding: 12px 10px;">Faster R-CNN</td>
                <td style="padding: 12px 10px;">0.842</td>
                <td style="padding: 12px 10px;">0.865</td>
                <td style="padding: 12px 10px;">6.10%</td>
                <td style="padding: 12px 10px;">${PathoMetricBadge({ category: "success", label: "Completed" })}</td>
              </tr>
            </tbody>
          </table>
        `
      })}
    </div>
  `;
}

function renderPublication() {
  return `
    <div class="view-container">
      <div style="margin-bottom: 24px;">
        <h1 class="h1">Publication Center & Exporters</h1>
        <p class="body-muted">One-click generation of Nature Medicine / MedIA format LaTeX tables and reports</p>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
        ${PathoCard({
          title: "LaTeX Table Export (Table 3: Clinical Agreement)",
          actionHtml: `<button class="btn btn-primary" onclick="alert('LaTeX Code Copied to Clipboard!')">Copy LaTeX</button>`,
          contentHtml: `
            <pre style="background: #0F172A; color: #F8FAFC; padding: 16px; border-radius: var(--radius-md); font-family: var(--font-mono); font-size: 12px; overflow-x: auto;">
\\begin{table}[htbp]
\\centering
\\caption{Clinical sTIL Scoring Agreement against Pathologist Ground Truth.}
\\label{tab:clinical_agreement}
\\begin{tabular}{lc}
\\hline
\\textbf{Metric} & \\textbf{Value} \\\\
\\hline
Intraclass Correlation (ICC) & \\textbf{0.9410} \\\\
Pearson Correlation ($r$) & \\textbf{0.9480} \\\\
Spearman Correlation ($\\rho$) & \\textbf{0.9320} \\\\
Mean Absolute Error (MAE) & 3.42\\% \\\\
Bland--Altman Bias & 0.52\\% \\\\
Bland--Altman 95\\% LoA & [-6.50\\%, 7.50\\%] \\\\
\\hline
\\end{tabular}
\\end{table}
            </pre>
          `
        })}

        ${PathoCard({
          title: "Download Clinical PDF Report Package",
          subtitle: "Complete clinical evaluation report for CASE-2026-8891",
          contentHtml: `
            <div style="padding: 20px; border: 1px dashed var(--border-color); border-radius: var(--radius-md); text-align: center;">
              <p style="font-size: 14px; color: var(--text-secondary); margin-bottom: 16px;">
                Generated PDF package includes WSI thumbnails, sTIL percentage breakdown, 95% bootstrap confidence bounds, and pathologist signature blocks.
              </p>
              <button class="btn btn-primary" onclick="alert('Downloading PDF Clinical Report Package...')">📥 Download PDF Report</button>
            </div>
          `
        })}
      </div>
    </div>
  `;
}

// Master Render App Function
window.renderApp = function() {
  const root = document.getElementById("app-root");
  if (!root) return;

  const route = window.appState.activeRoute;
  let viewHtml = "";

  if (route === "dashboard") viewHtml = renderDashboard();
  else if (route === "cases") viewHtml = renderCases();
  else if (route === "viewer") viewHtml = renderViewer();
  else if (route === "analysis") viewHtml = renderAnalysis();
  else if (route === "validation") viewHtml = renderValidation();
  else if (route === "experiments") viewHtml = renderExperiments();
  else if (route === "publication") viewHtml = renderPublication();
  else viewHtml = renderDashboard();

  root.innerHTML = `
    <div class="app-shell">
      ${window.PathoSidebar({ activeRoute: route })}
      <div class="main-content">
        ${window.PathoNavbar({ activeCase: window.appState.selectedCaseId })}
        <main style="flex: 1; overflow-y: auto;">
          ${viewHtml}
        </main>
      </div>
    </div>
  `;
};

// Initialize App
document.addEventListener("DOMContentLoaded", () => {
  window.renderApp();
});
