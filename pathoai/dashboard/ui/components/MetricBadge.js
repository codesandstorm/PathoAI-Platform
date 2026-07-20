/* MetricBadge.js - Clinical Risk & Status Badges */
window.PathoMetricBadge = function({ category, label }) {
  let bg = "var(--bg-subtle)";
  let color = "var(--text-secondary)";
  let border = "var(--border-color)";

  const cat = (category || label || "").toLowerCase();
  if (cat.includes("high") || cat.includes("error") || cat.includes("failed")) {
    bg = "var(--error-subtle)";
    color = "var(--error)";
    border = "rgba(220, 38, 38, 0.2)";
  } else if (cat.includes("intermediate") || cat.includes("warning") || cat.includes("pending")) {
    bg = "var(--warning-subtle)";
    color = "var(--warning)";
    border = "rgba(217, 119, 6, 0.2)";
  } else if (cat.includes("low") || cat.includes("success") || cat.includes("passed") || cat.includes("completed")) {
    bg = "var(--success-subtle)";
    color = "var(--success)";
    border = "rgba(22, 163, 74, 0.2)";
  }

  return `
    <span style="display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; background: ${bg}; color: ${color}; border: 1px solid ${border};">
      ${label || category}
    </span>
  `;
};
