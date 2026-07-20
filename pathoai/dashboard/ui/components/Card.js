/* Card.js - Standard Clinical Card Container */
window.PathoCard = function({ title, subtitle, actionHtml, contentHtml, style = "" }) {
  return `
    <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: var(--radius-lg); box-shadow: var(--shadow-subtle); padding: 20px; ${style}">
      ${title ? `
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; border-bottom: 1px solid var(--border-color); padding-bottom: 12px;">
          <div>
            <h3 style="font-size: 16px; font-weight: 600; color: var(--text-primary); margin: 0;">${title}</h3>
            ${subtitle ? `<p style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">${subtitle}</p>` : ''}
          </div>
          ${actionHtml ? `<div>${actionHtml}</div>` : ''}
        </div>
      ` : ''}
      <div>${contentHtml}</div>
    </div>
  `;
};
