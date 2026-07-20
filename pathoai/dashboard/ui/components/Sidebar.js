/* Sidebar.js - Navigation Shell */
window.PathoSidebar = function({ activeRoute }) {
  const items = [
    { key: "dashboard", label: "Dashboard", icon: "📊" },
    { key: "cases", label: "Cases", icon: "📁" },
    { key: "viewer", label: "WSI Viewer", icon: "🔬" },
    { key: "analysis", label: "AI Analysis", icon: "⚡" },
    { key: "validation", label: "Validation", icon: "🧪" },
    { key: "experiments", label: "Experiments", icon: "🎯" },
    { key: "publication", label: "Publication Center", icon: "📄" },
    { key: "models", label: "Model Registry", icon: "🤖" },
    { key: "settings", label: "Settings", icon: "⚙️" },
  ];

  return `
    <aside style="width: 240px; background: var(--bg-card); border-right: 1px solid var(--border-color); display: flex; flex-direction: column; justify-content: space-between; padding: 16px 12px;">
      <nav style="display: flex; flex-direction: column; gap: 4px;">
        ${items.map(item => {
          const isActive = item.key === activeRoute;
          const bg = isActive ? "var(--primary-subtle)" : "transparent";
          const color = isActive ? "var(--primary)" : "var(--text-secondary)";
          const fontWeight = isActive ? "600" : "500";
          return `
            <a href="#/${item.key}" 
               style="display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-radius: var(--radius-md); background: ${bg}; color: ${color}; font-weight: ${fontWeight}; font-size: 14px; transition: all 150ms ease-in-out;"
               onclick="window.navigate('${item.key}')">
              <span>${item.icon}</span>
              <span>${item.label}</span>
            </a>
          `;
        }).join('')}
      </nav>
      
      <div style="padding: 12px; border-top: 1px solid var(--border-color); font-size: 12px; color: var(--text-secondary);">
        <div style="font-weight: 600; color: var(--text-primary);">PathoAI Core Engine</div>
        <div>Pipeline Build: <code>v0.9.8-main</code></div>
        <div style="margin-top: 4px; color: var(--success); font-weight: 500;">✓ 609 Tests Verified</div>
      </div>
    </aside>
  `;
};
