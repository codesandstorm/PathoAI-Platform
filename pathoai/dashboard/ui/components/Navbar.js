/* Navbar.js - Top Clinical Header */
window.PathoNavbar = function({ activeCase, onSearch }) {
  return `
    <header style="height: 60px; border-bottom: 1px solid var(--border-color); background: var(--bg-card); display: flex; align-items: center; justify-content: space-between; padding: 0 24px; z-index: 10;">
      <div style="display: flex; align-items: center; gap: 16px;">
        <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 16px; color: var(--text-primary);">
          <span style="display: inline-block; width: 12px; height: 12px; background: var(--primary); border-radius: 2px;"></span>
          PathoAI Platform <span style="font-weight: 400; font-size: 12px; color: var(--text-secondary); background: var(--bg-subtle); padding: 2px 6px; border-radius: 4px;">v1.0 Clinical</span>
        </div>
        <span style="color: var(--border-color);">|</span>
        <div style="display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--text-secondary);">
          <span>Hospital:</span>
          <strong style="color: var(--text-primary);">Mayo Clinic Comprehensive Cancer Center</strong>
        </div>
      </div>
      
      <div style="display: flex; align-items: center; gap: 16px;">
        <input 
          type="text" 
          placeholder="Search Patient ID, Case #, or Slide..." 
          class="input" 
          style="width: 320px;" 
          onkeyup="if(event.key==='Enter') window.onGlobalSearch(this.value)"
        />
        <div style="display: flex; align-items: center; gap: 8px;">
          <span style="width: 8px; height: 8px; background: var(--success); border-radius: 50%; display: inline-block;"></span>
          <span style="font-size: 12px; color: var(--text-secondary);">GPU Cluster Online (84% Free)</span>
        </div>
      </div>
    </header>
  `;
};
