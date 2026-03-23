/* app.js — 1.option live refresh & interactions */

// ─── Tab Switcher (Options Flow) ──────────────────────────────────────────────
function switchTab(type) {
  const callsPane = document.getElementById('tab-calls');
  const putsPane  = document.getElementById('tab-puts');
  const callsBtn  = document.querySelector('.tab-calls');
  const putsBtn   = document.querySelector('.tab-puts');

  if (type === 'calls') {
    callsPane.style.display = 'block';
    putsPane.style.display  = 'none';
    callsBtn.classList.add('active');
    putsBtn.classList.remove('active');
  } else {
    putsPane.style.display  = 'block';
    callsPane.style.display = 'none';
    putsBtn.classList.add('active');
    callsBtn.classList.remove('active');
  }
}

// ─── Arc Gauge Animation ──────────────────────────────────────────────────────
function animateArcGauge() {
  const arcLen = 282.7;
  document.querySelectorAll('.arc-fill-path').forEach(el => {
    const targetOffset = parseFloat(el.dataset.target || arcLen);
    el.style.strokeDashoffset = arcLen;
    setTimeout(() => {
      el.style.transition = 'stroke-dashoffset 1.3s ease';
      el.style.strokeDashoffset = targetOffset;
    }, 250);
  });
}

// ─── C/P Bar Animations ───────────────────────────────────────────────────────
function animateCPBars() {
  document.querySelectorAll('.cp-calls-fill').forEach(el => {
    const pct = el.dataset.pct;
    el.style.width = '0%';
    setTimeout(() => {
      el.style.width = pct + '%';
    }, 500);
  });
}

// ─── Sector bar animations (landing page) ────────────────────────────────────
function animateSectorBars() {
  document.querySelectorAll('.sector-bar-fill').forEach(el => {
    const target = el.style.width;
    el.style.width = '0%';
    setTimeout(() => { el.style.width = target; }, 200);
  });
}

// ─── Live Market Ticker Refresh (every 60s) ───────────────────────────────────
async function refreshMarketTicker() {
  try {
    const res = await fetch('/api/market');
    if (!res.ok) return;
    const data = await res.json();
    if (!data || !data.length) return;

    const track = document.getElementById('tickerTrack');
    if (!track) return;

    const items = [...data, ...data];
    track.innerHTML = items.map(item => `
      <span class="ticker-item">
        <span class="ticker-name">${item.name}</span>
        <span class="ticker-price">${item.price}</span>
        <span class="ticker-change ${item.positive ? 'pos' : 'neg'}">
          ${item.positive ? '+' : ''}${item.pct}%
        </span>
      </span>
    `).join('');
  } catch (e) {
    console.error('Ticker refresh error:', e);
  }
}

// ─── Dashboard Data Refresh ───────────────────────────────────────────────────
async function refreshData() {
  const btn = document.querySelector('[onclick="refreshData()"]');
  if (btn) { btn.textContent = '↻ Refreshing...'; btn.disabled = true; }

  try {
    const res = await fetch('/api/picks');
    if (res.ok) {
      const data = await res.json();
      if (data.updated_at) {
        const el = document.getElementById('updatedAt');
        if (el) el.textContent = data.updated_at;
      }
    }
  } catch (e) {
    console.error('Refresh error:', e);
  } finally {
    if (btn) {
      setTimeout(() => {
        btn.textContent = '↻ Refresh';
        btn.disabled = false;
      }, 1000);
    }
  }
}

// ─── Auto-dismiss flash messages ─────────────────────────────────────────────
document.querySelectorAll('.flash').forEach(el => {
  setTimeout(() => {
    el.style.transition = 'opacity 0.5s';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 500);
  }, 4000);
});

// ─── Auto-reload dashboard every 2 minutes for live data ─────────────────────
function startDashboardAutoReload() {
  if (!document.querySelector('.dashboard')) return;

  let countdown = 120;
  const el = document.getElementById('updatedAt');

  setInterval(() => {
    countdown--;
    if (countdown <= 0) {
      window.location.reload();
    }
  }, 1000);
}

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  animateArcGauge();
  animateCPBars();
  animateSectorBars();

  // Ticker refreshes every 30s
  setInterval(refreshMarketTicker, 30000);

  // Dashboard: full page reload every 2 minutes to show fresh data
  startDashboardAutoReload();
});
