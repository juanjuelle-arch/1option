/* app.js — 1.option live refresh & interactions */

// ─── Tab Switcher (Options Flow) ──────────────────────────────────────────────
function switchTab(type) {
  const callsPane = document.getElementById('tab-calls');
  const putsPane  = document.getElementById('tab-puts');
  const callsBtn  = document.querySelector('.tab-calls');
  const putsBtn   = document.querySelector('.tab-puts');
  if (!callsPane || !putsPane) return;

  if (type === 'calls') {
    callsPane.style.display = 'block';
    putsPane.style.display  = 'none';
    if (callsBtn) callsBtn.classList.add('active');
    if (putsBtn)  putsBtn.classList.remove('active');
  } else {
    putsPane.style.display  = 'block';
    callsPane.style.display = 'none';
    if (putsBtn)  putsBtn.classList.add('active');
    if (callsBtn) callsBtn.classList.remove('active');
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

// ─── Live Market Ticker Refresh (every 30s) ──────────────────────────────────
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

// ─── Auto-dismiss flash messages ─────────────────────────────────────────────
document.querySelectorAll('.flash').forEach(el => {
  setTimeout(() => {
    el.style.transition = 'opacity 0.5s';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 500);
  }, 4000);
});

// ─── Dashboard auto-reload with visible countdown ───────────────────────────
function startDashboardAutoReload() {
  if (!document.querySelector('.dashboard')) return;

  let countdown = 120;
  const updatedEl = document.getElementById('updatedAt');

  const reloadInterval = setInterval(() => {
    countdown--;
    if (updatedEl && countdown <= 15) {
      updatedEl.textContent = 'Refreshing in ' + countdown + 's';
    }
    if (countdown <= 0) {
      clearInterval(reloadInterval);
      window.location.reload();
    }
  }, 1000);

  // Clean up on page leave
  window.addEventListener('beforeunload', () => clearInterval(reloadInterval));
}

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  animateArcGauge();
  animateCPBars();
  animateSectorBars();

  // Mobile hamburger toggle
  const navToggle = document.getElementById('navToggle');
  const navLinks = document.querySelector('.nav-links');
  if (navToggle && navLinks) {
    navToggle.addEventListener('click', () => {
      navLinks.classList.toggle('open');
    });
  }

  // Ticker refreshes every 30s (clean up on leave)
  const tickerInterval = setInterval(refreshMarketTicker, 30000);
  window.addEventListener('beforeunload', () => clearInterval(tickerInterval));

  // Dashboard: full page reload every 2 minutes to show fresh data
  startDashboardAutoReload();
});
