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
  var arcLen = 282.7;
  document.querySelectorAll('.arc-fill-path').forEach(function(el) {
    var targetOffset = parseFloat(el.dataset.target || arcLen);
    el.style.strokeDashoffset = arcLen;
    setTimeout(function() {
      el.style.transition = 'stroke-dashoffset 1.3s ease';
      el.style.strokeDashoffset = targetOffset;
    }, 250);
  });
}

// ─── C/P Bar Animations ───────────────────────────────────────────────────────
function animateCPBars() {
  document.querySelectorAll('.cp-calls-fill').forEach(function(el) {
    var pct = el.dataset.pct;
    el.style.width = '0%';
    setTimeout(function() {
      el.style.width = pct + '%';
    }, 500);
  });
}

// ─── Sector bar animations (landing page) ────────────────────────────────────
function animateSectorBars() {
  document.querySelectorAll('.sector-bar-fill').forEach(function(el) {
    var target = el.style.width;
    el.style.width = '0%';
    setTimeout(function() { el.style.width = target; }, 200);
  });
}

// ─── Escape HTML to prevent XSS in dynamic content ───────────────────────────
function escapeHTML(str) {
  if (typeof str !== 'string') return String(str);
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ─── Live Market Ticker Refresh (every 30s) ──────────────────────────────────
function refreshMarketTicker() {
  fetch('/api/market')
    .then(function(res) {
      if (!res.ok) return;
      return res.json();
    })
    .then(function(data) {
      if (!data || !data.length) return;

      var track = document.getElementById('tickerTrack');
      if (!track) return;

      var items = data.concat(data);
      // C-1 (XSS fix): Use escapeHTML on all dynamic values
      track.innerHTML = items.map(function(item) {
        return '<span class="ticker-item">' +
          '<span class="ticker-name">' + escapeHTML(item.name) + '</span>' +
          '<span class="ticker-price">' + escapeHTML(item.price) + '</span>' +
          '<span class="ticker-change ' + (item.positive ? 'pos' : 'neg') + '">' +
            (item.positive ? '+' : '') + escapeHTML(item.pct) + '%' +
          '</span>' +
        '</span>';
      }).join('');
    })
    .catch(function() {
      // Silently fail — ticker will just not update
    });
}

// ─── Auto-dismiss flash messages ─────────────────────────────────────────────
document.querySelectorAll('.flash').forEach(function(el) {
  setTimeout(function() {
    el.style.transition = 'opacity 0.5s';
    el.style.opacity = '0';
    setTimeout(function() { el.remove(); }, 500);
  }, 4000);
});

// ─── Dashboard auto-reload with visible countdown ───────────────────────────
function startDashboardAutoReload() {
  if (!document.querySelector('.dashboard')) return;

  var countdown = 120;
  var updatedEl = document.getElementById('updatedAt');

  var reloadInterval = setInterval(function() {
    countdown--;
    if (updatedEl && countdown <= 15) {
      updatedEl.textContent = 'Refreshing in ' + countdown + 's';
    }
    if (countdown <= 0) {
      clearInterval(reloadInterval);
      window.location.reload();
    }
  }, 1000);

  window.addEventListener('beforeunload', function() { clearInterval(reloadInterval); });
}

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
  animateArcGauge();
  animateCPBars();
  animateSectorBars();

  // Mobile hamburger toggle
  var navToggle = document.getElementById('navToggle');
  var navLinks = document.querySelector('.nav-links');
  if (navToggle && navLinks) {
    navToggle.addEventListener('click', function() {
      navLinks.classList.toggle('open');
    });
  }

  // Ticker refreshes every 30s (clean up on leave)
  var tickerInterval = setInterval(refreshMarketTicker, 30000);
  window.addEventListener('beforeunload', function() { clearInterval(tickerInterval); });

  // Dashboard: full page reload every 2 minutes to show fresh data
  startDashboardAutoReload();
});
