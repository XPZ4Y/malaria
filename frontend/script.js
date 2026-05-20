const views = {
  home: document.getElementById('homeView'),
  scan: document.getElementById('scanView'),
  reports: document.getElementById('reportsView'),
  settings: document.getElementById('settingsView')
};

const navItems = document.querySelectorAll('.nav-item');
const uploadBtn = document.getElementById('uploadBtn');
const imageInput = document.getElementById('imageInput');
const patientNameInput = document.getElementById('patientNameInput');
const scanStatus = document.getElementById('scanStatus');
const scanResult = document.getElementById('scanResult');
const recentReports = document.getElementById('recentReports');
const historyList = document.getElementById('historyList');
const testsTodayCount = document.getElementById('testsTodayCount');
const positiveCasesCount = document.getElementById('positiveCasesCount');

const supabaseClient = window.supabase && window.SUPABASE_URL && window.SUPABASE_KEY
  ? window.supabase.createClient(window.SUPABASE_URL, window.SUPABASE_KEY)
  : null;

let activeTab = 'home';
let isScanning = false;

function setActiveTab(tab) {
  activeTab = tab;

  Object.entries(views).forEach(([key, el]) => {
    el.classList.toggle('hidden', key !== tab);
  });

  navItems.forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tab);
  });
}

function setScanning(state, message = 'Ready for a new slide.') {
  isScanning = state;
  uploadBtn.disabled = state;
  scanStatus.textContent = message;

  if (state) {
    uploadBtn.innerHTML = `<span class="spinner"></span> Processing Slide...`;
    document.getElementById('homeView').classList.add('loading');
    document.getElementById('scanView').classList.add('loading');
  } else {
    uploadBtn.textContent = 'Upload Image';
    document.getElementById('homeView').classList.remove('loading');
    document.getElementById('scanView').classList.remove('loading');
  }
}

function formatDate(dateString) {
  const date = new Date(dateString);
  if (Number.isNaN(date.getTime())) {
    return dateString;
  }
  return date.toLocaleString([], {
    hour: '2-digit',
    minute: '2-digit',
    day: '2-digit',
    month: 'short'
  });
}

function reportDisplayName(report) {
  return report.patient_name || 'Unknown Patient';
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'N/A';
  }
  return `${Number(value).toFixed(2)}%`;
}

function mapReportToRow(report) {
  return {
    id: report.id,
    patient_name: report.patient_name,
    created_at: report.created_at,
    filename: report.filename,
    status: report.status,
    diagnosis: report.diagnosis,
    infection_rate: report.infection_rate,
    model_confidence: report.model_confidence ?? report.confidence ?? null,
    inference_mode: report.inference_mode,
    model_probability: report.model_probability ?? null,
    total_cells: report.total_cells ?? null,
    infected_cells: report.infected_cells ?? null,
    healthy_cells: report.healthy_cells ?? null,
    quality_ok: report.quality_ok ?? null,
    quality_message: report.quality_message ?? null
  };
}

async function queueReport(row) {
  if (!('serviceWorker' in navigator)) {
    return;
  }
  const registration = await navigator.serviceWorker.ready;
  if (registration.active) {
    registration.active.postMessage({ type: 'enqueueReport', payload: row });
  }
}

async function saveReport(report) {
  if (!supabaseClient) {
    console.error('Supabase client not configured. Check config.js and Supabase scripts.');
    return;
  }
  const row = mapReportToRow(report);

  if (!navigator.onLine) {
    await queueReport(row);
    return;
  }

  const { error } = await supabaseClient.from('reports').insert([row]);
  if (error) {
    console.error('Supabase insert failed, queueing report.', error);
    await queueReport(row);
  }
}

function renderReportCard(report) {
  const label = report.inference_mode === 'direct_image'
    ? 'Model Confidence'
    : 'Infection Rate';
  const score = report.inference_mode === 'direct_image'
    ? (report.model_confidence ?? report.confidence ?? report.infection_rate)
    : report.infection_rate;

  return `
    <div class="report-card">
      <div class="report-row">
        <div>
          <div class="test-id">${reportDisplayName(report)}</div>
          <div class="report-meta">${formatDate(report.created_at)} · ${report.filename}</div>
        </div>
        <div class="status ${report.status}">${report.status.toUpperCase()}</div>
      </div>
      <div class="report-meta">${report.diagnosis}</div>
      <div class="report-stats">
        <div class="report-stat">
          <small>${label}</small>
          <strong>${formatPercent(score)}</strong>
        </div>
        <div class="report-stat">
          <small>Inference</small>
          <strong>${report.inference_mode === 'direct_image' ? 'Direct Image' : 'Cell Pipeline'}</strong>
        </div>
      </div>
    </div>
  `;
}

function renderRecentReport(report) {
  return `
    <div class="test">
      <div class="left">
        <div class="dot ${report.status === 'positive' ? 'red' : 'green'}"></div>
        <div>
          <div class="test-id">${reportDisplayName(report)}</div>
          <div class="meta">${formatDate(report.created_at)} · ${report.filename}</div>
        </div>
      </div>
      <div class="status ${report.status}">${report.status.toUpperCase()}</div>
    </div>
  `;
}

function updateStats(reports) {
  const today = new Date().toDateString();
  const todaysReports = reports.filter(report => new Date(report.created_at).toDateString() === today);
  const positives = todaysReports.filter(report => report.status === 'positive');

  testsTodayCount.textContent = String(todaysReports.length);
  positiveCasesCount.textContent = String(positives.length);
}

function renderHistory(reports) {
  updateStats(reports);

  if (!reports.length) {
    const emptyMarkup = '<div class="empty-state">No reports yet. Run a scan to populate this list.</div>';
    recentReports.innerHTML = emptyMarkup;
    historyList.innerHTML = emptyMarkup;
    return;
  }

  recentReports.innerHTML = reports.slice(0, 3).map(renderRecentReport).join('');
  historyList.innerHTML = reports.map(renderReportCard).join('');
}

async function fetchHistory() {
  try {
    if (!supabaseClient) {
      throw new Error('Supabase not configured');
    }
    const { data, error } = await supabaseClient
      .from('reports')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(25);

    if (error) {
      throw error;
    }

    renderHistory(data || []);
  } catch (error) {
    console.error('Failed to load report history from Supabase.', error);
    const fallback = '<div class="empty-state">Could not load report history.</div>';
    recentReports.innerHTML = fallback;
    historyList.innerHTML = fallback;
  }
}

function renderScanResult(report) {
  scanResult.classList.remove('hidden');
  document.getElementById('resultPatientName').textContent = report.patient_name;
  document.getElementById('resultDiagnosisText').textContent = report.diagnosis;
  const scoreLabel = report.inference_mode === 'direct_image'
    ? 'Model Confidence'
    : 'Detected Infection Rate';
  const scoreValue = report.inference_mode === 'direct_image'
    ? (report.model_confidence ?? report.confidence ?? report.infection_rate)
    : report.infection_rate;
  document.getElementById('resultScoreLabel').textContent = scoreLabel;
  document.getElementById('resultInfectionRate').textContent = formatPercent(scoreValue);

  const statusPill = document.getElementById('resultStatusPill');
  statusPill.textContent = report.status === 'positive' ? 'MALARIA' : 'NO MALARIA';
  statusPill.className = `status ${report.status}`;
}

async function uploadSlide(file) {
  const patientName = patientNameInput.value.trim();
  if (!patientName) {
    scanStatus.textContent = 'Enter patient name before uploading.';
    setActiveTab('scan');
    patientNameInput.focus();
    return;
  }

  const payload = new FormData();
  payload.append('image', file);
  payload.append('patient_name', patientName);

  setScanning(true, 'Analyzing uploaded slide...');
  setActiveTab('scan');

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: payload
    });

    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || 'Prediction failed.');
    }

    renderScanResult(result);
    scanStatus.textContent = `Completed report for ${result.patient_name}.`;
    await saveReport(result);
    await fetchHistory();
  } catch (error) {
    console.error('Upload failed.', error);
    scanStatus.textContent = error.message;
  } finally {
    setScanning(false, scanStatus.textContent);
    imageInput.value = '';
  }
}

navItems.forEach(btn => {
  btn.addEventListener('click', () => {
    setActiveTab(btn.dataset.tab);
  });
});

uploadBtn.addEventListener('click', () => {
  if (!isScanning) {
    imageInput.click();
  }
});

imageInput.addEventListener('change', event => {
  const [file] = event.target.files;
  if (file) {
    uploadSlide(file);
  }
});

async function requestQueueSync() {
  if (!('serviceWorker' in navigator)) {
    return;
  }
  const registration = await navigator.serviceWorker.ready;
  if (registration.active) {
    registration.active.postMessage({ type: 'syncReports' });
  }
}

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').then(() => {
    if (navigator.onLine) {
      requestQueueSync();
    }
  }).catch(error => {
    console.error('Service worker registration failed.', error);
  });

  window.addEventListener('online', () => {
    requestQueueSync();
    fetchHistory();
  });
}

setActiveTab('home');
fetchHistory();
