// State
let activeTab = 'Home';
let isScanning = false;
let scanStatus = 'ready';
let selectedSampleName = '';
let patientDetails = { name: '', dob: '', spO2: '', bp: '' };

const navItems = [
  { label: 'Home', icon: 'house' },
  { label: 'Scan', icon: 'scan-line' },
  { label: 'Reports', icon: 'file-text' },
  { label: 'Settings', icon: 'settings' }
];

const mockReports = [
  { id: 'PT-00250', name: 'Ramesh Kumar', dob: '1985-04-12', spO2: 95, bp: '120/80', status: 'POSITIVE', date: '2026-05-20', time: '11:32 AM' },
  { id: 'PT-00249', name: 'Sunita Sharma', dob: '1992-08-05', spO2: 98, bp: '110/70', status: 'NEGATIVE', date: '2026-05-20', time: '10:18 AM' },
  { id: 'PT-00248', name: 'Abdul Rehman', dob: '1970-11-20', spO2: 96, bp: '135/85', status: 'NEGATIVE', date: '2026-05-20', time: '09:02 AM' }
];

const escapeHTML = (value = '') => String(value).replace(/[&<>"']/g, (char) => ({
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;'
}[char]));

const calculateAge = (dob) => {
  if (!dob) return '';

  const [year, month, day] = dob.split('-').map(Number);
  const birthDate = new Date(year, month - 1, day);
  const today = new Date();
  let age = today.getFullYear() - birthDate.getFullYear();
  const monthOffset = today.getMonth() - birthDate.getMonth();

  if (monthOffset < 0 || (monthOffset === 0 && today.getDate() < birthDate.getDate())) {
    age--;
  }

  return age;
};

const brandMarkSvg = `
  <svg viewBox="0 0 40 40" aria-hidden="true">
    <rect x="4" y="4" width="32" height="32" rx="8"></rect>
    <path d="M13 22h14M20 13v14"></path>
    <circle cx="28" cy="12" r="3"></circle>
  </svg>
`;

const scanSampleSvg = `
  <svg class="sample-svg" viewBox="0 0 320 220" role="img" aria-label="Blood smear sample scan">
    <defs>
      <radialGradient id="cellCore" cx="50%" cy="45%" r="58%">
        <stop offset="0%" stop-color="#ffffff"></stop>
        <stop offset="70%" stop-color="#f5b7bd"></stop>
        <stop offset="100%" stop-color="#d45a65"></stop>
      </radialGradient>
      <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="7" stdDeviation="8" flood-color="#9b111e" flood-opacity="0.14"></feDropShadow>
      </filter>
    </defs>
    <rect x="8" y="8" width="304" height="204" rx="8" fill="#fff" stroke="#eadce0"></rect>
    <g filter="url(#softShadow)" fill="url(#cellCore)" stroke="#cf5863" stroke-width="2" opacity="0.94">
      <circle cx="66" cy="62" r="22"></circle>
      <circle cx="132" cy="48" r="18"></circle>
      <circle cx="232" cy="56" r="24"></circle>
      <circle cx="270" cy="122" r="18"></circle>
      <circle cx="176" cy="118" r="25"></circle>
      <circle cx="96" cy="142" r="20"></circle>
      <circle cx="217" cy="168" r="18"></circle>
    </g>
    <g fill="#007f73" opacity="0.88">
      <circle cx="132" cy="48" r="4"></circle>
      <circle cx="176" cy="118" r="5"></circle>
      <circle cx="217" cy="168" r="4"></circle>
    </g>
    <path d="M42 185c38-19 76-18 116 3 40 21 79 22 120 0" fill="none" stroke="#007f73" stroke-width="3" stroke-linecap="round" opacity="0.32"></path>
  </svg>
`;

const statusText = {
  ready: {
    title: 'Sample required',
    body: 'Capture or upload a microscope image to begin analysis.'
  },
  scanning: {
    title: 'Analyzing smear',
    body: 'Cell regions are being checked for malaria indicators.'
  },
  complete: {
    title: 'Scan complete',
    body: 'A preliminary screening result is ready for clinical review.'
  }
};

window.setActiveTab = (tab) => {
  activeTab = tab;
  render();
};

window.handleDetailChange = (event) => {
  patientDetails[event.target.name] = event.target.value;
};

window.handleStartDetection = () => {
  isScanning = true;
  scanStatus = 'ready';
  render();

  setTimeout(() => {
    isScanning = false;
    activeTab = 'Scan';
    render();
  }, 1200);
};

window.openSamplePicker = (inputId) => {
  const input = document.getElementById(inputId);
  if (input) input.click();
};

window.handleSampleSelect = (event) => {
  const file = event.target.files && event.target.files[0];
  selectedSampleName = file ? file.name : '';
  scanStatus = selectedSampleName ? 'ready' : scanStatus;
  render();
};

window.handleRunSampleScan = () => {
  if (!selectedSampleName || scanStatus === 'scanning') return;

  scanStatus = 'scanning';
  render();

  setTimeout(() => {
    scanStatus = 'complete';
    render();
  }, 1800);
};

function renderShell(root) {
  root.innerHTML = `
    <div class="app">
      <header class="topbar">
        <div class="brand">
          <span class="logo-mark">${brandMarkSvg}</span>
          <div class="logo">JWARA.AI</div>
        </div>
        <div class="badge">AI malaria screening</div>
      </header>
      <main class="content" id="content-area"></main>
      <nav class="navbar" aria-label="Primary navigation">
        ${navItems.map((item) => `
          <button class="nav-item ${(activeTab === item.label || (item.label === 'Home' && activeTab === 'PatientForm')) ? 'active' : ''}" type="button" onclick="setActiveTab('${item.label}')">
            <i data-lucide="${item.icon}"></i>
            <span>${item.label}</span>
          </button>
        `).join('')}
      </nav>
    </div>
  `;
}

function renderHome(area) {
  area.innerHTML = `
    <section class="hero-panel">
      <div>
        <div class="eyebrow"><i data-lucide="activity"></i><span>Clinic screening console</span></div>
        <h1 class="main-title">Early detection for malaria screening.</h1>
        <p class="sub">AI-assisted blood smear review with patient vitals, sample capture, and clinical report tracking in one workflow.</p>
      </div>
      <div class="mini-scan" aria-hidden="true">
        ${scanSampleSvg}
        <span class="mini-scan-line"></span>
      </div>
    </section>

    <section class="stats" aria-label="Daily summary">
      <div class="card">
        <small>Tests Today</small>
        <h3>14</h3>
      </div>
      <div class="card">
        <small>Positive Cases</small>
        <h3>3</h3>
      </div>
    </section>

    <section class="action-grid" aria-label="Main actions">
      <button class="action-card" type="button" onclick="setActiveTab('PatientForm')">
        <span class="icon-wrapper"><i data-lucide="activity"></i></span>
        <span class="action-content">
          <strong>Start New Test</strong>
          <span>Enter patient vitals and begin sample screening</span>
        </span>
        <i class="action-arrow" data-lucide="chevron-right"></i>
      </button>
      <button class="action-card" type="button" onclick="setActiveTab('Reports')">
        <span class="icon-wrapper accent"><i data-lucide="database"></i></span>
        <span class="action-content">
          <strong>Clinical Database</strong>
          <span>Review saved patient reports and outcomes</span>
        </span>
        <i class="action-arrow" data-lucide="chevron-right"></i>
      </button>
    </section>
  `;
}

function renderPatientForm(area) {
  area.innerHTML = `
    <div class="header-with-back">
      <button class="back-btn" type="button" onclick="setActiveTab('Home')" aria-label="Back to home"><i data-lucide="arrow-left"></i></button>
      <h2>Patient Details</h2>
    </div>
    <section class="health-card">
      <p class="form-note">Add the required vitals before opening the scanner.</p>
      <label class="question">
        <span>Patient Name</span>
        <input type="text" name="name" value="${escapeHTML(patientDetails.name)}" autocomplete="name" oninput="handleDetailChange(event)">
      </label>
      <label class="question">
        <span>Date of Birth</span>
        <input type="date" name="dob" value="${escapeHTML(patientDetails.dob)}" oninput="handleDetailChange(event)">
      </label>
      <label class="question">
        <span>Blood Oxygen Level (SpO2 %)</span>
        <input type="number" name="spO2" min="50" max="100" value="${escapeHTML(patientDetails.spO2)}" oninput="handleDetailChange(event)">
      </label>
      <label class="question">
        <span>Blood Pressure (mmHg)</span>
        <input type="text" name="bp" value="${escapeHTML(patientDetails.bp)}" inputmode="numeric" oninput="handleDetailChange(event)">
      </label>
      <button class="btn" type="button" onclick="handleStartDetection()" ${isScanning ? 'disabled' : ''}>
        ${isScanning ? '<i data-lucide="loader-2" class="animate-spin"></i><span>Opening scanner</span>' : '<i data-lucide="scan-line"></i><span>Diagnose Patient</span>'}
      </button>
    </section>
  `;
}

function renderScan(area) {
  const copy = statusText[scanStatus];
  const patientName = patientDetails.name.trim() ? escapeHTML(patientDetails.name.trim()) : 'Unassigned patient';
  const sampleLabel = selectedSampleName ? escapeHTML(selectedSampleName) : 'No image selected';
  const canRunScan = Boolean(selectedSampleName) && scanStatus !== 'scanning';
  const isRunning = scanStatus === 'scanning';

  area.innerHTML = `
    <div class="header-with-back">
      <button class="back-btn" type="button" onclick="setActiveTab('PatientForm')" aria-label="Back to patient details"><i data-lucide="arrow-left"></i></button>
      <h2>AI Scanner</h2>
    </div>

    <section class="scan-console">
      <div class="scan-meta">
        <span><i data-lucide="user-round"></i>${patientName}</span>
        <span><i data-lucide="image"></i>${sampleLabel}</span>
      </div>
      <div class="scan-frame ${scanStatus}">
        ${scanSampleSvg}
        <span class="scan-corner top-left"></span>
        <span class="scan-corner top-right"></span>
        <span class="scan-corner bottom-left"></span>
        <span class="scan-corner bottom-right"></span>
        <span class="scan-beam"></span>
      </div>
      <div class="scan-status">
        <span class="status-dot ${scanStatus}"></span>
        <div>
          <strong>${copy.title}</strong>
          <p>${copy.body}</p>
        </div>
      </div>
    </section>

    <section class="sample-actions" aria-label="Sample actions">
      <input class="visually-hidden" id="cameraInput" type="file" accept="image/*" capture="environment" onchange="handleSampleSelect(event)">
      <input class="visually-hidden" id="uploadInput" type="file" accept="image/*,.png,.jpg,.jpeg,.webp,.tif,.tiff" onchange="handleSampleSelect(event)">
      <button class="btn secondary" type="button" onclick="openSamplePicker('cameraInput')">
        <i data-lucide="camera"></i><span>Capture Microscope Image</span>
      </button>
      <button class="btn secondary" type="button" onclick="openSamplePicker('uploadInput')">
        <i data-lucide="upload"></i><span>Upload Sample Image</span>
      </button>
      <button class="btn accent" type="button" onclick="handleRunSampleScan()" ${canRunScan ? '' : 'disabled'}>
        ${isRunning ? '<i data-lucide="loader-2" class="animate-spin"></i><span>Scanning Sample</span>' : '<i data-lucide="radar"></i><span>Run AI Scan</span>'}
      </button>
    </section>

    ${scanStatus === 'complete' ? `
      <section class="result-panel">
        <div>
          <small>Preliminary result</small>
          <strong>Possible malaria indicators detected</strong>
        </div>
        <span class="status-badge positive">REVIEW</span>
      </section>
    ` : ''}
  `;
}

function renderReports(area) {
  area.innerHTML = `
    <h1 class="main-title page-title">Clinical Database</h1>
    <section class="report-list" aria-label="Clinical reports">
      ${mockReports.map((report) => `
        <article class="report-card ${report.status.toLowerCase()}">
          <div class="report-header">
            <div>
              <div class="report-name">${escapeHTML(report.name)}</div>
              <div class="report-demographics">${escapeHTML(report.id)} | DOB ${escapeHTML(report.dob)} | Age ${calculateAge(report.dob)}</div>
            </div>
            <div class="status-badge ${report.status.toLowerCase()}">${escapeHTML(report.status)}</div>
          </div>
          <div class="vitals-grid">
            <div class="vital-item">
              <div class="vital-label">SpO2 Level</div>
              <div class="vital-value">${report.spO2}%</div>
            </div>
            <div class="vital-item">
              <div class="vital-label">Blood Pressure</div>
              <div class="vital-value">${escapeHTML(report.bp)}</div>
            </div>
          </div>
          <div class="report-time"><i data-lucide="clock-3"></i>${escapeHTML(report.date)} at ${escapeHTML(report.time)}</div>
        </article>
      `).join('')}
    </section>
  `;
}

function renderSettings(area) {
  area.innerHTML = `
    <h1 class="main-title page-title">Settings</h1>
    <section class="settings-list" aria-label="Application settings">
      <div class="settings-row">
        <div>
          <strong>Clinic Profile</strong>
          <span>Primary care unit</span>
        </div>
        <button class="icon-btn" type="button" aria-label="Edit clinic profile"><i data-lucide="pencil"></i></button>
      </div>
      <div class="settings-row">
        <div>
          <strong>AI Threshold</strong>
          <span>High sensitivity</span>
        </div>
        <button class="icon-btn" type="button" aria-label="Adjust AI threshold"><i data-lucide="sliders-horizontal"></i></button>
      </div>
      <div class="settings-row">
        <div>
          <strong>Report Sync</strong>
          <span>Manual review before upload</span>
        </div>
        <button class="icon-btn" type="button" aria-label="Configure report sync"><i data-lucide="cloud-upload"></i></button>
      </div>
    </section>
  `;
}

function render() {
  const root = document.getElementById('root');
  renderShell(root);

  const area = document.getElementById('content-area');

  if (activeTab === 'Home') renderHome(area);
  if (activeTab === 'PatientForm') renderPatientForm(area);
  if (activeTab === 'Scan') renderScan(area);
  if (activeTab === 'Reports') renderReports(area);
  if (activeTab === 'Settings') renderSettings(area);

  if (window.lucide && typeof window.lucide.createIcons === 'function') {
    window.lucide.createIcons();
  }
}

render();
