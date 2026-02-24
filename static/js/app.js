// ── DOM refs ──────────────────────────────────────────────────────────────────
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileNameEl = document.getElementById('fileName');
const enhBtn = document.getElementById('enhBtn');
const resetBtn = document.getElementById('resetBtn');
const btnTxt = document.getElementById('btnTxt');
const spinner = document.getElementById('spinner');
const placeholder = document.getElementById('placeholder');
const resultsFeed = document.getElementById('resultsFeed');
const toast = document.getElementById('toast');
const alignCheck = document.getElementById('alignCheck');

const sliders = {
  enhance: { el: document.getElementById('eSlider'), badge: document.getElementById('eVal') },
  brightness: { el: document.getElementById('bSlider'), badge: document.getElementById('bVal') },
  sharpness: { el: document.getElementById('shSlider'), badge: document.getElementById('shVal') },
  contrast: { el: document.getElementById('cSlider'), badge: document.getElementById('cVal') },
  gamma: { el: document.getElementById('gSlider'), badge: document.getElementById('gVal') },
  saturation: { el: document.getElementById('sSlider'), badge: document.getElementById('sVal') },
  denoise: { el: document.getElementById('dSlider'), badge: document.getElementById('dVal') },
};

let processingQueue = [];
let isProcessing = false;
let lastFile = null;   // last file that was successfully enhanced
let debounceTimer = null;
const DEBOUNCE_MS = 700;    // ms to wait after slider stops before re-enhancing

// ── Live CSS preview (instant, zero-latency) ──────────────────────────────────
function applyLivePreview() {
  const brightness = parseFloat(sliders.brightness.el.value);
  const contrast = parseFloat(sliders.contrast.el.value);
  const saturation = parseFloat(sliders.saturation.el.value);
  // Approximate gamma with brightness offset (not pixel-accurate, but good enough for preview)
  const gamma = parseFloat(sliders.gamma.el.value);
  const gammaBrightness = gamma > 1 ? 1 + (gamma - 1) * 0.25 : gamma;

  const cssFilter = `brightness(${brightness * gammaBrightness}) contrast(${contrast}) saturate(${saturation})`;

  document.querySelectorAll('.enhanced-img').forEach(img => {
    if (img.getAttribute('data-enhanced') === 'true') {
      img.style.filter = cssFilter;
    }
  });
}

// Strip the live CSS filter once accurate server result is applied
function clearLiveFilter() {
  document.querySelectorAll('.enhanced-img').forEach(img => {
    img.style.filter = '';
  });
}

// ── Debounced server re-enhance ───────────────────────────────────────────────
function scheduleReEnhance() {
  clearTimeout(debounceTimer);
  if (!lastFile) return;   // no image enhanced yet – nothing to do

  debounceTimer = setTimeout(async () => {
    clearLiveFilter();
    await processFile(lastFile);
  }, DEBOUNCE_MS);
}

// ── Slider live-update ────────────────────────────────────────────────────────
Object.values(sliders).forEach(s => {
  s.el.addEventListener('input', () => {
    s.badge.textContent = parseFloat(s.el.value).toFixed(2);
    applyLivePreview();       // instant CSS feedback
    scheduleReEnhance();      // accurate server result after pause
  });
});

// Auto-align toggle also triggers re-enhance
alignCheck.addEventListener('change', () => {
  scheduleReEnhance();
});

// ── Reset ─────────────────────────────────────────────────────────────────────
resetBtn.addEventListener('click', () => {
  sliders.enhance.el.value = '1'; sliders.enhance.badge.textContent = '1.00';
  sliders.brightness.el.value = '1.50'; sliders.brightness.badge.textContent = '1.50';
  sliders.sharpness.el.value = '2.00'; sliders.sharpness.badge.textContent = '2.00';
  sliders.contrast.el.value = '1.00'; sliders.contrast.badge.textContent = '1.00';
  sliders.gamma.el.value = '1.30'; sliders.gamma.badge.textContent = '1.30';
  sliders.saturation.el.value = '1.00'; sliders.saturation.badge.textContent = '1.00';
  sliders.denoise.el.value = '0.00'; sliders.denoise.badge.textContent = '0.00';
  alignCheck.checked = false;
  clearLiveFilter();
  scheduleReEnhance();   // re-enhance with reset defaults
});

// ── File selection ────────────────────────────────────────────────────────────
function addFiles(files) {
  if (!files.length) return;
  const allowed = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp', 'image/tiff'];

  for (const file of files) {
    if (!allowed.includes(file.type)) {
      showError(`Format ignored: ${file.name}`);
      continue;
    }
    processingQueue.push(file);
    createResultCard(file);
  }

  placeholder.classList.add('hidden');
  enhBtn.disabled = false;
  fileNameEl.textContent = `${processingQueue.length} images in queue`;
  fileNameEl.classList.remove('hidden');
}

// ── Result Card Factory ───────────────────────────────────────────────────────
function createResultCard(file) {
  const cardId = 'card-' + Math.random().toString(36).substr(2, 9);
  const card = document.createElement('div');
  card.className = 'result-card';
  card.id = cardId;
  card.innerHTML = `
    <div class="result-card-header">
      <span class="filename">${file.name}</span>
      <span class="status status-processing">Pending</span>
    </div>
    <div class="result-card-body">
      <div class="img-box">
        <span class="label">Original</span>
        <img src="${URL.createObjectURL(file)}" />
      </div>
      <div class="img-box result-img-box">
        <span class="label">Enhanced</span>
        <img class="enhanced-img" src="" />
      </div>
    </div>
    <div class="result-card-footer">
      <span class="info">Waiting for enhancement...</span>
      <a class="btn-primary btn-sm dl-btn hidden" download="enhanced_${file.name}">Download</a>
    </div>
  `;
  resultsFeed.prepend(card);
  file.cardId = cardId;
}

// ── Drag & Drop ───────────────────────────────────────────────────────────────
dropZone.addEventListener('click', (e) => {
  if (e.target.tagName !== 'LABEL' && e.target.id !== 'fileInput') {
    fileInput.click();
  }
});

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  addFiles(e.dataTransfer.files);
  fileInput.value = '';
});

// ── File Selection ────────────────────────────────────────────────────────────
fileInput.addEventListener('change', () => {
  addFiles(fileInput.files);
  fileInput.value = '';
});

enhBtn.addEventListener('click', async () => {
  if (isProcessing || processingQueue.length === 0) return;

  isProcessing = true;
  enhBtn.disabled = true;
  spinner.classList.remove('hidden');
  btnTxt.textContent = 'Enhancing...';

  while (processingQueue.length > 0) {
    const file = processingQueue.shift();
    await processFile(file);
    fileNameEl.textContent = `${processingQueue.length} images remaining`;
  }

  isProcessing = false;
  enhBtn.disabled = false;
  spinner.classList.add('hidden');
  btnTxt.textContent = 'Enhance All';
  fileNameEl.textContent = 'Batch Complete';
});

// ── Core enhancement function ─────────────────────────────────────────────────
async function processFile(file) {
  const card = document.getElementById(file.cardId);
  if (!card) return;

  const statusEl = card.querySelector('.status');
  const infoEl = card.querySelector('.info');
  const enhImgEl = card.querySelector('.enhanced-img');
  const dlBtn = card.querySelector('.dl-btn');

  statusEl.textContent = 'Processing…';
  statusEl.className = 'status status-processing';
  infoEl.textContent = 'Sending to model…';

  const form = new FormData();
  form.append('file', file);
  form.append('enhance', sliders.enhance.el.value);
  form.append('brightness', sliders.brightness.el.value);
  form.append('sharpness', sliders.sharpness.el.value);
  form.append('contrast', sliders.contrast.el.value);
  form.append('gamma', sliders.gamma.el.value);
  form.append('saturation', sliders.saturation.el.value);
  form.append('denoise', sliders.denoise.el.value);
  form.append('auto_align', alignCheck.checked ? 'true' : 'false');

  try {
    const res = await fetch('/enhance', { method: 'POST', body: form });

    // Guard: server may return HTML (502/503 proxy error) instead of JSON
    const contentType = res.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
      const text = await res.text();
      throw new Error(`Server error ${res.status} — ${text.replace(/<[^>]+>/g, '').trim().slice(0, 200)}`);
    }

    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Server error');

    enhImgEl.src = 'data:image/png;base64,' + data.enhanced_b64;
    enhImgEl.setAttribute('data-enhanced', 'true');
    enhImgEl.style.filter = '';   // clear any live CSS preview
    dlBtn.href = enhImgEl.src;
    dlBtn.classList.remove('hidden');

    statusEl.textContent = 'Done ✓';
    statusEl.className = 'status status-done';
    infoEl.textContent = `${data.width} × ${data.height} px`;

    lastFile = file;   // store for slider-triggered re-enhance

  } catch (err) {
    statusEl.textContent = 'Error';
    statusEl.className = 'status status-error';
    infoEl.textContent = err.message;
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function showError(msg) {
  toast.textContent = '⚠ ' + msg;
  toast.classList.remove('hidden');
  setTimeout(() => toast.classList.add('hidden'), 5000);
}
