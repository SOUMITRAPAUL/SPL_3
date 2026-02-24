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

// ── Slider live-update ────────────────────────────────────────────────────────
Object.values(sliders).forEach(s => {
  s.el.addEventListener('input', () => {
    s.badge.textContent = parseFloat(s.el.value).toFixed(2);
  });
});

resetBtn.addEventListener('click', () => {
  sliders.enhance.el.value = '1'; sliders.enhance.badge.textContent = '1.00';
  sliders.brightness.el.value = '1.50'; sliders.brightness.badge.textContent = '1.50';
  sliders.sharpness.el.value = '2.00'; sliders.sharpness.badge.textContent = '2.00';
  sliders.contrast.el.value = '1.00'; sliders.contrast.badge.textContent = '1.00';
  sliders.gamma.el.value = '1.30'; sliders.gamma.badge.textContent = '1.30';
  sliders.saturation.el.value = '1.00'; sliders.saturation.badge.textContent = '1.00';
  sliders.denoise.el.value = '0.00'; sliders.denoise.badge.textContent = '0.00';
  alignCheck.checked = false;
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
  btnTxt.textContent = 'Enhancing Batch...';

  // Process files one by one (to keep server stable and responsive)
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

async function processFile(file) {
  const card = document.getElementById(file.cardId);
  const statusEl = card.querySelector('.status');
  const infoEl = card.querySelector('.info');
  const enhImgEl = card.querySelector('.enhanced-img');
  const dlBtn = card.querySelector('.dl-btn');

  statusEl.textContent = 'Processing...';
  statusEl.className = 'status status-processing';

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
    const data = await res.json();

    if (!res.ok || data.error) throw new Error(data.error || 'Server error');

    enhImgEl.src = 'data:image/png;base64,' + data.enhanced_b64;
    dlBtn.href = enhImgEl.src;
    dlBtn.classList.remove('hidden');

    statusEl.textContent = 'Done';
    statusEl.className = 'status status-done';
    infoEl.textContent = `${data.width} × ${data.height} px`;

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
