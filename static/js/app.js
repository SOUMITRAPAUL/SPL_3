// ── Components mapping to Architectural Diagram ────────────────────────────────

const ImageUploadModule = {
  dropZone: document.getElementById('dropZone'),
  fileInput: document.getElementById('fileInput'),
  fileNameEl: document.getElementById('fileName'),

  init() {
    this.dropZone.addEventListener('click', (e) => {
      if (e.target.tagName !== 'LABEL' && e.target.id !== 'fileInput') {
        this.fileInput.click();
      }
    });

    this.dropZone.addEventListener('dragover', e => { e.preventDefault(); this.dropZone.classList.add('drag-over'); });
    this.dropZone.addEventListener('dragleave', () => this.dropZone.classList.remove('drag-over'));
    this.dropZone.addEventListener('drop', e => {
      e.preventDefault();
      this.dropZone.classList.remove('drag-over');
      this.addFiles(e.dataTransfer.files);
      this.fileInput.value = '';
    });

    this.fileInput.addEventListener('change', () => {
      this.addFiles(this.fileInput.files);
      this.fileInput.value = '';
    });
  },

  addFiles(files) {
    if (!files.length) return;
    const allowed = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp', 'image/tiff'];

    for (const file of files) {
      if (!allowed.includes(file.type)) {
        showError(`Format ignored: ${file.name}`);
        continue;
      }
      processingQueue.push(file);
      OutputDisplay.createResultCard(file);
    }

    placeholder.classList.add('hidden');
    enhBtn.disabled = false;
    this.fileNameEl.textContent = `${processingQueue.length} images in queue`;
    this.fileNameEl.classList.remove('hidden');
  }
};

const PreferencePanel = {
  sliders: {
    enhance: { el: document.getElementById('eSlider'), badge: document.getElementById('eVal') },
    brightness: { el: document.getElementById('bSlider'), badge: document.getElementById('bVal') },
    sharpness: { el: document.getElementById('shSlider'), badge: document.getElementById('shVal') },
    contrast: { el: document.getElementById('cSlider'), badge: document.getElementById('cVal') },
    gamma: { el: document.getElementById('gSlider'), badge: document.getElementById('gVal') },
    saturation: { el: document.getElementById('sSlider'), badge: document.getElementById('sVal') },
    denoise: { el: document.getElementById('dSlider'), badge: document.getElementById('dVal') },
  },
  alignCheck: document.getElementById('alignCheck'),
  resetBtn: document.getElementById('resetBtn'),

  init() {
    Object.values(this.sliders).forEach(s => {
      s.el.addEventListener('input', () => {
        s.badge.textContent = parseFloat(s.el.value).toFixed(2);
        applyLivePreview();
        scheduleReEnhance();
      });
    });

    this.alignCheck.addEventListener('change', () => scheduleReEnhance());

    this.resetBtn.addEventListener('click', () => {
      this.resetToDefaults();
      clearLiveFilter();
      scheduleReEnhance();
    });
  },

  resetToDefaults() {
    this.sliders.enhance.el.value = '1'; this.sliders.enhance.badge.textContent = '1.00';
    this.sliders.brightness.el.value = '1.50'; this.sliders.brightness.badge.textContent = '1.50';
    this.sliders.sharpness.el.value = '2.00'; this.sliders.sharpness.badge.textContent = '2.00';
    this.sliders.contrast.el.value = '1.00'; this.sliders.contrast.badge.textContent = '1.00';
    this.sliders.gamma.el.value = '1.30'; this.sliders.gamma.badge.textContent = '1.30';
    this.sliders.saturation.el.value = '1.00'; this.sliders.saturation.badge.textContent = '1.00';
    this.sliders.denoise.el.value = '0.00'; this.sliders.denoise.badge.textContent = '0.00';
    this.alignCheck.checked = false;
  },

  getValues() {
    return {
      enhance: this.sliders.enhance.el.value,
      brightness: this.sliders.brightness.el.value,
      sharpness: this.sliders.sharpness.el.value,
      contrast: this.sliders.contrast.el.value,
      gamma: this.sliders.gamma.el.value,
      saturation: this.sliders.saturation.el.value,
      denoise: this.sliders.denoise.el.value,
      auto_align: this.alignCheck.checked ? 'true' : 'false'
    };
  }
};

const OutputDisplay = {
  resultsFeed: document.getElementById('resultsFeed'),
  placeholder: document.getElementById('placeholder'),

  createResultCard(file) {
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
    this.resultsFeed.prepend(card);
    file.cardId = cardId;
  },

  updateCard(cardId, data, error) {
    const card = document.getElementById(cardId);
    if (!card) return;
    const statusEl = card.querySelector('.status');
    const infoEl = card.querySelector('.info');
    const enhImgEl = card.querySelector('.enhanced-img');
    const dlBtn = card.querySelector('.dl-btn');

    if (error) {
      statusEl.textContent = 'Error';
      statusEl.className = 'status status-error';
      infoEl.textContent = error;
      return;
    }

    enhImgEl.src = 'data:image/png;base64,' + data.enhanced_b64;
    enhImgEl.setAttribute('data-enhanced', 'true');
    enhImgEl.style.filter = '';
    dlBtn.href = enhImgEl.src;
    dlBtn.classList.remove('hidden');

    statusEl.textContent = 'Done ✓';
    statusEl.className = 'status status-done';
    infoEl.textContent = `${data.width} × ${data.height} px`;
  }
};

// ── Global State & Controller ──────────────────────────────────────────────────
const enhBtn = document.getElementById('enhBtn');
const btnTxt = document.getElementById('btnTxt');
const spinner = document.getElementById('spinner');
const toast = document.getElementById('toast');
const placeholder = document.getElementById('placeholder');

let processingQueue = [];
let isProcessing = false;
let lastFile = null;
let debounceTimer = null;
const DEBOUNCE_MS = 700;

function applyLivePreview() {
  const brightness = parseFloat(PreferencePanel.sliders.brightness.el.value);
  const contrast = parseFloat(PreferencePanel.sliders.contrast.el.value);
  const saturation = parseFloat(PreferencePanel.sliders.saturation.el.value);
  const gamma = parseFloat(PreferencePanel.sliders.gamma.el.value);
  const gammaBrightness = gamma > 1 ? 1 + (gamma - 1) * 0.25 : gamma;

  const cssFilter = `brightness(${brightness * gammaBrightness}) contrast(${contrast}) saturate(${saturation})`;
  document.querySelectorAll('.enhanced-img').forEach(img => {
    if (img.getAttribute('data-enhanced') === 'true') {
      img.style.filter = cssFilter;
    }
  });
}

function clearLiveFilter() {
  document.querySelectorAll('.enhanced-img').forEach(img => { img.style.filter = ''; });
}

function scheduleReEnhance() {
  clearTimeout(debounceTimer);
  if (!lastFile) return;
  debounceTimer = setTimeout(async () => {
    clearLiveFilter();
    await processFile(lastFile);
  }, DEBOUNCE_MS);
}

enhBtn.addEventListener('click', async () => {
  if (isProcessing || processingQueue.length === 0) return;
  isProcessing = true;
  enhBtn.disabled = true;
  spinner.classList.remove('hidden');
  btnTxt.textContent = 'Enhancing...';

  while (processingQueue.length > 0) {
    const file = processingQueue.shift();
    await processFile(file);
    ImageUploadModule.fileNameEl.textContent = `${processingQueue.length} images remaining`;
  }

  isProcessing = false;
  enhBtn.disabled = false;
  spinner.classList.add('hidden');
  btnTxt.textContent = 'Enhance All';
  ImageUploadModule.fileNameEl.textContent = 'Batch Complete';
});

async function processFile(file) {
  const card = document.getElementById(file.cardId);
  if (!card) return;
  const statusEl = card.querySelector('.status');
  const infoEl = card.querySelector('.info');
  statusEl.textContent = 'Processing…';
  statusEl.className = 'status status-processing';
  infoEl.textContent = 'Sending to model…';

  const form = new FormData();
  form.append('file', file);
  const prefs = PreferencePanel.getValues();
  Object.keys(prefs).forEach(k => form.append(k, prefs[k]));

  try {
    const res = await fetch('/enhance', { method: 'POST', body: form });
    const contentType = res.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
      const text = await res.text();
      throw new Error(`Server error ${res.status}`);
    }
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Server error');

    OutputDisplay.updateCard(file.cardId, data);
    lastFile = file;
  } catch (err) {
    OutputDisplay.updateCard(file.cardId, null, err.message);
  }
}

function showError(msg) {
  toast.textContent = '⚠ ' + msg;
  toast.classList.remove('hidden');
  setTimeout(() => toast.classList.add('hidden'), 5000);
}

// ── Boot ──
ImageUploadModule.init();
PreferencePanel.init();

