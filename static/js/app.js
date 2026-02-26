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
      });
    });

    this.alignCheck.addEventListener('change', () => applyLivePreview());

    this.resetBtn.addEventListener('click', () => {
      this.resetToDefaults();
      clearLiveFilter();
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
  cardFiles: new Map(), // Track File object for each cardId

  createResultCard(file) {
    const cardId = 'card-' + Math.random().toString(36).substr(2, 9);
    this.cardFiles.set(cardId, file);

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
        <div class="actions">
          <button class="btn-ghost btn-sm apply-btn hidden" title="Apply current slider settings to this image">Re-Apply Settings</button>
          <a class="btn-primary btn-sm dl-btn hidden" download="enhanced_${file.name}">Download</a>
        </div>
      </div>
    `;

    card.querySelector('.apply-btn').onclick = () => {
      clearLiveFilter();
      processFile(file);
    };

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
    const applyBtn = card.querySelector('.apply-btn');

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
    applyBtn.classList.remove('hidden');

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
  const applyBtn = card.querySelector('.apply-btn');

  statusEl.textContent = 'Uploading…';
  statusEl.className = 'status status-processing';
  if (applyBtn) applyBtn.disabled = true;

  const form = new FormData();
  form.append('file', file);
  const prefs = PreferencePanel.getValues();
  Object.keys(prefs).forEach(k => form.append(k, prefs[k]));

  try {
    // 1. Submit task to server
    const res = await fetch('/enhance', { method: 'POST', body: form });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    const { task_id, error } = await res.json();
    if (error) throw new Error(error);

    // 2. Poll for results (Bypasses browser timeout)
    statusEl.textContent = 'Processing…';
    infoEl.textContent = 'Enhancing in background...';

    let completed = false;
    let errorCount = 0;
    while (!completed) {
      await new Promise(r => setTimeout(r, 4000)); // Poll every 4 seconds

      try {
        const statusRes = await fetch(`/task_status/${task_id}`);
        if (!statusRes.ok) throw new Error("Lost contact with server");

        const data = await statusRes.json();
        errorCount = 0;

        if (data.status === 'complete') {
          OutputDisplay.updateCard(file.cardId, data.result);
          lastFile = file;
          completed = true;
        } else if (data.status === 'error') {
          throw new Error(data.message || "Model error");
        }
      } catch (pollErr) {
        errorCount++;
        console.warn(`Retry ${errorCount}/10:`, pollErr);
        if (errorCount > 10) throw new Error("Connection lost after multiple retries.");
        infoEl.textContent = `Re-connecting (${errorCount}/10)...`;
      }
    }
  } catch (err) {
    console.error("Task failed:", err);
    OutputDisplay.updateCard(file.cardId, null, err.message);
  } finally {
    if (applyBtn) applyBtn.disabled = false;
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
