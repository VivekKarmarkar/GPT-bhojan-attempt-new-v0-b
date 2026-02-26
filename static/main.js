const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const previewSection = document.getElementById('preview-section');
const previewImg = document.getElementById('preview-img');
const results = document.getElementById('results');
const annotatedImg = document.getElementById('annotated-img');
const descriptionEl = document.getElementById('description');
const itemsEl = document.getElementById('items');
const resetBtn = document.getElementById('reset-btn');

let radarChart = null;

// --- Upload handlers ---
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

// --- Reset ---
resetBtn.addEventListener('click', () => {
  results.classList.add('hidden');
  previewSection.classList.add('hidden');
  uploadZone.classList.remove('hidden');
  fileInput.value = '';
  if (radarChart) { radarChart.destroy(); radarChart = null; }
});

// --- Core logic ---
async function handleFile(file) {
  if (!file.type.startsWith('image/')) return;

  // Show raw image preview immediately
  uploadZone.classList.add('hidden');
  previewImg.src = URL.createObjectURL(file);
  previewSection.classList.remove('hidden');

  const formData = new FormData();
  formData.append('image', file);

  try {
    const res = await fetch('/analyze', { method: 'POST', body: formData });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();
    showResults(data);
  } catch (err) {
    alert('Analysis failed: ' + err.message);
    previewSection.classList.add('hidden');
    uploadZone.classList.remove('hidden');
  }
}

function showResults(data) {
  previewSection.classList.add('hidden');

  // Annotated image
  annotatedImg.src = 'data:image/png;base64,' + data.annotated_b64;

  // Description
  descriptionEl.textContent = data.description;

  // Radar chart
  renderRadar(data.scores);

  // Item pills
  itemsEl.innerHTML = '';
  data.items.forEach((item) => {
    const pill = document.createElement('span');
    pill.className = 'pill';
    pill.textContent = item.name;
    pill.style.backgroundColor = item.color;
    itemsEl.appendChild(pill);
  });

  results.classList.remove('hidden');
}

function renderRadar(scores) {
  const ctx = document.getElementById('radar-chart').getContext('2d');
  if (radarChart) radarChart.destroy();

  radarChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Health', 'Satiety', 'Bloat', 'Tasty', 'Addiction'],
      datasets: [{
        data: [scores.health, scores.satiety, scores.bloat, scores.tasty, scores.addiction],
        backgroundColor: 'rgba(249, 115, 22, 0.25)',
        borderColor: '#f97316',
        borderWidth: 3,
        pointBackgroundColor: '#f97316',
        pointBorderColor: '#fff',
        pointBorderWidth: 2,
        pointRadius: 6,
        pointHoverRadius: 8,
      }]
    },
    options: {
      scales: {
        r: {
          beginAtZero: true,
          max: 10,
          ticks: {
            stepSize: 2,
            color: '#666',
            backdropColor: 'transparent',
            font: { size: 13, weight: '600' }
          },
          grid: { color: '#2e3140', lineWidth: 1.5 },
          angleLines: { color: '#2e3140', lineWidth: 1.5 },
          pointLabels: {
            color: '#e0e0e0',
            font: { size: 17, weight: '700' },
            padding: 16,
          },
        }
      },
      plugins: {
        legend: { display: false }
      },
      responsive: true,
      maintainAspectRatio: true,
    }
  });
}
