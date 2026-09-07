/* ==========================================
   TaxTrack Dashboard JavaScript
   ========================================== */

const API = "";  // Same origin

// ─── Tab Navigation ────────────────────────

document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const tab = link.dataset.tab;
        switchTab(tab);
    });
});

function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    document.getElementById(`tab-${tabName}`).classList.add('active');
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

    // Load data when switching tabs
    if (tabName === 'dashboard') loadStats();
    if (tabName === 'vehicles') loadVehicles();
    if (tabName === 'camera') loadCameraStatus();
}

// ─── Dashboard Stats ──────────────────────

async function loadStats() {
    try {
        const res = await fetch(`${API}/api/stats`);
        const data = await res.json();

        document.getElementById('statTotal').textContent = data.total_vehicles;
        document.getElementById('statPaid').textContent = data.tax_paid;
        document.getElementById('statUnpaid').textContent = data.tax_unpaid;
        document.getElementById('statExpired').textContent = data.tax_expired;
        document.getElementById('statPending').textContent = data.tax_pending;
        document.getElementById('statScans').textContent = data.total_scans;

        // Recent scans table
        const tbody = document.getElementById('recentScansBody');
        if (data.recent_scans.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No scans yet</td></tr>';
            return;
        }

        tbody.innerHTML = data.recent_scans.map(scan => `
            <tr>
                <td><strong>${esc(scan.plate_number)}</strong></td>
                <td>${esc(scan.ocr_text)}</td>
                <td>${statusBadge(scan.tax_status)}</td>
                <td>${(scan.ocr_confidence * 100).toFixed(1)}%</td>
                <td>${formatDate(scan.scanned_at)}</td>
            </tr>
        `).join('');
    } catch (err) {
        console.error('Failed to load stats:', err);
    }
}

// ─── Vehicle Management ───────────────────

let searchTimeout;
function debouncedLoadVehicles() {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(loadVehicles, 300);
}

async function loadVehicles() {
    try {
        const search = document.getElementById('searchInput').value;
        const status = document.getElementById('statusFilter').value;
        const params = new URLSearchParams();
        if (search) params.set('search', search);
        if (status) params.set('status', status);

        const res = await fetch(`${API}/api/vehicles/?${params}`);
        const vehicles = await res.json();

        const tbody = document.getElementById('vehiclesBody');
        if (vehicles.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No vehicles found</td></tr>';
            return;
        }

        tbody.innerHTML = vehicles.map(v => `
            <tr>
                <td>${v.id}</td>
                <td><strong>${esc(v.plate_number)}</strong></td>
                <td>${esc(v.owner_name)}</td>
                <td>${esc(v.vehicle_type)}</td>
                <td>${esc(v.vehicle_brand || '--')}</td>
                <td>${statusBadge(v.tax_status)}</td>
                <td>
                    <button class="btn btn-danger btn-sm" onclick="deleteVehicle(${v.id})">Delete</button>
                </td>
            </tr>
        `).join('');
    } catch (err) {
        console.error('Failed to load vehicles:', err);
    }
}

function showAddVehicleModal() {
    document.getElementById('addVehicleModal').classList.add('show');
}

function closeModal(id) {
    document.getElementById(id).classList.remove('show');
}

async function addVehicle(e) {
    e.preventDefault();
    const form = e.target;
    const data = Object.fromEntries(new FormData(form));

    try {
        const res = await fetch(`${API}/api/vehicles/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });

        if (res.ok) {
            toast('Vehicle registered successfully!', 'success');
            closeModal('addVehicleModal');
            form.reset();
            loadVehicles();
            loadStats();
        } else {
            const err = await res.json();
            toast(err.detail || 'Failed to register vehicle', 'error');
        }
    } catch (err) {
        toast('Network error', 'error');
    }
}

async function deleteVehicle(id) {
    if (!confirm('Delete this vehicle and all its records?')) return;

    try {
        const res = await fetch(`${API}/api/vehicles/${id}`, { method: 'DELETE' });
        if (res.ok) {
            toast('Vehicle deleted', 'success');
            loadVehicles();
            loadStats();
        } else {
            toast('Failed to delete vehicle', 'error');
        }
    } catch (err) {
        toast('Network error', 'error');
    }
}

// ─── Image Upload & Scan ──────────────────

const fileInput = document.getElementById('imageFile');
const dropZone = document.getElementById('dropZone');
const preview = document.getElementById('imagePreview');
const placeholder = document.getElementById('uploadPlaceholder');
const scanBtn = document.getElementById('scanBtn');

fileInput.addEventListener('change', handleFileSelect);

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelect();
    }
});

function handleFileSelect() {
    const file = fileInput.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = 'block';
        placeholder.style.display = 'none';
        scanBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

async function scanImage() {
    const file = fileInput.files[0];
    if (!file) return;

    scanBtn.disabled = true;
    scanBtn.textContent = 'Processing...';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`${API}/api/detect/scan`, {
            method: 'POST',
            body: formData,
        });

        if (res.ok) {
            const data = await res.json();
            showScanResult(data);
            toast('Scan complete!', 'success');
            loadStats();
        } else {
            const err = await res.json();
            toast(err.detail || 'Scan failed', 'error');
        }
    } catch (err) {
        toast('Network error during scan', 'error');
    } finally {
        scanBtn.disabled = false;
        scanBtn.textContent = 'Run Detection + OCR';
    }
}

async function lookupPlate() {
    const plate = document.getElementById('plateInput').value.trim();
    if (!plate) {
        toast('Enter a plate number', 'error');
        return;
    }

    try {
        const res = await fetch(`${API}/api/detect/scan/plate-text?plate=${encodeURIComponent(plate)}`, {
            method: 'POST',
        });

        if (res.ok) {
            const data = await res.json();
            showScanResult(data);
            toast('Lookup complete!', 'success');
            loadStats();
        } else {
            const err = await res.json();
            toast(err.detail || 'Lookup failed', 'error');
        }
    } catch (err) {
        toast('Network error', 'error');
    }
}

function showScanResult(data) {
    const panel = document.getElementById('scanResult');
    panel.style.display = 'block';

    document.getElementById('resultPlate').textContent = data.plate_number;
    document.getElementById('resultFound').textContent = data.vehicle_found ? 'Yes' : 'No';
    document.getElementById('resultDetConf').textContent = (data.detection_confidence * 100).toFixed(1) + '%';
    document.getElementById('resultOcrConf').textContent = (data.ocr_confidence * 100).toFixed(1) + '%';

    const statusEl = document.getElementById('resultStatus');
    if (data.tax_status) {
        statusEl.innerHTML = statusBadge(data.tax_status);
    } else {
        statusEl.textContent = data.vehicle_found ? 'Unknown' : 'Not registered';
    }

    const ownerInfo = document.getElementById('ownerInfo');
    if (data.vehicle_found && data.vehicle) {
        ownerInfo.style.display = 'flex';
        document.getElementById('resultOwner').textContent = data.vehicle.owner_name;
    } else {
        ownerInfo.style.display = 'none';
    }

    panel.scrollIntoView({ behavior: 'smooth' });
}

// ─── Camera Feed ───────────────────────────

let streamActive = false;
let detectedPlatesMap = {};  // plate_text → {text, confidence, taxStatus, ...}
let totalDetections = 0;
let platePollInterval = null;
let selectedSource = 'demo';

function selectSource(source) {
    selectedSource = source;
    document.getElementById('cameraSource').value = source;
    document.getElementById('srcDemoBtn').classList.toggle('active', source === 'demo');
    document.getElementById('srcWebcamBtn').classList.toggle('active', source === 'webcam');

    if (streamActive) {
        stopStream();
        const delay = source === 'webcam' ? 800 : 300;
        setTimeout(startStream, delay);
    }
}

function startStream() {
    const source = document.getElementById('cameraSource').value;
    const img = document.getElementById('cameraStream');
    const placeholder = document.getElementById('cameraPlaceholder');

    const ts = Date.now();
    img.src = `${API}/api/camera/stream?source=${encodeURIComponent(source)}&_=${ts}`;
    img.style.display = 'block';
    placeholder.style.display = 'none';

    document.getElementById('startStreamBtn').disabled = true;
    document.getElementById('stopStreamBtn').disabled = false;
    document.getElementById('statusDot').classList.add('live');

    const isWebcam = source === 'webcam';
    document.getElementById('statusText').textContent = isWebcam ? 'Connecting to webcam...' : 'Streaming...';

    if (isWebcam) {
        img.onload = () => {
            document.getElementById('statusText').textContent = 'Webcam Live';
        };
    } else {
        document.getElementById('statusText').textContent = 'Streaming...';
    }

    streamActive = true;
    startPlatePolling();

    toast(isWebcam ? 'Connecting to webcam...' : 'Traffic demo feed started', 'success');
}

function stopStream() {
    const img = document.getElementById('cameraStream');
    const placeholder = document.getElementById('cameraPlaceholder');

    img.src = '';
    img.style.display = 'none';
    placeholder.style.display = 'flex';

    document.getElementById('startStreamBtn').disabled = false;
    document.getElementById('stopStreamBtn').disabled = true;
    document.getElementById('statusDot').classList.remove('live');
    document.getElementById('statusText').textContent = 'Stopped';

    streamActive = false;

    if (platePollInterval) {
        clearInterval(platePollInterval);
        platePollInterval = null;
    }

    toast('Camera feed stopped', 'info');
}

async function loadCameraStatus() {
    try {
        const res = await fetch(`${API}/api/camera/status`);
        const data = await res.json();
        document.getElementById('cameraMode').textContent = data.mode;
    } catch (err) {
        document.getElementById('cameraMode').textContent = 'Unknown';
    }
}

function startPlatePolling() {
    if (platePollInterval) clearInterval(platePollInterval);
    platePollInterval = setInterval(fetchDetections, 2000);
}

async function fetchDetections() {
    if (!streamActive) return;

    try {
        const res = await fetch(`${API}/api/camera/detections`);
        const data = await res.json();

        if (!data.detections || data.detections.length === 0) return;

        // Process each detection with a confirmed plate
        let newPlateFound = false;
        for (const det of data.detections) {
            if (!det.plate_text || det.plate_text.length < 4) continue;

            const key = det.plate_text;
            if (!detectedPlatesMap[key]) {
                detectedPlatesMap[key] = {
                    text: det.plate_text,
                    confidence: det.plate_confidence,
                    taxStatus: det.tax_status,
                    vehicleFound: det.vehicle_found || false,
                    trackId: det.track_id,
                    time: new Date().toISOString(),
                };
                addPlateEntry(detectedPlatesMap[key]);
                newPlateFound = true;
            } else {
                // Update confidence/tax status if better
                const existing = detectedPlatesMap[key];
                if (det.plate_confidence > existing.confidence) {
                    existing.confidence = det.plate_confidence;
                }
                if (det.tax_status) {
                    existing.taxStatus = det.tax_status;
                }
            }
        }

        // Update tracking count in status
        if (data.tracking_count !== undefined) {
            document.getElementById('statusText').textContent =
                `Streaming — Tracking ${data.tracking_count} vehicle(s)`;
        }

    } catch (err) {
        // Silently ignore polling errors
    }
}

function addPlateEntry(entry) {
    const log = document.getElementById('plateLog');
    const empty = document.getElementById('plateLogEmpty');
    if (empty) empty.remove();

    totalDetections++;

    const el = document.createElement('div');
    const matched = entry.vehicleFound;
    el.className = `plate-log-entry ${matched ? 'plate-match' : 'plate-nomatch'}`;

    let statusIcon;
    if (!matched) {
        statusIcon = '<span class="badge badge-pending">NOT FOUND</span>';
    } else if (entry.taxStatus === 'paid') {
        statusIcon = '<span class="badge badge-paid">PAID</span>';
    } else if (entry.taxStatus === 'expired') {
        statusIcon = '<span class="badge badge-expired">EXPIRED</span>';
    } else if (entry.taxStatus === 'unpaid') {
        statusIcon = '<span class="badge badge-unpaid">UNPAID</span>';
    } else {
        statusIcon = '<span class="badge badge-pending">UNKNOWN</span>';
    }

    el.innerHTML = `
        <div>
            <span class="plate-text">${esc(entry.text)}</span>
            <span class="plate-conf">${(entry.confidence * 100).toFixed(0)}%</span>
            <span class="plate-track-id">ID:${entry.trackId || '--'}</span>
        </div>
        <div style="display:flex;align-items:center;gap:0.5rem">
            ${statusIcon}
            <span class="plate-time">${formatTime(entry.time)}</span>
        </div>
    `;

    log.insertBefore(el, log.firstChild);

    // Keep max 50 entries
    while (log.children.length > 50) {
        log.removeChild(log.lastChild);
    }

    // Update stats
    document.getElementById('camTotalDetections').textContent = totalDetections;
    document.getElementById('camUniquePlates').textContent = Object.keys(detectedPlatesMap).length;
    document.getElementById('camLastPlate').textContent = entry.text;
}

function clearPlateLog() {
    const log = document.getElementById('plateLog');
    log.innerHTML = '<div class="empty-state" id="plateLogEmpty">Log cleared.<br>New detections will appear here.</div>';
    detectedPlatesMap = {};
    totalDetections = 0;
    document.getElementById('camTotalDetections').textContent = '0';
    document.getElementById('camUniquePlates').textContent = '0';
    document.getElementById('camLastPlate').textContent = '--';
}

function formatTime(dateStr) {
    if (!dateStr) return '--';
    const d = new Date(dateStr);
    return d.toLocaleTimeString();
}

// ─── Utility Functions ────────────────────

function statusBadge(status) {
    if (!status) return '<span class="badge badge-pending">UNKNOWN</span>';
    const s = status.toLowerCase();
    return `<span class="badge badge-${s}">${s.toUpperCase()}</span>`;
}

function formatDate(dateStr) {
    if (!dateStr) return '--';
    const d = new Date(dateStr);
    return d.toLocaleString();
}

function esc(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function toast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => el.remove(), 4000);
}

// ─── Init ─────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    loadStats();
});
