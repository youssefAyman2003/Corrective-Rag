/**
 * Corrective RAG (CRAG) Frontend Application
 * Interacts with FastAPI backend for querying, indexing, and workflow visualization.
 */

// Configuration State
const state = {
  apiBaseUrl: localStorage.getItem('crag_api_url') || (window.location.origin.startsWith('http') ? window.location.origin : 'http://localhost:8000'),
  history: JSON.parse(localStorage.getItem('crag_history') || '[]'),
  defaultUrls: [
    'https://lilianweng.github.io/posts/2023-06-23-agent/',
    'https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/',
    'https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/'
  ],
  isQuerying: false,
  isIndexing: false
};

// DOM Elements
const elements = {
  // Navigation
  navTabs: document.querySelectorAll('.nav-tab'),
  tabPanels: document.querySelectorAll('.tab-panel'),

  // Status
  serverStatusPill: document.getElementById('serverStatusPill'),
  telemGroq: document.getElementById('telemGroq'),
  telemOllama: document.getElementById('telemOllama'),
  telemOllamaBadge: document.getElementById('telemOllamaBadge'),
  telemVectorstore: document.getElementById('telemVectorstore'),
  telemVectorBadge: document.getElementById('telemVectorBadge'),

  // Query Form
  queryForm: document.getElementById('queryForm'),
  questionInput: document.getElementById('questionInput'),
  submitQueryBtn: document.getElementById('submitQueryBtn'),
  presetChips: document.querySelectorAll('.chip'),

  // Results
  resultCard: document.getElementById('resultCard'),
  decisionBadge: document.getElementById('decisionBadge'),
  decisionBadgeText: document.getElementById('decisionBadgeText'),
  latencyText: document.getElementById('latencyText'),
  answerContent: document.getElementById('answerContent'),
  copyAnswerBtn: document.getElementById('copyAnswerBtn'),

  // Stepper
  stepRetrieve: document.getElementById('stepRetrieve'),
  stepGrade: document.getElementById('stepGrade'),
  stepCorrection: document.getElementById('stepCorrection'),
  correctionStepSub: document.getElementById('correctionStepSub'),
  stepGenerate: document.getElementById('stepGenerate'),

  // Documents
  toggleDocsHeader: document.getElementById('toggleDocsHeader'),
  documentsSection: document.querySelector('.documents-section'),
  docsCountPill: document.getElementById('docsCountPill'),
  documentsList: document.getElementById('documentsList'),

  // Indexer
  urlsList: document.getElementById('urlsList'),
  addUrlBtn: document.getElementById('addUrlBtn'),
  resetUrlsBtn: document.getElementById('resetUrlsBtn'),
  triggerIndexBtn: document.getElementById('triggerIndexBtn'),
  indexStatusCard: document.getElementById('indexStatusCard'),
  indexStatusMsg: document.getElementById('indexStatusMsg'),
  indexChunksCount: document.getElementById('indexChunksCount'),
  indexEmbedModel: document.getElementById('indexEmbedModel'),

  // Graph
  mermaidContainer: document.getElementById('mermaidContainer'),
  refreshGraphBtn: document.getElementById('refreshGraphBtn'),

  // History
  historyList: document.getElementById('historyList'),
  clearHistoryBtn: document.getElementById('clearHistoryBtn'),

  // Settings Modal
  openSettingsBtn: document.getElementById('openSettingsBtn'),
  closeSettingsBtn: document.getElementById('closeSettingsBtn'),
  cancelSettingsBtn: document.getElementById('cancelSettingsBtn'),
  saveSettingsBtn: document.getElementById('saveSettingsBtn'),
  settingsModal: document.getElementById('settingsModal'),
  apiUrlInput: document.getElementById('apiUrlInput'),

  // Toasts
  toastContainer: document.getElementById('toastContainer')
};

// Initialize Mermaid.js
if (window.mermaid) {
  mermaid.initialize({
    startOnLoad: false,
    theme: 'dark',
    themeVariables: {
      primaryColor: '#2563eb',
      primaryTextColor: '#f8fafc',
      primaryBorderColor: '#38bdf8',
      lineColor: '#64748b',
      secondaryColor: '#0f172a',
      tertiaryColor: '#1e293b'
    }
  });
}

// -----------------------------------------------------------------------------
// Toast Notifications
// -----------------------------------------------------------------------------
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  elements.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 250);
  }, 3500);
}

// -----------------------------------------------------------------------------
// API Health & Status
// -----------------------------------------------------------------------------
async function checkHealth() {
  try {
    const res = await fetch(`${state.apiBaseUrl}/health`, { method: 'GET' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    elements.serverStatusPill.className = 'status-indicator online';
    elements.serverStatusPill.querySelector('.status-label').textContent = 'Online';

    if (data.groq_model) elements.telemGroq.textContent = data.groq_model;
    if (data.ollama_model) {
      elements.telemOllama.textContent = data.ollama_model;
      elements.telemOllamaBadge.textContent = 'Active';
      elements.telemOllamaBadge.className = 'status-badge status-ok';
    }

    if (data.vectorstore_ready) {
      elements.telemVectorBadge.textContent = 'Ready';
      elements.telemVectorBadge.className = 'status-badge status-ok';
    } else {
      elements.telemVectorBadge.textContent = 'Not Indexed';
      elements.telemVectorBadge.className = 'status-badge status-warning';
    }

    return true;
  } catch (err) {
    elements.serverStatusPill.className = 'status-indicator offline';
    elements.serverStatusPill.querySelector('.status-label').textContent = 'Offline';
    elements.telemOllamaBadge.textContent = 'Unreachable';
    elements.telemOllamaBadge.className = 'status-badge';
    elements.telemVectorBadge.textContent = 'Unknown';
    elements.telemVectorBadge.className = 'status-badge';
    return false;
  }
}

// -----------------------------------------------------------------------------
// Tab Switching
// -----------------------------------------------------------------------------
function setupTabs() {
  elements.navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.dataset.tab;

      elements.navTabs.forEach(t => t.classList.remove('active'));
      elements.tabPanels.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) targetPanel.classList.add('active');

      if (targetId === 'graph-tab') {
        renderGraphDiagram();
      }
    });
  });
}

// -----------------------------------------------------------------------------
// Query Assistant
// -----------------------------------------------------------------------------
async function submitQuery(question) {
  if (!question || state.isQuerying) return;

  state.isQuerying = true;
  const submitBtn = elements.submitQueryBtn;
  submitBtn.disabled = true;
  submitBtn.querySelector('.btn-text').textContent = 'Evaluating & Generating...';
  submitBtn.querySelector('.spinner').style.display = 'inline-block';
  submitBtn.querySelector('.btn-icon').style.display = 'none';

  // Show result card with initial analyzing state
  elements.resultCard.style.display = 'block';
  elements.answerContent.innerHTML = '<p class="text-muted">Querying local FAISS vectorstore and evaluating document relevance with Groq...</p>';
  elements.decisionBadge.className = 'decision-badge';
  elements.decisionBadgeText.textContent = 'Evaluating Relevance...';
  elements.latencyText.textContent = '...';

  // Reset stepper nodes
  setStepState(elements.stepRetrieve, 'active');
  setStepState(elements.stepGrade, 'pending');
  setStepState(elements.stepCorrection, 'pending');
  setStepState(elements.stepGenerate, 'pending');

  const startTime = performance.now();

  try {
    const res = await fetch(`${state.apiBaseUrl}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });

    const endTime = performance.now();
    const duration = Math.round(endTime - startTime);
    elements.latencyText.textContent = `${duration} ms`;

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned ${res.status}`);
    }

    const data = await res.json();
    renderQueryResult(data);
    addToHistory(question, data.web_search_used, duration);
  } catch (err) {
    elements.answerContent.innerHTML = `<p style="color: var(--accent-rose);"><strong>Execution Error:</strong> ${escapeHtml(err.message)}</p>`;
    elements.decisionBadgeText.textContent = 'Failed';
    elements.decisionBadge.className = 'decision-badge';
    showToast(err.message, 'error');
  } finally {
    state.isQuerying = false;
    submitBtn.disabled = false;
    submitBtn.querySelector('.btn-text').textContent = 'Generate Answer';
    submitBtn.querySelector('.spinner').style.display = 'none';
    submitBtn.querySelector('.btn-icon').style.display = 'inline-block';
  }
}

function setStepState(element, stateName) {
  element.classList.remove('completed', 'active', 'fallback', 'pending');
  if (stateName) element.classList.add(stateName);
}

function renderQueryResult(data) {
  // Stepper updates
  setStepState(elements.stepRetrieve, 'completed');
  setStepState(elements.stepGrade, 'completed');

  if (data.web_search_used) {
    setStepState(elements.stepCorrection, 'fallback');
    elements.correctionStepSub.textContent = 'Tavily Web Search';
    elements.decisionBadge.className = 'decision-badge web';
    elements.decisionBadgeText.textContent = 'Self-Corrected: Web Search Fallback Triggered';
  } else {
    setStepState(elements.stepCorrection, 'completed');
    elements.correctionStepSub.textContent = 'Local Verification OK';
    elements.decisionBadge.className = 'decision-badge local';
    elements.decisionBadgeText.textContent = 'Grounded from Local Vectorstore';
  }

  setStepState(elements.stepGenerate, 'completed');

  // Format generated markdown paragraphs
  const formattedHtml = formatAnswerText(data.generation);
  elements.answerContent.innerHTML = formattedHtml;

  // Documents
  const docs = data.documents || [];
  elements.docsCountPill.textContent = docs.length;
  elements.documentsList.innerHTML = '';

  if (docs.length === 0) {
    elements.documentsList.innerHTML = '<div class="text-muted" style="font-size:0.8rem; padding: 0.5rem;">No documents attached.</div>';
  } else {
    docs.forEach((doc, idx) => {
      const isWeb = data.web_search_used && idx === docs.length - 1;
      const card = document.createElement('div');
      card.className = 'doc-card';
      card.innerHTML = `
        <div class="doc-card-header">
          <span class="doc-source-tag ${isWeb ? 'web' : ''}">${isWeb ? 'Tavily Web Search Result' : `Document Passage #${idx + 1}`}</span>
          <span class="text-muted">${doc.length} characters</span>
        </div>
        <div class="doc-card-body">${escapeHtml(doc)}</div>
      `;
      elements.documentsList.appendChild(card);
    });
  }
}

function formatAnswerText(text) {
  if (!text) return '<p>No answer text produced.</p>';
  // Simple paragraph & bullet point formatter
  const lines = text.split('\n\n');
  return lines.map(paragraph => {
    const trimmed = paragraph.trim();
    if (!trimmed) return '';
    return `<p>${escapeHtml(trimmed)}</p>`;
  }).join('');
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// -----------------------------------------------------------------------------
// Document Indexer Management
// -----------------------------------------------------------------------------
function renderUrlInputs(urls) {
  elements.urlsList.innerHTML = '';
  urls.forEach((url, i) => {
    const row = document.createElement('div');
    row.className = 'url-row';
    row.innerHTML = `
      <input type="url" class="url-input" value="${escapeHtml(url)}" placeholder="https://example.com/article" required>
      <button type="button" class="remove-url-btn" title="Remove URL">&times;</button>
    `;
    row.querySelector('.remove-url-btn').addEventListener('click', () => {
      if (elements.urlsList.children.length > 1) {
        row.remove();
      } else {
        showToast('At least one URL is required.', 'error');
      }
    });
    elements.urlsList.appendChild(row);
  });
}

async function triggerIndexing() {
  if (state.isIndexing) return;

  const urlInputs = Array.from(elements.urlsList.querySelectorAll('.url-input'));
  const urls = urlInputs.map(input => input.value.trim()).filter(Boolean);

  if (urls.length === 0) {
    showToast('Please specify at least one valid URL to index.', 'error');
    return;
  }

  state.isIndexing = true;
  const btn = elements.triggerIndexBtn;
  btn.disabled = true;
  btn.querySelector('.btn-text').textContent = 'Scraping, Chunking & Embedding...';
  btn.querySelector('.spinner').style.display = 'inline-block';
  btn.querySelector('.btn-icon').style.display = 'none';

  try {
    const res = await fetch(`${state.apiBaseUrl}/index`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ urls })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    elements.indexStatusCard.style.display = 'block';
    elements.indexStatusMsg.textContent = data.message || 'Indexed Successfully';
    elements.indexChunksCount.textContent = data.chunks_count || 'N/A';
    elements.indexEmbedModel.textContent = 'bge-m3:latest';

    showToast('FAISS Vectorstore updated successfully!', 'success');
    checkHealth();
  } catch (err) {
    showToast(`Indexing failed: ${err.message}`, 'error');
  } finally {
    state.isIndexing = false;
    btn.disabled = false;
    btn.querySelector('.btn-text').textContent = 'Build / Refresh FAISS Index';
    btn.querySelector('.spinner').style.display = 'none';
    btn.querySelector('.btn-icon').style.display = 'inline-block';
  }
}

// -----------------------------------------------------------------------------
// Graph Workflow Visualization
// -----------------------------------------------------------------------------
async function renderGraphDiagram() {
  try {
    const res = await fetch(`${state.apiBaseUrl}/graph`);
    if (res.ok) {
      const mermaidCode = await res.text();
      if (mermaidCode && window.mermaid) {
        elements.mermaidContainer.removeAttribute('data-processed');
        elements.mermaidContainer.textContent = mermaidCode;
        mermaid.run({ nodes: [elements.mermaidContainer] });
        return;
      }
    }
  } catch (err) {
    console.warn('Could not fetch remote mermaid diagram, using default layout.', err);
  }

  // Fallback default diagram rendering
  if (window.mermaid) {
    mermaid.run({ nodes: [elements.mermaidContainer] });
  }
}

// -----------------------------------------------------------------------------
// History Management
// -----------------------------------------------------------------------------
function addToHistory(question, webSearchUsed, latency) {
  const item = {
    question,
    webSearchUsed,
    latency,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  };

  // Add to front of history
  state.history.unshift(item);
  if (state.history.length > 15) state.history.pop();
  localStorage.setItem('crag_history', JSON.stringify(state.history));
  renderHistory();
}

function renderHistory() {
  const list = elements.historyList;
  list.innerHTML = '';

  if (state.history.length === 0) {
    list.innerHTML = '<div class="empty-history">No queries submitted yet.</div>';
    return;
  }

  state.history.forEach(item => {
    const div = document.createElement('div');
    div.className = 'history-item';
    div.innerHTML = `
      <div class="history-question">${escapeHtml(item.question)}</div>
      <div class="history-meta">
        <span style="color: ${item.webSearchUsed ? 'var(--accent-amber)' : 'var(--accent-emerald)'}">
          ${item.webSearchUsed ? 'Web Search' : 'Local Only'}
        </span>
        <span>${item.latency} ms &bull; ${item.timestamp}</span>
      </div>
    `;
    div.addEventListener('click', () => {
      elements.questionInput.value = item.question;
      // Switch to assistant tab
      document.querySelector('[data-tab="query-tab"]').click();
      elements.questionInput.focus();
    });
    list.appendChild(div);
  });
}

// -----------------------------------------------------------------------------
// Settings Modal
// -----------------------------------------------------------------------------
function setupSettings() {
  elements.apiUrlInput.value = state.apiBaseUrl;

  elements.openSettingsBtn.addEventListener('click', () => {
    elements.apiUrlInput.value = state.apiBaseUrl;
    elements.settingsModal.classList.add('open');
  });

  const closeModal = () => elements.settingsModal.classList.remove('open');
  elements.closeSettingsBtn.addEventListener('click', closeModal);
  elements.cancelSettingsBtn.addEventListener('click', closeModal);

  elements.saveSettingsBtn.addEventListener('click', () => {
    let url = elements.apiUrlInput.value.trim();
    if (url.endsWith('/')) url = url.slice(0, -1);
    if (!url) url = 'http://localhost:8000';

    state.apiBaseUrl = url;
    localStorage.setItem('crag_api_url', url);
    closeModal();
    showToast(`API URL updated to: ${url}`, 'info');
    checkHealth();
  });
}

// -----------------------------------------------------------------------------
// Event Listeners & Startup
// -----------------------------------------------------------------------------
function init() {
  setupTabs();
  setupSettings();
  renderUrlInputs(state.defaultUrls);
  renderHistory();

  // Query form submission
  elements.queryForm.addEventListener('submit', e => {
    e.preventDefault();
    const q = elements.questionInput.value.trim();
    if (q) submitQuery(q);
  });

  // Shift + Enter submission
  elements.questionInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      elements.queryForm.dispatchEvent(new Event('submit'));
    }
  });

  // Preset query chips
  elements.presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      elements.questionInput.value = chip.dataset.query;
      elements.questionInput.focus();
    });
  });

  // Copy Answer
  elements.copyAnswerBtn.addEventListener('click', () => {
    const text = elements.answerContent.innerText;
    if (text) {
      navigator.clipboard.writeText(text).then(() => {
        showToast('Answer copied to clipboard!', 'info');
      }).catch(() => {
        showToast('Failed to copy text', 'error');
      });
    }
  });

  // Toggle Documents Drawer
  elements.toggleDocsHeader.addEventListener('click', () => {
    const isOpen = elements.documentsSection.classList.toggle('open');
    elements.documentsList.style.display = isOpen ? 'flex' : 'none';
  });

  // Indexer URL controls
  elements.addUrlBtn.addEventListener('click', () => {
    const row = document.createElement('div');
    row.className = 'url-row';
    row.innerHTML = `
      <input type="url" class="url-input" placeholder="https://example.com/article" required>
      <button type="button" class="remove-url-btn" title="Remove URL">&times;</button>
    `;
    row.querySelector('.remove-url-btn').addEventListener('click', () => row.remove());
    elements.urlsList.appendChild(row);
  });

  elements.resetUrlsBtn.addEventListener('click', () => {
    renderUrlInputs(state.defaultUrls);
    showToast('Reset URLs to defaults', 'info');
  });

  elements.triggerIndexBtn.addEventListener('click', triggerIndexing);
  elements.refreshGraphBtn.addEventListener('click', renderGraphDiagram);

  // Clear History
  elements.clearHistoryBtn.addEventListener('click', () => {
    state.history = [];
    localStorage.removeItem('crag_history');
    renderHistory();
    showToast('Query history cleared', 'info');
  });

  // Initial Health Check & Polling every 15s
  checkHealth();
  setInterval(checkHealth, 15000);
}

document.addEventListener('DOMContentLoaded', init);
