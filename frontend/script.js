/**
 * Legal Assistant — Frontend Logic
 * Handles chat interaction, API calls, and response rendering.
 */

// ==================== DOM ELEMENTS ====================
const chatMessages = document.getElementById('chatMessages');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const scrollBtn = document.getElementById('scrollBtn');

let currentSearchMode = 'hybrid';
let isLoading = false;

// ==================== SEARCH MODE TOGGLE ====================
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentSearchMode = btn.dataset.mode;
    });
});

// ==================== AUTO-RESIZE TEXTAREA ====================
queryInput.addEventListener('input', () => {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + 'px';
});

// ==================== TEMPLATE BUTTONS ====================
const btnAnalyzeCase = document.getElementById('btnAnalyzeCase');
const btnSpotIssues = document.getElementById('btnSpotIssues');

if (btnAnalyzeCase) {
    btnAnalyzeCase.addEventListener('click', () => {
        queryInput.value = "[FACTS]: \n\n[GOAL]: \n\n[EVIDENCE/DOUBTS]: ";
        queryInput.focus();
        queryInput.style.height = 'auto';
        queryInput.style.height = queryInput.scrollHeight + 'px';
    });
}

if (btnSpotIssues) {
    btnSpotIssues.addEventListener('click', () => {
        queryInput.value = "[FACTS]: \n\n[SPECIFIC_QUESTION_OR_CONCERN]: ";
        queryInput.focus();
        queryInput.style.height = 'auto';
        queryInput.style.height = queryInput.scrollHeight + 'px';
    });
}

// ==================== SEND ON ENTER ====================
queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

sendBtn.addEventListener('click', handleSend);

// ==================== SCROLL BUTTON ====================
const chatContainer = document.querySelector('.chat-container');
chatContainer.addEventListener('scroll', () => {
    const isScrolledUp = chatContainer.scrollTop < (chatContainer.scrollHeight - chatContainer.clientHeight - 100);
    scrollBtn.classList.toggle('visible', isScrolledUp);
});

scrollBtn.addEventListener('click', () => {
    chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
});

// ==================== MAIN SEND HANDLER ====================
async function handleSend() {
    if (isLoading) return;
    
    const question = queryInput.value.trim();
    if (!question) return;

    isLoading = true;
    sendBtn.disabled = true;
    queryInput.value = '';
    queryInput.style.height = 'auto';

    // Add user message
    appendMessage('user', question);

    // Add loading indicator
    const loadingEl = appendLoading();

    try {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                search_mode: currentSearchMode,
                min_confidence: 0.35
            })
        });

        // Remove loading
        loadingEl.remove();

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error (${response.status})`);
        }

        const data = await response.json();
        appendAssistantResponse(data);

    } catch (error) {
        appendError(error.message || 'Failed to connect to the server. Please try again.');
    } finally {
        if (loadingEl && loadingEl.parentNode) {
            loadingEl.remove();
        }
        isLoading = false;
        sendBtn.disabled = false;
        queryInput.focus();
    }
}

// ==================== MESSAGE RENDERING ====================

function appendMessage(role, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;

    const avatar = role === 'user' ? '👤' : '⚖️';

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-bubble">
                <p>${escapeHtml(text)}</p>
            </div>
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function appendAssistantResponse(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';

    const formattedAnswer = formatMarkdown(data.answer);
    const confidenceHtml = buildConfidenceBadge(data.confidence, data.confidence_score);
    const groundingHtml = buildGroundingMetrics(data.grounding);
    const queriesHtml = buildExpandedQueries(data.queries_used);
    const citationsHtml = buildCitations(data.citations);
    const warningHtml = data.warning 
        ? `<p class="disclaimer ${data.confidence === 'rejected' ? 'rejected-warning' : ''}">⚠️ ${escapeHtml(data.warning)}</p>` 
        : '';

    messageDiv.innerHTML = `
        <div class="message-avatar">⚖️</div>
        <div class="message-content">
            <div class="message-bubble">
                ${formattedAnswer}
                ${warningHtml}
            </div>
            ${confidenceHtml}
            ${groundingHtml}
            ${queriesHtml}
            ${citationsHtml}
        </div>
    `;

    chatMessages.appendChild(messageDiv);

    // Wire up all toggles
    messageDiv.querySelectorAll('.citations-toggle, .grounding-toggle, .queries-toggle').forEach(toggle => {
        toggle.addEventListener('click', () => {
            toggle.classList.toggle('open');
            const list = toggle.nextElementSibling;
            if (list) list.classList.toggle('open');
        });
    });

    scrollToBottom();
}

function appendLoading() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message loading-message';
    messageDiv.innerHTML = `
        <div class="message-avatar">⚖️</div>
        <div class="message-content">
            <div class="message-bubble">
                <div class="loading-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
}

function appendError(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant-message';
    messageDiv.innerHTML = `
        <div class="message-avatar">⚠️</div>
        <div class="message-content">
            <div class="message-bubble error-bubble">
                <p class="error-text">❌ ${escapeHtml(message)}</p>
                <p style="font-size:12px;color:var(--text-muted);margin-top:8px;">
                    Please check your connection and try again.
                </p>
            </div>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// ==================== UI BUILDERS ====================

function buildConfidenceBadge(level, score) {
    const labels = {
        high: '🟢 High Confidence',
        medium: '🟡 Medium Confidence',
        low: '🔴 Low Confidence',
        very_low: '🔴 Very Low Confidence',
        none: '⚪ No Match',
        rejected: '🛑 Safety Refusal'
    };

    const label = labels[level] || labels.none;
    const pct = (score * 100).toFixed(1);

    return `
        <div class="confidence-bar confidence-${level}">
            <div class="confidence-dot"></div>
            <span>${label} (${pct}%)</span>
        </div>
    `;
}

function buildGroundingMetrics(grounding) {
    if (!grounding) return '';

    const statusIcon = grounding.is_grounded ? '✅' : '⚠️';
    const statusText = grounding.is_grounded ? 'Grounded' : 'Partially Grounded';
    const overall = (grounding.overall_score * 100).toFixed(1);

    const makeBar = (label, value, color) => {
        const pct = (value * 100).toFixed(0);
        return `
            <div style="display:flex;align-items:center;gap:8px;font-size:11px;">
                <span style="min-width:82px;color:var(--text-secondary)">${label}</span>
                <div style="flex:1;height:6px;background:var(--bg-primary);border-radius:3px;overflow:hidden">
                    <div style="width:${pct}%;height:100%;background:${color};border-radius:3px;transition:width 0.5s"></div>
                </div>
                <span style="min-width:36px;text-align:right;color:var(--text-secondary)">${pct}%</span>
            </div>`;
    };

    const ungroundedHtml = grounding.ungrounded_claims && grounding.ungrounded_claims.length > 0
        ? `<div style="margin-top:6px;font-size:11px;color:var(--red);">
            ⚠️ Ungrounded: ${grounding.ungrounded_claims.map(c => escapeHtml(c)).join('; ')}
           </div>`
        : '';

    return `
        <div class="citations-wrapper">
            <button class="citations-toggle grounding-toggle">
                <span class="arrow">▶</span>
                ${statusIcon} Grounding: ${statusText} (${overall}%)
            </button>
            <div class="citations-list" style="padding:12px;">
                ${makeBar('Faithfulness', grounding.faithfulness, 'var(--green)')}
                ${makeBar('Relevance', grounding.relevance, 'var(--blue)')}
                ${makeBar('Coverage', grounding.coverage, 'var(--gold)')}
                ${ungroundedHtml}
            </div>
        </div>
    `;
}

function buildExpandedQueries(queries) {
    if (!queries || queries.length <= 1) return '';

    const items = queries.map((q, i) => {
        const label = i === 0 ? '🔍 Original' : `🔄 Expansion ${i}`;
        return `<div style="font-size:12px;color:var(--text-secondary);padding:4px 0;border-bottom:1px solid var(--border-color);">
            <span style="color:var(--gold);font-weight:500">${label}:</span> ${escapeHtml(q)}
        </div>`;
    }).join('');

    return `
        <div class="citations-wrapper">
            <button class="citations-toggle queries-toggle">
                <span class="arrow">▶</span>
                🔀 ${queries.length} Search Queries Used
            </button>
            <div class="citations-list" style="padding:10px 14px;">
                ${items}
            </div>
        </div>
    `;
}

function buildCitations(citations) {
    if (!citations || citations.length === 0) return '';

    const cards = citations.map((c, i) => {
        // Build reference label
        const refParts = [];
        if (c.article_number) refParts.push(c.article_number);
        if (c.section) refParts.push(c.section);
        if (c.act_name) refParts.push(c.act_name);
        if (c.part) refParts.push(c.part);
        const refLabel = refParts.length > 0 ? refParts.join(' • ') : `Source ${i + 1}`;

        // Scores
        const simScore = (c.similarity_score * 100).toFixed(1);
        const rerankHtml = c.rerank_score != null 
            ? `<span class="score-pill">Rerank: ${(c.rerank_score * 100).toFixed(1)}%</span>` 
            : '';

        return `
            <div class="citation-card">
                <div class="citation-header">
                    <span class="citation-ref">📌 ${escapeHtml(refLabel)}</span>
                    <div class="citation-scores">
                        <span class="score-pill">Sim: ${simScore}%</span>
                        ${rerankHtml}
                    </div>
                </div>
                <div class="citation-text">${escapeHtml(c.text)}</div>
            </div>
        `;
    }).join('');

    return `
        <div class="citations-wrapper">
            <button class="citations-toggle">
                <span class="arrow">▶</span>
                📜 View ${citations.length} Source Citation${citations.length > 1 ? 's' : ''}
            </button>
            <div class="citations-list">
                ${cards}
            </div>
        </div>
    `;
}

// ==================== UTILITIES ====================

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatMarkdown(text) {
    if (!text) return '';
    
    let html = escapeHtml(text);
    
    // Bold: **text** or __text__
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');
    
    // IRAC Highlights
    html = html.replace(/<strong>(ISSUE|RULE|APPLICATION|CONCLUSION):?<\/strong>/g, '<strong class="irac-highlight">$1:</strong>');
    
    
    // Italic: *text* or _text_  (Only match when separated by space or boundary to avoid clobbering bold leftovers or footnotes)
    html = html.replace(/\b_([^_]+)_\b/g, '<em>$1</em>');
    html = html.replace(/(^|\s)\*([^*]+)\*(?=\s|[.,;:]|$)/g, '$1<em>$2</em>');
    
    // Citations in brackets: [Article 21, Constitution of India]
    html = html.replace(/\[([^\]]+)\]/g, '<strong style="color:var(--gold)">[$1]</strong>');
    
    // Line breaks
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');
    
    // Numbered lists: "1. " at start of line
    html = html.replace(/(?:^|<br>)(\d+)\.\s/g, '<br><strong>$1.</strong> ');
    
    // Sanitize generated HTML using DOMPurify
    if (typeof DOMPurify !== 'undefined') {
        html = DOMPurify.sanitize(`<p>${html}</p>`, { ALLOWED_TAGS: ['p', 'strong', 'em', 'br'] });
    } else {
        html = `<p>${html}</p>`; 
    }
    
    return html;
}
