const API_URL = 'http://localhost:5000/api';
let sessionId = null;
let isProcessing = false;
let backendOnline = false;
let currentPage = 1;
let totalPages = 1;
let rowsPerPage = 50;
let totalRows = 0;

const fileInput = document.getElementById('fileInput');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const dataPreview = document.getElementById('dataPreview');
const columnsList = document.getElementById('columnsList');
const undoBtn = document.getElementById('undoBtn');
const downloadBtn = document.getElementById('downloadBtn');
const typingIndicator = document.getElementById('typingIndicator');
const chartContainer = document.getElementById('chartContainer');
const backendStatus = document.getElementById('backendStatus');
const statusText = document.getElementById('statusText');

// Search and pagination elements
let searchInput, clearSearchBtn, paginationControls;

fileInput.addEventListener('change', handleFileUpload);
sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !isProcessing) sendMessage();
});
undoBtn.addEventListener('click', undoLastAction);
downloadBtn.addEventListener('click', downloadCSV);

async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_URL}/health`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (response.ok) {
            const data = await response.json();
            backendOnline = true;
            backendStatus.className = 'status-indicator status-online';
            statusText.textContent = 'Backend connected • Ready to process';
            console.log('✅ Backend health:', data);
        } else {
            throw new Error('Backend not responding');
        }
    } catch (error) {
        backendOnline = false;
        backendStatus.className = 'status-indicator status-offline';
        statusText.textContent = 'Backend offline • Please start Flask server';
        addMessage('⚠️ <strong>Backend not connected!</strong><br><br>Please make sure the Flask server is running on <code>http://localhost:5000</code><br><br>Run: <code>python app.py</code>', 'assistant');
        console.error('❌ Backend check failed:', error);
    }
}

async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!backendOnline) {
        addMessage('❌ Cannot upload file. Backend server is not running.', 'assistant');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        addMessage(`📤 Uploading <strong>${file.name}</strong>...`, 'assistant');
        
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            sessionId = data.session_id;
            totalRows = data.total_rows || 0;
            
            // Reset pagination
            currentPage = 1;
            totalPages = Math.ceil(totalRows / rowsPerPage);
            
            // Display data preview with pagination
            await loadPageData(1);
            updateColumnsList(data.profile);
            
            // Enable controls
            chatInput.disabled = false;
            sendBtn.disabled = false;
            undoBtn.disabled = false;
            downloadBtn.disabled = false;

            // Show AI analysis
            if (data.ai_analysis) {
                setTimeout(() => {
                    displayAIResponse(data.ai_analysis);
                }, 500);
            }
            
            addMessage(`✅ <strong>File uploaded successfully!</strong><br>Session ID: <code>${sessionId}</code><br>Total rows: <strong>${totalRows.toLocaleString()}</strong>`, 'assistant');
        } else {
            addMessage(`❌ <strong>Upload failed:</strong> ${data.error}`, 'assistant');
        }
    } catch (error) {
        addMessage(`❌ <strong>Upload error:</strong> ${error.message}`, 'assistant');
    }
}

async function loadPageData(page, searchQuery = '') {
    if (!sessionId) return;
    
    try {
        const response = await fetch(`${API_URL}/get_page`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                page: page,
                rows_per_page: rowsPerPage,
                search_query: searchQuery
            })
        });

        const data = await response.json();

        if (data.success) {
            currentPage = page;
            totalPages = data.total_pages;
            totalRows = data.total_rows;
            displayData(data.preview, searchQuery);
            updatePaginationControls();
        } else {
            addMessage(`❌ <strong>Error loading data:</strong> ${data.error}`, 'assistant');
        }
    } catch (error) {
        console.error('Error loading page:', error);
        addMessage(`❌ <strong>Error loading data:</strong> ${error.message}`, 'assistant');
    }
}

function createSearchAndPaginationUI() {
    // Create search box
    const searchContainer = document.createElement('div');
    searchContainer.style.cssText = 'margin-bottom: 15px; display: flex; gap: 10px; align-items: center;';
    
    searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.placeholder = '🔍 Search data (e.g., column:value or keyword)...';
    searchInput.style.cssText = 'flex: 1; padding: 10px 15px; border: 2px solid rgba(102, 126, 234, 0.3); border-radius: 8px; font-size: 14px; background: rgba(17, 24, 39, 0.5); color: #e5e7eb;';
    
    clearSearchBtn = document.createElement('button');
    clearSearchBtn.textContent = '✕ Clear';
    clearSearchBtn.style.cssText = 'padding: 10px 20px; background: rgba(239, 68, 68, 0.2); border: 2px solid rgba(239, 68, 68, 0.5); border-radius: 8px; color: #fca5a5; cursor: pointer; font-size: 14px;';
    clearSearchBtn.style.display = 'none';
    
    // Search event handlers
    let searchTimeout;
    searchInput.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        const query = e.target.value.trim();
        
        if (query) {
            clearSearchBtn.style.display = 'block';
            searchTimeout = setTimeout(() => {
                performSearch(query);
            }, 500); // Debounce 500ms
        } else {
            clearSearchBtn.style.display = 'none';
            loadPageData(1);
        }
    });
    
    clearSearchBtn.addEventListener('click', () => {
        searchInput.value = '';
        clearSearchBtn.style.display = 'none';
        loadPageData(1);
    });
    
    searchContainer.appendChild(searchInput);
    searchContainer.appendChild(clearSearchBtn);
    
    // Create pagination controls
    paginationControls = document.createElement('div');
    paginationControls.style.cssText = 'margin-top: 15px; display: flex; justify-content: space-between; align-items: center; padding: 10px; background: rgba(17, 24, 39, 0.5); border-radius: 8px;';
    
    // Insert before data preview
    dataPreview.parentNode.insertBefore(searchContainer, dataPreview);
    dataPreview.parentNode.insertBefore(paginationControls, dataPreview.nextSibling);
}

async function performSearch(query) {
    if (!sessionId) return;
    
    try {
        const response = await fetch(`${API_URL}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                query: query,
                page: 1,
                rows_per_page: rowsPerPage
            })
        });

        const data = await response.json();

        if (data.success) {
            currentPage = 1;
            totalPages = data.total_pages;
            totalRows = data.total_results;
            displayData(data.results, query);
            updatePaginationControls();
            
            if (totalRows === 0) {
                addMessage(`🔍 No results found for: <strong>${query}</strong>`, 'assistant');
            } else {
                addMessage(`🔍 Found <strong>${totalRows.toLocaleString()}</strong> results for: <strong>${query}</strong>`, 'assistant');
            }
        } else {
            addMessage(`❌ <strong>Search failed:</strong> ${data.error}`, 'assistant');
        }
    } catch (error) {
        console.error('Search error:', error);
        addMessage(`❌ <strong>Search error:</strong> ${error.message}`, 'assistant');
    }
}

function updatePaginationControls() {
    if (!paginationControls) return;
    
    const isSearching = searchInput && searchInput.value.trim() !== '';
    
    paginationControls.innerHTML = `
        <div style="display: flex; gap: 10px; align-items: center;">
            <button id="firstPageBtn" style="padding: 8px 12px; background: rgba(102, 126, 234, 0.2); border: 2px solid rgba(102, 126, 234, 0.5); border-radius: 6px; color: #a5b4fc; cursor: pointer; font-size: 14px;" ${currentPage === 1 ? 'disabled' : ''}>⏮ First</button>
            <button id="prevPageBtn" style="padding: 8px 12px; background: rgba(102, 126, 234, 0.2); border: 2px solid rgba(102, 126, 234, 0.5); border-radius: 6px; color: #a5b4fc; cursor: pointer; font-size: 14px;" ${currentPage === 1 ? 'disabled' : ''}>← Prev</button>
            <span style="color: #9ca3af; font-size: 14px;">
                Page <strong style="color: #e5e7eb;">${currentPage}</strong> of <strong style="color: #e5e7eb;">${totalPages}</strong>
                ${isSearching ? '(filtered)' : ''}
            </span>
            <button id="nextPageBtn" style="padding: 8px 12px; background: rgba(102, 126, 234, 0.2); border: 2px solid rgba(102, 126, 234, 0.5); border-radius: 6px; color: #a5b4fc; cursor: pointer; font-size: 14px;" ${currentPage >= totalPages ? 'disabled' : ''}>Next →</button>
            <button id="lastPageBtn" style="padding: 8px 12px; background: rgba(102, 126, 234, 0.2); border: 2px solid rgba(102, 126, 234, 0.5); border-radius: 6px; color: #a5b4fc; cursor: pointer; font-size: 14px;" ${currentPage >= totalPages ? 'disabled' : ''}>Last ⏭</button>
        </div>
        <div style="display: flex; gap: 10px; align-items: center;">
            <span style="color: #9ca3af; font-size: 13px;">
                Showing ${((currentPage - 1) * rowsPerPage) + 1}-${Math.min(currentPage * rowsPerPage, totalRows)} of ${totalRows.toLocaleString()} rows
            </span>
            <select id="rowsPerPageSelect" style="padding: 6px 10px; background: rgba(17, 24, 39, 0.8); border: 2px solid rgba(102, 126, 234, 0.3); border-radius: 6px; color: #e5e7eb; font-size: 13px; cursor: pointer;">
                <option value="25" ${rowsPerPage === 25 ? 'selected' : ''}>25 rows</option>
                <option value="50" ${rowsPerPage === 50 ? 'selected' : ''}>50 rows</option>
                <option value="100" ${rowsPerPage === 100 ? 'selected' : ''}>100 rows</option>
                <option value="200" ${rowsPerPage === 200 ? 'selected' : ''}>200 rows</option>
            </select>
        </div>
    `;
    
    // Add event listeners
    document.getElementById('firstPageBtn')?.addEventListener('click', () => navigateToPage(1));
    document.getElementById('prevPageBtn')?.addEventListener('click', () => navigateToPage(currentPage - 1));
    document.getElementById('nextPageBtn')?.addEventListener('click', () => navigateToPage(currentPage + 1));
    document.getElementById('lastPageBtn')?.addEventListener('click', () => navigateToPage(totalPages));
    document.getElementById('rowsPerPageSelect')?.addEventListener('change', (e) => {
        rowsPerPage = parseInt(e.target.value);
        totalPages = Math.ceil(totalRows / rowsPerPage);
        navigateToPage(1);
    });
    
    // Style disabled buttons
    const buttons = paginationControls.querySelectorAll('button[disabled]');
    buttons.forEach(btn => {
        btn.style.opacity = '0.5';
        btn.style.cursor = 'not-allowed';
    });
}

function navigateToPage(page) {
    if (page < 1 || page > totalPages) return;
    
    const searchQuery = searchInput?.value.trim() || '';
    if (searchQuery) {
        performSearch(searchQuery);
    } else {
        loadPageData(page);
    }
}

function displayData(data, highlightQuery = '') {
    if (!data || data.length === 0) {
        dataPreview.innerHTML = '<div class="info-banner">No data to display</div>';
        return;
    }

    const headers = Object.keys(data[0]);
    let html = '<div style="overflow-x: auto;"><table style="width: 100%; border-collapse: collapse;"><thead><tr>';
    
    // Add row number column
    html += '<th style="padding: 12px; background: rgba(102, 126, 234, 0.1); border-bottom: 2px solid rgba(102, 126, 234, 0.3); text-align: left; color: #a5b4fc; font-weight: 600;">#</th>';
    
    headers.forEach(h => {
        const highlighted = highlightQuery && h.toLowerCase().includes(highlightQuery.toLowerCase()) 
            ? `background: rgba(251, 191, 36, 0.2);` 
            : '';
        html += `<th style="padding: 12px; background: rgba(102, 126, 234, 0.1); border-bottom: 2px solid rgba(102, 126, 234, 0.3); text-align: left; color: #a5b4fc; font-weight: 600; ${highlighted}">${h}</th>`;
    });
    html += '</tr></thead><tbody>';

    data.forEach((row, idx) => {
        const rowNum = ((currentPage - 1) * rowsPerPage) + idx + 1;
        html += '<tr style="border-bottom: 1px solid rgba(75, 85, 99, 0.3);">';
        html += `<td style="padding: 10px; color: #6b7280; font-size: 12px;">${rowNum}</td>`;
        
        headers.forEach(h => {
            let value = row[h] !== null && row[h] !== undefined ? String(row[h]) : '';
            
            // Highlight search matches
            if (highlightQuery && value.toLowerCase().includes(highlightQuery.toLowerCase())) {
                const regex = new RegExp(`(${highlightQuery})`, 'gi');
                value = value.replace(regex, '<mark style="background: rgba(251, 191, 36, 0.3); color: #fbbf24; padding: 2px 4px; border-radius: 3px;">$1</mark>');
            }
            
            html += `<td style="padding: 10px; color: #e5e7eb; font-size: 14px;">${value}</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table></div>';
    dataPreview.innerHTML = html;
}

function updateColumnsList(profile) {
    if (!profile || !profile.columns) {
        columnsList.innerHTML = '<div class="info-banner">No columns available</div>';
        return;
    }

    let html = '';
    profile.columns.forEach(col => {
        const dtype = profile.dtypes[col] || 'unknown';
        const missing = profile.missing_values[col] || 0;
        
        html += `
            <div class="column-chip">
                <span title="${missing} missing values">${col}</span>
                <span class="column-type">${dtype.replace('int64', 'int').replace('float64', 'float').replace('object', 'str')}</span>
            </div>
        `;
    });
    
    columnsList.innerHTML = html;
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || !sessionId || isProcessing) return;

    if (!backendOnline) {
        addMessage('❌ Cannot send message. Backend server is not running.', 'assistant');
        return;
    }

    isProcessing = true;
    addMessage(message, 'user');
    chatInput.value = '';
    chatInput.disabled = true;
    sendBtn.disabled = true;

    typingIndicator.style.display = 'block';
    scrollToBottom();

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                message: message
            })
        });

        const data = await response.json();
        typingIndicator.style.display = 'none';

        if (data.type) {
            displayAIResponse(data);
            
            // Refresh data view if data was modified
            if (data.type === 'code' && data.success) {
                await loadPageData(currentPage);
                if (data.profile) updateColumnsList(data.profile);
            }
        } else if (data.error) {
            addMessage(`❌ <strong>Error:</strong> ${data.error}`, 'assistant');
        }
    } catch (error) {
        typingIndicator.style.display = 'none';
        addMessage(`❌ <strong>Request failed:</strong> ${error.message}`, 'assistant');
    } finally {
        isProcessing = false;
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();
    }
}

function displayAIResponse(data) {
    const type = data.type;
    const explanation = data.explanation || '';
    const content = data.content || '';

    let message = '';

    switch (type) {
        case 'question':
            message = `❓ <strong>Clarification Needed:</strong><br><br>${content}`;
            addMessage(message, 'assistant');
            break;

        case 'code':
            if (data.success) {
                message = `✅ <strong>Data Modified Successfully!</strong><br><br>`;
                if (explanation) message += `${explanation}<br><br>`;
                message += `<strong>Code executed:</strong><br><code>${content.substring(0, 150)}${content.length > 150 ? '...' : ''}</code>`;
            } else {
                message = `❌ <strong>Operation Failed</strong><br><br>${data.error}`;
            }
            addMessage(message, 'assistant');
            break;

        case 'find':
            if (data.success) {
                message = `🔍 <strong>Search Results:</strong><br><br>`;
                if (explanation) message += `${explanation}<br><br>`;
                
                if (Array.isArray(data.result)) {
                    message += `<span class="stat-badge">Found ${data.result.length} results</span><br>`;
                    const preview = data.result.slice(0, 10);
                    message += '<pre>' + JSON.stringify(preview, null, 2) + '</pre>';
                    if (data.result.length > 10) {
                        message += `<em>Showing first 10 of ${data.result.length} results</em>`;
                    }
                } else if (typeof data.result === 'object') {
                    message += '<pre>' + JSON.stringify(data.result, null, 2) + '</pre>';
                } else {
                    message += `<strong>Result:</strong> ${data.result}`;
                }
            } else {
                message = `❌ <strong>Search Failed:</strong> ${data.error}`;
            }
            addMessage(message, 'assistant');
            break;

        case 'plot':
            if (data.success && data.plot) {
                message = `📊 <strong>Visualization Created!</strong><br><br>`;
                if (explanation) message += explanation;
                addMessage(message, 'assistant');
                
                chartContainer.style.display = 'block';
                const plotData = JSON.parse(data.plot);
                Plotly.newPlot(chartContainer, plotData.data, plotData.layout, {responsive: true});
                
                setTimeout(() => {
                    chartContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }, 300);
            } else {
                message = `❌ <strong>Visualization Failed:</strong> ${data.error}`;
                addMessage(message, 'assistant');
            }
            break;

        case 'analysis':
            if (data.success) {
                message = `📈 <strong>Analysis Complete!</strong><br><br>`;
                if (explanation) message += `${explanation}<br><br>`;
                
                if (typeof data.result === 'object') {
                    message += '<pre>' + JSON.stringify(data.result, null, 2) + '</pre>';
                } else {
                    message += `<strong>Result:</strong> ${data.result}`;
                }
            } else {
                message = `❌ <strong>Analysis Failed:</strong> ${data.error}`;
            }
            addMessage(message, 'assistant');
            break;

        case 'message':
        default:
            addMessage(content, 'assistant');
            break;
    }
}

function addMessage(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    messageDiv.innerHTML = content;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

async function undoLastAction() {
    if (!sessionId || isProcessing || !backendOnline) return;

    try {
        const response = await fetch(`${API_URL}/undo`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });

        const data = await response.json();

        if (data.success) {
            await loadPageData(currentPage);
            updateColumnsList(data.profile);
            addMessage('↩ <strong>Action undone successfully!</strong>', 'assistant');
        } else {
            addMessage(`❌ ${data.error}`, 'assistant');
        }
    } catch (error) {
        addMessage(`❌ <strong>Undo failed:</strong> ${error.message}`, 'assistant');
    }
}

async function downloadCSV() {
    if (!sessionId || !backendOnline) return;

    try {
        const response = await fetch(`${API_URL}/download`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'cleaned_data.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            addMessage('✅ <strong>Downloaded cleaned data successfully!</strong>', 'assistant');
        } else {
            const errorData = await response.json();
            addMessage(`❌ <strong>Download failed:</strong> ${errorData.error}`, 'assistant');
        }
    } catch (error) {
        addMessage(`❌ <strong>Download failed:</strong> ${error.message}`, 'assistant');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Cleansera Frontend Initialized');
    console.log('📡 Backend URL:', API_URL);
    
    // Create search and pagination UI
    createSearchAndPaginationUI();
    
    checkBackendHealth();
    setInterval(checkBackendHealth, 30000);

    setTimeout(() => {
        if (!backendOnline) {
            addMessage(`
                🛠️ <strong>Quick Setup Guide:</strong><br><br>
                1. Install dependencies: <code>pip install -r requirements.txt</code><br>
                2. Set API key: <code>export GEMINI_API_KEY="your_key"</code><br>
                3. Start server: <code>python app.py</code><br><br>
                Once the backend is running, refresh this page!
            `, 'assistant');
        }
    }, 2000);
});

const uploadZone = document.querySelector('.upload-zone');

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'rgba(102, 126, 234, 1)';
    uploadZone.style.background = 'rgba(102, 126, 234, 0.1)';
});

uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'rgba(102, 126, 234, 0.3)';
    uploadZone.style.background = 'transparent';
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'rgba(102, 126, 234, 0.3)';
    uploadZone.style.background = 'transparent';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileUpload({ target: { files: files } });
    }
});

document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Z for undo
    if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !undoBtn.disabled) {
        e.preventDefault();
        undoLastAction();
    }
    
    // Ctrl/Cmd + S for download
    if ((e.ctrlKey || e.metaKey) && e.key === 's' && !downloadBtn.disabled) {
        e.preventDefault();
        downloadCSV();
    }
    
    // Ctrl/Cmd + F for search
    if ((e.ctrlKey || e.metaKey) && e.key === 'f' && searchInput) {
        e.preventDefault();
        searchInput.focus();
    }
    
    // Arrow keys for pagination
    if (e.key === 'ArrowLeft' && currentPage > 1 && !e.target.matches('input')) {
        navigateToPage(currentPage - 1);
    }
    if (e.key === 'ArrowRight' && currentPage < totalPages && !e.target.matches('input')) {
        navigateToPage(currentPage + 1);
    }
});

window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
});
