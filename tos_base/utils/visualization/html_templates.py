# html_templates.py
# HTML templates, CSS styles, and JavaScript code for visualization

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SpatialGym Dashboard</title>
    <style>
{css_styles}
    </style>
</head>
<body>
    <div id='nav'>
        <button onclick="prevPage({total_pages})">Prev</button>
        <button onclick="nextPage({total_pages})">Next</button>
        <input id="goto" type="number" min="1" max="{total_pages}" placeholder="page" onkeydown="if(event.key==='Enter') gotoPage({total_pages});">
        <button onclick="gotoPage({total_pages})">Go</button>
        <span id='counter'></span>
    </div>
    
    <h1>Model: {model_name}</h1>
    
    <script>
{javascript_code}
    </script>
"""

CSS_STYLES = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

#nav {
    position: fixed;
    top: 0;
    width: 100%;
    background: rgba(51, 51, 51, 0.95);
    backdrop-filter: blur(10px);
    color: #fff;
    padding: 15px 0;
    text-align: center;
    z-index: 999;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

#nav button {
    margin: 0 8px;
    padding: 8px 16px;
    border: none;
    background: #4CAF50;
    color: #fff;
    cursor: pointer;
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.3s ease;
}

#nav button:hover {
    background: #45a049;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

#nav input {
    width: 80px;
    padding: 8px;
    border-radius: 6px;
    border: 1px solid #ddd;
    margin-left: 12px;
    text-align: center;
    font-size: 14px;
}

#counter {
    margin-left: 12px;
    font-size: 0.9em;
    opacity: 0.9;
    font-weight: 500;
}

.sample-page {
    display: none;
    padding: 100px 24px 24px 24px;
    max-width: 1200px;
    margin: auto;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    margin-top: 20px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
}

.sample-page.active {
    display: block;
    animation: fadeIn 0.5s ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Summary sections */
.summary-section {
    margin: 30px 0;
    padding: 20px;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 12px;
    border: 1px solid #dee2e6;
}

.summary-card {
    background: white;
    border-radius: 8px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border-left: 4px solid;
}

.summary-card.exploration {
    border-left-color: #17a2b8;
}

.summary-card.evaluation {
    border-left-color: #28a745;
}

.summary-card h4 {
    margin: 0 0 15px 0;
    color: #495057;
    font-size: 1.2em;
    font-weight: 600;
}

.config-summaries {
    margin: 30px 0;
}

.config-summary {
    background: white;
    border-radius: 8px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border: 1px solid #dee2e6;
}

.config-summary h4 {
    margin: 0 0 15px 0;
    color: #495057;
    font-size: 1.1em;
    font-weight: 600;
}

.config-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    margin-bottom: 15px;
}

.stat-item {
    background: #e9ecef;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.9em;
    color: #495057;
    font-weight: 500;
}

.group-metrics {
    margin: 10px 0;
    padding: 10px;
    background: #f8f9fa;
    border-radius: 6px;
    border-left: 3px solid #6c757d;
}

.group-metrics strong {
    color: #495057;
    font-weight: 600;
    display: block;
    margin-bottom: 8px;
}

/* Improved dictionary styling */
.dict-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 8px;
    margin-top: 10px;
}

.dict-item {
    background: #f8f9fa;
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid #dee2e6;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.9em;
}

.dict-key {
    font-weight: 600;
    color: #495057;
    margin-right: 8px;
}

.dict-value {
    color: #6c757d;
    font-weight: 500;
}

.dict-value.number {
    color: #007bff;
    font-weight: 600;
}

.dict-value.true {
    color: #28a745;
    font-weight: 600;
}

.dict-value.false {
    color: #dc3545;
    font-weight: 600;
}

.dict-value.string {
    color: #6f42c1;
    font-weight: 600;
}

.dict-item.nested {
    flex-direction: column;
    align-items: flex-start;
}

.dict-value.nested-dict {
    width: 100%;
    margin-top: 8px;
    padding: 8px;
    background: #ffffff;
    border: 1px solid #e9ecef;
    border-radius: 4px;
}

.empty-dict {
    color: #6c757d;
    font-style: italic;
    padding: 8px 12px;
    background: #f8f9fa;
    border-radius: 6px;
    border: 1px dashed #dee2e6;
}

.turn {
    background: #fff;
    border: 1px solid #e1e5e9;
    border-radius: 12px;
    margin: 20px 0;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
}

.turn:hover {
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}

.turn h3 {
    margin: 0 0 15px 0;
    font-size: 18px;
    color: #2c3e50;
    font-weight: 600;
    border-bottom: 2px solid #3498db;
    padding-bottom: 8px;
}

.turn-split {
    background: #fff;
    border: 1px solid #e1e5e9;
    border-radius: 12px;
    margin: 30px 0;
    padding: 30px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
    display: flex;
    flex-direction: column;
    min-height: 400px;
}

.turn-split:hover {
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}

.turn-split h3 {
    margin: 0 0 25px 0;
    font-size: 18px;
    color: #2c3e50;
    font-weight: 600;
    border-bottom: 2px solid #3498db;
    padding-bottom: 8px;
}

.turn-left {
    flex: 1;
    margin-right: 20px;
}

.turn-right {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 15px;
    padding: 10px;
}

.room-plot {
    max-width: 100%;
    max-height: 400px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.message-image {
    max-width: 100%;
    max-height: 300px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}

.message-image:hover {
    transform: scale(1.02);
}

@media (min-width: 768px) {
    .turn-split {
        flex-direction: row;
    }
}

.block {
    padding: 15px;
    border-radius: 8px;
    margin: 12px 0;
    font-size: 14px;
    line-height: 1.6;
    border-left: 4px solid;
    position: relative;
    overflow: hidden;
}

.block::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    opacity: 0.05;
    background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.3) 50%, transparent 70%);
    transform: translateX(-100%);
    transition: transform 0.6s ease;
}

.block:hover::before {
    transform: translateX(100%);
}

.block.user {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left-color: #2196F3;
    color: #1565c0;
}

.block.user.expandable {
    cursor: pointer;
    transition: all 0.2s ease;
}

.block.user.expandable:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(33, 150, 243, 0.2);
}

.expand-hint {
    font-size: 0.8em;
    color: #1976d2;
    font-weight: normal;
}

.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.7);
    z-index: 1000;
    display: none;
    justify-content: center;
    align-items: center;
}

.modal-content {
    background: white;
    border-radius: 12px;
    padding: 30px;
    max-width: 80%;
    max-height: 80%;
    overflow-y: auto;
    position: relative;
    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
}

.modal-close {
    position: absolute;
    top: 15px;
    right: 20px;
    font-size: 24px;
    cursor: pointer;
    color: #666;
    font-weight: bold;
}

.modal-close:hover {
    color: #333;
}

.block.think {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    border-left-color: #ff9800;
    color: #e65100;
    font-style: italic;
}

.block.think.expandable {
    cursor: pointer;
    transition: all 0.2s ease;
}

.block.think.expandable:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(255, 152, 0, 0.2);
}

.block.answer {
    background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
    border-left-color: #4caf50;
    color: #2e7d32;
}

.block.evaluation {
    background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
    border-left-color: #9c27b0;
    color: #6a1b9a;
}

.block.cogmap {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left-color: #2196f3;
    color: #0d47a1;
}

.metrics {
    margin-top: 15px;
    font-size: 13px;
    color: #555;
    background: #f8f9fa;
    padding: 12px;
    border-radius: 6px;
    border-left: 3px solid #6c757d;
}

.metrics div {
    margin: 4px 0;
    padding: 2px 0;
}

.metrics strong {
    color: #495057;
    font-weight: 600;
}

img.room {
    max-width: 300px;
    height: auto;
    border: 2px solid #e1e5e9;
    border-radius: 8px;
    margin: 15px 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}

img.room:hover {
    transform: scale(1.02);
}

h1 {
    margin-top: 60px;
    color: #fff;
    text-align: center;
    font-size: 2.5em;
    font-weight: 300;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    margin-bottom: 30px;
}

h2 {
    color: #2c3e50;
    margin-top: 30px;
    font-size: 1.8em;
    font-weight: 500;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
}

h3 {
    color: #34495e;
    margin-top: 25px;
    font-size: 1.4em;
    font-weight: 500;
    border-bottom: 1px solid #bdc3c7;
    padding-bottom: 8px;
}

ul {
    list-style: none;
    padding-left: 0;
}

ul ul {
    padding-left: 20px;
}

li {
    margin: 8px 0;
    padding: 8px 12px;
    border-radius: 6px;
    transition: all 0.3s ease;
}

li:hover {
    background: rgba(52, 152, 219, 0.1);
}

a {
    color: #3498db;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.3s ease;
}

a:hover {
    color: #2980b9;
    text-decoration: underline;
}

/* Summary sections */
.summary-section {
    margin: 30px 0;
    padding: 20px;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 12px;
    border: 1px solid #dee2e6;
}

.summary-card {
    background: white;
    border-radius: 8px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border-left: 4px solid;
}

.summary-card.exploration {
    border-left-color: #17a2b8;
}

.summary-card.evaluation {
    border-left-color: #28a745;
}

.summary-card h4 {
    margin: 0 0 15px 0;
    color: #495057;
    font-size: 1.2em;
    font-weight: 600;
}

.config-summaries {
    margin: 30px 0;
}

.config-summary {
    background: white;
    border-radius: 8px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border: 1px solid #dee2e6;
}

.config-summary h4 {
    margin: 0 0 15px 0;
    color: #495057;
    font-size: 1.1em;
    font-weight: 600;
}

.config-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    margin-bottom: 15px;
}

.stat-item {
    background: #e9ecef;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 0.9em;
    color: #495057;
    font-weight: 500;
}

.group-metrics {
    margin: 10px 0;
    padding: 10px;
    background: #f8f9fa;
    border-radius: 6px;
    border-left: 3px solid #6c757d;
}

.group-metrics strong {
    color: #495057;
    font-weight: 600;
    display: block;
    margin-bottom: 8px;
}

/* Improved dictionary styling */
.dict-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 8px;
    margin-top: 10px;
}

.dict-item {
    background: #f8f9fa;
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid #dee2e6;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.9em;
}

.dict-key {
    font-weight: 600;
    color: #495057;
    margin-right: 8px;
}

.dict-value {
    color: #6c757d;
    font-weight: 500;
}

.dict-value.number {
    color: #007bff;
    font-weight: 600;
}

.dict-value.true {
    color: #28a745;
    font-weight: 600;
}

.dict-value.false {
    color: #dc3545;
    font-weight: 600;
}

.empty-dict {
    color: #6c757d;
    font-style: italic;
    padding: 8px 12px;
    background: #f8f9fa;
    border-radius: 6px;
    border: 1px dashed #dee2e6;
}

/* Responsive design */
@media (max-width: 768px) {
    .sample-page {
        padding: 80px 16px 16px 16px;
        margin: 10px;
    }
    
    h1 {
        font-size: 2em;
        margin-top: 50px;
    }
    
    .turn {
        padding: 15px;
    }
    
    img.room {
        max-width: 100%;
    }
    
    #nav input {
        width: 60px;
    }
    
    .dict-container {
        grid-template-columns: 1fr;
    }
    
    .config-stats {
        flex-direction: column;
    }
}
"""

JAVASCRIPT_CODE = """
let currentPage = 0;

function showPage(n, total) {
    currentPage = Math.max(0, Math.min(total-1, n));
    const pages = document.querySelectorAll('.sample-page');
    pages.forEach((p, i) => {
        p.classList.toggle('active', i === currentPage);
    });
    document.getElementById('counter').innerText = (currentPage+1) + ' / ' + total;
    document.getElementById('goto').value = currentPage+1;
    location.hash = '#p' + (currentPage+1);
    
    // Smooth scroll to top
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

function nextPage(total) {
    showPage(currentPage+1, total);
}

function prevPage(total) {
    showPage(currentPage-1, total);
}

function gotoPage(total) {
    const v = parseInt(document.getElementById('goto').value, 10);
    if (!isNaN(v)) {
        showPage(v-1, total);
    }
}

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === 'PageDown') {
        nextPage({total_pages});
    }
    if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
        prevPage({total_pages});
    }
    if (e.key === 'Home') {
        showPage(0, {total_pages});
    }
    if (e.key === 'End') {
        showPage({total_pages}-1, {total_pages});
    }
});

// Initialize on page load
window.addEventListener('load', () => {
    const m = location.hash.match(/#p(\\d+)/);
    if (m) {
        showPage(parseInt(m[1], 10)-1, {total_pages});
    } else {
        showPage(0, {total_pages});
    }
});

// Add smooth transitions
document.addEventListener('DOMContentLoaded', () => {
    const style = document.createElement('style');
    style.textContent = `
        .sample-page {
            transition: opacity 0.3s ease, transform 0.3s ease;
        }
    `;
    document.head.appendChild(style);
});

// Expand observation functionality
function expandObservation(obsId) {
    const fullContent = document.getElementById(obsId).innerHTML;
    
    // Create modal if it doesn't exist
    let modal = document.getElementById('obs-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'obs-modal';
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <span class="modal-close" onclick="closeObservation()">&times;</span>
                <h3>🔍 Full Environment Observation</h3>
                <div id="modal-obs-content"></div>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Close on background click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeObservation();
        });
        
        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeObservation();
        });
    }
    
    // Set content and show modal
    document.getElementById('modal-obs-content').innerHTML = fullContent;
    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function closeObservation() {
    const modal = document.getElementById('obs-modal');
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto';
    }
}

// Expand thinking functionality
function expandThinking(thinkId) {
    const fullContent = document.getElementById(thinkId).innerHTML;
    
    // Create modal if it doesn't exist
    let modal = document.getElementById('think-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'think-modal';
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <span class="modal-close" onclick="closeThinking()">&times;</span>
                <h3>🤔 Full Assistant Thinking</h3>
                <div id="modal-think-content"></div>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Close on background click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeThinking();
        });
    }
    
    // Set content and show modal
    document.getElementById('modal-think-content').innerHTML = fullContent;
    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function closeThinking() {
    const modal = document.getElementById('think-modal');
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto';
    }
}
"""