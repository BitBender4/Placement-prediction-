document.getElementById('predictionForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const payload = Object.fromEntries(formData.entries());

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        console.log("API Response:", result);

        if (result.success) {
            // Update main results
            const placementText = result.placement_prediction ?? "Prediction Failed";
            const salaryText = result.salary_prediction ?? "N/A";
            const suggestions = Array.isArray(result.skill_suggestions) ? result.skill_suggestions : [];

            const predictionResult = document.getElementById('predictionResult');
            if (predictionResult) {
                predictionResult.innerHTML = `
                    <p><strong>Placement Prediction:</strong> ${placementText}</p>
                    <p><strong>Predicted Salary:</strong> ${salaryText.toString().startsWith('₹') ? salaryText : '₹' + salaryText}</p>
                    ${suggestions.length
                        ? `<h5>Suggestions:</h5><ul>${suggestions.map(s => <li>${s}</li>).join('')}</ul>`
                        : ''}
                `;
            }

            // Update detailed stats
            if (result.prediction) {
                const p = result.prediction;
                const probEl = document.getElementById('placement-probability');
                const statusEl = document.getElementById('placement-status');
                const salaryEl = document.getElementById('predicted-salary');
                const confEl = document.getElementById('salary-confidence');

                if (probEl) probEl.textContent = (p.placement_probability * 100).toFixed(1) + '%';
                if (statusEl) statusEl.textContent = p.is_placed ? 'Yes ✅' : 'No ❌';
                if (salaryEl) salaryEl.textContent = '₹' + Number(p.predicted_salary || 0).toLocaleString();
                if (confEl) confEl.textContent = (p.salary_confidence * 100).toFixed(1) + '%';

                const resultsSection = document.getElementById('results');
                if (resultsSection) {
                    resultsSection.style.display = 'block';
                    resultsSection.scrollIntoView({ behavior: 'smooth' });
                    resultsSection.classList.add('fade-in');
                }
            }
        } else {
            showNotification(result.error || 'Error: Prediction failed', 'error');
        }
    } catch (err) {
        console.error("Prediction fetch error:", err);
        showNotification('Error: Could not fetch prediction', 'error');
    }
});


// Social Integrations (unchanged logic)
async function handleSocialIntegrations() {
    const githubUsername = document.getElementById('github_username')?.value;
    const linkedinUrl = document.getElementById('linkedin_url')?.value;

    if (githubUsername) {
        try {
            const response = await fetch('/api/github-integration', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ github_username: githubUsername })
            });
            const result = await response.json();
            if (result.success) showNotification('GitHub profile analyzed successfully!', 'success');
        } catch (error) {
            console.error('GitHub integration error:', error);
        }
    }

    if (linkedinUrl) {
        try {
            const response = await fetch('/api/linkedin-integration', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ linkedin_url: linkedinUrl })
            });
            const result = await response.json();
            if (result.success) showNotification('LinkedIn profile analyzed successfully!', 'success');
        } catch (error) {
            console.error('LinkedIn integration error:', error);
        }
    }
}


// Chatbot fixes: backend returns { success, reply }, not "response"
async function sendMessage() {
    const chatInput = document.getElementById('chatInput');
    const chatMessages = document.getElementById('chatMessages');
    const message = chatInput?.value.trim();
    if (!message) return;

    addMessageToChat(message, 'user');
    if (chatInput) chatInput.value = '';

    const typingIndicator = addTypingIndicator();

    try {
        const response = await fetch('/api/chatbot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        const result = await response.json();
        typingIndicator.remove();
        if (result.success) {
            addMessageToChat(result.reply, 'bot');
        } else {
            addMessageToChat('Sorry, I encountered an error. Please try again.', 'bot');
        }
    } catch (error) {
        typingIndicator.remove();
        console.error('Chatbot error:', error);
        addMessageToChat('Sorry, I\'m having trouble connecting. Please try again later.', 'bot');
    }
}


// Analytics: align with backend { success, data } and { success, importance }
async function loadAnalytics() {
    try {
        const [analyticsResponse, importanceResponse] = await Promise.all([
            fetch('/api/analytics'),
            fetch('/api/feature-importance')
        ]);

        const analyticsData = await analyticsResponse.json();
        const importanceData = await importanceResponse.json();

        if (analyticsData.success && importanceData.success) {
            updateAnalyticsCharts(analyticsData.data, importanceData.importance);
        }
    } catch (error) {
        console.error('Analytics loading error:', error);
    }
}

function updateAnalyticsCharts(analytics, importance) {
    // Feature Importance Chart
    createFeatureImportanceChart(importance);

    // Activity Chart
    const byHour = analytics?.predictions_by_hour || {};
    createActivityChart(byHour);
}

function createFeatureImportanceChart(importance) {
    const ctx = document.getElementById('featureImportanceChart');
    if (!ctx || !importance) return;

    const labels = Object.keys(importance);
    const data = Object.values(importance);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Feature Importance',
                data,
                borderWidth: 1,
                borderRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) { return (value * 100).toFixed(0) + '%'; }
                    }
                },
                x: { ticks: { maxRotation: 45 } }
            }
        }
    });
}

function createActivityChart(activityData) {
    const ctx = document.getElementById('activityChart');
    if (!ctx) return;

    const hours = Array.from({length: 24}, (_, i) => i);
    const data = hours.map(h => Number(activityData[h] || activityData[String(h)] || 0));

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: hours.map(hour => hour + ':00'),
            datasets: [{
                label: 'Predictions per Hour',
                data,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true } }
        }
    });
}

// Navigation + Chat helpers (kept)
function setupScrollNavigation() {
    document.querySelectorAll('a[href^=\"#\"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });
}

function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function addMessageToChat(message, sender) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.innerHTML = (sender === 'bot')
        ? `<i class="fas fa-robot me-2"></i>${message}`
        : `<i class="fas fa-user me-2"></i>${message}`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addTypingIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.innerHTML = `
        <i class="fas fa-robot me-2"></i>
        <span class="typing-dots"><span>.</span><span>.</span><span>.</span></span>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return typingDiv;
}

// Utility
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type === 'error' ? 'danger' : type} notification`;
    notification.innerHTML = `
        <div class="d-flex align-items-center">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'} me-2"></i>
            <span>${message}</span>
            <button type="button" class="btn-close ms-auto" onclick="this.parentElement.parentElement.remove()"></button>
        </div>
    `;
    document.body.appendChild(notification);
    setTimeout(() => notification.remove(), 5000);
}

// CSS for typing indicator (keep existing)
const style = document.createElement('style');
style.textContent = `
    .typing-indicator .typing-dots span { animation: typing 1.4s infinite; opacity: 0; }
    .typing-indicator .typing-dots span:nth-child(1) { animation-delay: 0.2s; }
    .typing-indicator .typing-dots span:nth-child(2) { animation-delay: 0.4s; }
    .typing-indicator .typing-dots span:nth-child(3) { animation-delay: 0.6s; }
    @keyframes typing { 0%, 60%, 100% { opacity: 0; } 30% { opacity: 1; } }
`;
document.head.appendChild(style);

// Expose functions
window.sendMessage = sendMessage;
window.scrollToSection = scrollToSection;
window.loadAnalytics = loadAnalytics;
window.handleSocialIntegrations = handleSocialIntegrations;