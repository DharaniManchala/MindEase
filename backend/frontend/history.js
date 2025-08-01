// Call this function after each completed session to record a new day for the user
function recordDayUsed() {
    const email = localStorage.getItem('userEmail');
    if (!email) return;
    let days = JSON.parse(localStorage.getItem('history_days_' + email) || '[]');
    const today = new Date().toISOString().slice(0, 10);
    if (!days.includes(today)) {
        days.push(today);
        if (days.length > 5) days = days.slice(-5); // Keep only last 5 days
        localStorage.setItem('history_days_' + email, JSON.stringify(days));
    }
}

// Call this function to get the list of days used (max 5) for the current user
function getHistoryDays() {
    const email = localStorage.getItem('userEmail');
    if (!email) return [];
    let days = JSON.parse(localStorage.getItem('history_days_' + email) || '[]');
    return days;
}

// Call this after a session is completed to save all details for the day for the user
function saveDayDetails(details) {
    const email = localStorage.getItem('userEmail');
    if (!email) return;
    if (!details.timestamp) {
        details.timestamp = new Date().toISOString().slice(0, 10);
    }
    details.email = email; // Attach email to each session
    const key = 'history_details_' + details.timestamp + '_' + email;
    let sessions = JSON.parse(localStorage.getItem(key) || '[]');
    if (!Array.isArray(sessions)) {
        sessions = sessions && typeof sessions === 'object' ? [sessions] : [];
    }
    sessions.push(details);
    localStorage.setItem(key, JSON.stringify(sessions));
}