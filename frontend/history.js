

// Call this function after each completed session to record a new day
function recordDayUsed() {
    let days = JSON.parse(localStorage.getItem('history_days') || '[]');
    const today = new Date().toISOString().slice(0, 10);
    if (!days.includes(today)) {
        days.push(today);
        if (days.length > 5) days = days.slice(-5); // Keep only last 5 days
        localStorage.setItem('history_days', JSON.stringify(days));
    }
}

// Call this function to get the list of days used (max 5)
function getHistoryDays() {
    let days = JSON.parse(localStorage.getItem('history_days') || '[]');
    return days;
}

// Call this after a session is completed to save all details for the day
function saveDayDetails(details) {
    if (!details.timestamp) {
        details.timestamp = new Date().toISOString().slice(0, 10);
    }
    const key = 'history_details_' + details.timestamp;
    let sessions = JSON.parse(localStorage.getItem(key) || '[]');
    // Fix: If sessions is not an array (old data), convert it to an array
    if (!Array.isArray(sessions)) {
        sessions = sessions && typeof sessions === 'object' ? [sessions] : [];
    }
    sessions.push(details);
    localStorage.setItem(key, JSON.stringify(sessions));
}








