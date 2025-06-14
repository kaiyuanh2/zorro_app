function updateCounter() {
    const textarea = document.getElementById('dataDesc');
    const counter = document.getElementById('descCounter');
    const maxLength = 300;
    if (textarea && counter) {
        counter.textContent = `${textarea.value.length}/${maxLength} characters`;
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const textarea = document.getElementById('dataDesc');
    if (textarea) {
        textarea.addEventListener('input', updateCounter);
        updateCounter();
    }
});