const datasetDropdown = document.getElementById('datasetSelect');
const datasetDropdownMenu = document.getElementById('item');
const testDropdown = document.getElementById('testSelect');

function showTestDropdown() {
    if (datasetDropdownMenu.value) {
        testDropdown.style.display = 'block';
                datasetDropdown.className = '';
                datasetDropdown.classList.add('col-md-5');
    } else {
        testDropdown.style.display = 'none';
        datasetDropdown.className = '';
        datasetDropdown.classList.add('col-md-10');
    }
}
    
datasetDropdownMenu.addEventListener('change', showTestDropdown);
document.addEventListener('DOMContentLoaded', showTestDropdown);