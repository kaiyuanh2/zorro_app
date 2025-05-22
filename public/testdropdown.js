const datasetDropdown = document.getElementById('datasetSelect');
const datasetDropdownMenu = document.getElementById('item');
const testDropdown = document.getElementById('testSelect');
const LRInput = document.getElementById('learningRate');
const regInput = document.getElementById('regularization');

function showTestDropdown() {
    if (datasetDropdownMenu.value) {
        testDropdown.style.display = 'block';
        LRInput.style.display = 'block';
        regInput.style.display = 'block';
        datasetDropdown.className = '';
        datasetDropdown.classList.add('col-md-4');
    } else {
        testDropdown.style.display = 'none';
        LRInput.style.display = 'none';
        regInput.style.display = 'none';
        datasetDropdown.className = '';
        datasetDropdown.classList.add('col-md-10');
    }
}

function updateTestOptions() {
    const itemSelect = document.getElementById('item');
    const testSelect = document.getElementById('test');
    const selectedItem = itemSelect.value;
    
    while (testSelect.options.length > 1) {
        testSelect.options.remove(1);
    }
    
    if (selectedItem && lengthMap[selectedItem]) {
        const count = lengthMap[selectedItem];
        for (let i = 1; i <= count; i++) {
            const option = document.createElement('option');
            option.value = 't' + i;
            option.text = 'Test Set ' + i;
            testSelect.add(option);
        }
    }
}
    
datasetDropdownMenu.addEventListener('change', showTestDropdown);
document.addEventListener('DOMContentLoaded', showTestDropdown);

document.getElementById('item').addEventListener('change', updateTestOptions);
updateTestOptions();