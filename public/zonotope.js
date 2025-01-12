// window.onresize = function() {
//     Plotly.Plots.resize(document.getElementById('zonotopeGraph'));
// };

const f1Dropdown = document.getElementById('feature1');
const f2Dropdown = document.getElementById('feature2');
const zonotopeG = document.getElementById('zonotopeG');

function drawZonotope() {
    if (!f1Dropdown.value && !f2Dropdown.value) {
        zonotopeG.innerHTML = '<p class="card-text">Select 2 different features (including offset) to show zonotope.</p>'
    } else if (!f1Dropdown.value || !f2Dropdown.value) {
        zonotopeG.innerHTML = '<p class="card-text">Select 1 more feature (including offset, make sure 2 features are different) to show zonotope.</p>'
    } else if (f1Dropdown.value === f2Dropdown.value) {
        zonotopeG.innerHTML = '<p class="card-text">Please select 2 different features. Check the dropdowns and try again.</p>'
    } else {
        zonotopeG.innerHTML = '<div id="zonotopeGraph"></div>'
        const keys2D = Object.keys(jsonData2D);
        // console.log(keys2D);
        const key2D1 = f1Dropdown.value + "," + f2Dropdown.value;
        // console.log(key2D1);
        // console.log(key2D2);
        if (keys2D.includes(key2D1)) {
            // console.log(jsonData2D[key2D1][0]);
            var trace1 = { x: jsonData2D[key2D1][0],
                y: jsonData2D[key2D1][1],
                name: 'Zonotope', fill: 'toself', type: 'scatter' };
            
            var trace2 = {
                    x: [jsonData2D[f1Dropdown.value]],
                    y: [jsonData2D[f2Dropdown.value]],
                    name: 'Ground Truth',
                    mode: 'markers',
                    type: 'scatter'
            };
            var zdata = [trace1, trace2];
            var zlayout = {
                title: '2D Zonotope',
                xaxis: { title: 'Feature 1', tickformat: '.4f' }, yaxis: { title: 'Feature 2', tickformat: '.4f' }
            };
            Plotly.newPlot('zonotopeGraph', zdata, zlayout);
        }
    }
}

function loadInstructions() {
    zonotopeG.innerHTML = '<p class="card-text">Select 2 different features (including offset) to show zonotope.</p>'
}

f1Dropdown.addEventListener('change', drawZonotope);
f2Dropdown.addEventListener('change', drawZonotope);
document.addEventListener('DOMContentLoaded', loadInstructions);