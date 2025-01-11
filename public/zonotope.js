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
        var trace1 = { x: [13186.8601263555, 13192.3417000781, 13195.7639231732, 13200.4476482393,
            13226.4446672194, 13233.1658830590, 13236.8482157872, 13239.6725868144,
            13243.0395191149, 13248.6740904175, 13250.4100714157, 13251.5363525808,
            13252.9526634506, 13247.4710897281, 13244.0488666330, 13239.3651415669,
            13213.3681225868, 13206.6469067472, 13202.9645740190, 13200.1402029918,
            13196.7732706913, 13191.1386993887, 13189.4027183905, 13188.2764372254, 13186.8601263555],
            y: [3400.14011161642, 3388.77659929494, 3386.67135054473, 3386.12257187276,
                3384.10604244336, 3383.87402024111, 3384.09596886920, 3386.50494959996,
                3392.88838548804, 3412.30996513806, 3434.69016465433, 3458.59949864044,
                3489.00236859030, 3500.36588091178, 3502.47112966198, 3503.01990833395,
                3505.03643776335, 3505.26845996561, 3505.04651133752, 3502.63753060675,
                3496.25409471867, 3476.83251506866, 3454.45231555239, 3430.54298156628, 3400.14011161642],
            name: 'Zonotope', fill: 'toself', type: 'scatter' };
        
        var trace2 = {
                x: [13211.466615457586],
                y: [3441.065865402959],
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

function loadInstructions() {
    zonotopeG.innerHTML = '<p class="card-text">Select 2 different features (including offset) to show zonotope.</p>'
}

f1Dropdown.addEventListener('change', drawZonotope);
f2Dropdown.addEventListener('change', drawZonotope);
document.addEventListener('DOMContentLoaded', loadInstructions);