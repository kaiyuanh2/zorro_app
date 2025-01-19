// window.onresize = function() {
//     Plotly.Plots.resize(document.getElementById('zonotopeGraph'));
// };

const f1Dropdown3d = document.getElementById('feature13d');
const f2Dropdown3d = document.getElementById('feature23d');
const f3Dropdown3d = document.getElementById('feature33d');
const zonotopeG3d = document.getElementById('zonotopeG3D');

function drawZonotope3D() {
    if (!f1Dropdown3d.value || !f2Dropdown3d.value || !f3Dropdown3d.value) {
        zonotopeG3d.innerHTML = '<p class="card-text">Select 3 different features (including offset) to show zonotope.</p>'
    } else if ((f1Dropdown3d.value === f2Dropdown3d.value) || (f1Dropdown3d.value === f3Dropdown3d.value) || (f2Dropdown3d.value === f3Dropdown3d.value)) {
        zonotopeG3d.innerHTML = '<p class="card-text">At least 2 features are the same. Check the dropdowns and try again.</p>'
    } else {
        zonotopeG3d.innerHTML = '<div id="zonotopeGraph3D"></div>'
        const keys3D = Object.keys(jsonData3D);
        const key3D1 = f1Dropdown3d.value + "," + f2Dropdown3d.value + "," + f3Dropdown3d.value;
        if (keys3D.includes(key3D1)) {
            var trace1_3d = { x: jsonData3D[key3D1][0],
            y: jsonData3D[key3D1][1],
            z: jsonData3D[key3D1][2],
            i: jsonData3D[key3D1][3],
            j: jsonData3D[key3D1][4],
            k: jsonData3D[key3D1][5],
            name: 'Zonotope', opacity: 0.3, type: 'mesh3d', color: 'blue' };
        
            var trace2_3d = {
                    x: [jsonData3D[f1Dropdown3d.value]],
                    y: [jsonData3D[f2Dropdown3d.value]],
                    z: [jsonData3D[f3Dropdown3d.value]],
                    name: 'Ground Truth',
                    mode: 'markers',
                    type: 'scatter3d',
                    marker: { size: 7, color: 'orange' }
            };
            var z3ddata = [trace1_3d, trace2_3d];
            var z3dlayout = {
                title: '<b>3D Zonotope</b>',
                showlegend: true,
                scene: {xaxis: { title: f1Dropdown3d.value, tickformat: '.4f' }, yaxis: { title: f2Dropdown3d.value, tickformat: '.4f' }, zaxis: { title: f3Dropdown3d.value, tickformat: '.4f' }}
            };
            Plotly.newPlot('zonotopeGraph3D', z3ddata, z3dlayout);
        }
        
    }
}

function loadInstructions3D() {
    zonotopeG3d.innerHTML = '<p class="card-text">Select 3 different features (including offset) to show zonotope.</p>'
}

f1Dropdown3d.addEventListener('change', drawZonotope3D);
f2Dropdown3d.addEventListener('change', drawZonotope3D);
f3Dropdown3d.addEventListener('change', drawZonotope3D);
document.addEventListener('DOMContentLoaded', loadInstructions3D);