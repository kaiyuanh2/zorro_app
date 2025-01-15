var stats = { x: x_labels, y: weights,
    error_y: { type: 'data', symmetric: false,
                array: weights_max,
                arrayminus: weights_min,
                visible: true },
    mode: 'markers' };
var data = [stats];
var layout = {
    title: 'Ranges of Abstract Model Weights',
    xaxis: { title: 'Features' }, yaxis: { title: 'Weights', tickformat: '.4f' }
 };
Plotly.newPlot('weightGraph', data, layout);

var trace_robust = {
    x: [1, 2, 3, 4, 5, 6],
    y: robustness_ratios,
    type: 'scatter'
};
var data_robust = [trace_robust];
var layout_robust = {
    title: 'Robustness Ratio Under Different Robustness Radius',
    xaxis: { title: 'Robustness Radius = Percentage of Median of y_test', tickvals: [1, 2, 3, 4, 5, 6], ticktext: ["1%", "2%", "3%", "5%", "10%", "20%"] },
    yaxis: { title: 'Robustness Ratio', tickformat: ',.0%' }
};
Plotly.newPlot('robustnessGraph', data_robust, layout_robust);

const fmDropdown = document.getElementById('featurem');
const missingG = document.getElementById('missingG');

function drawMissing() {
    if (!fmDropdown.value) {
        missingG.innerHTML = '<p class="card-text">Select a non-missing feature from above.</p>'
    } else {
        missingG.innerHTML = '<div id="missingGraph"></div>'
        if (fmDropdown.value == 'f1') {
            var missing_x = missing_f1;
            var xlabel = x_labels[0]
        } else if (fmDropdown.value == 'f2') {
            var missing_x = missing_f2;
            var xlabel = x_labels[2]
        }

        var trace_missing = {
            x: missing_x,
            y: missing_y,
            mode: 'markers',
            type: 'scatter',
            marker: { size: 10}
        };
        var data_missing = [trace_missing];
        var layout_missing = {
            title: 'Distribution of Missing Data Points',
            xaxis: { title: xlabel }, yaxis: { title: 'Label', tickformat: '.4f'  }
         };
        Plotly.newPlot('missingGraph', data_missing, layout_missing);
    }
}

function loadInstructionsM() {
    missingG.innerHTML = '<p class="card-text">Select a non-missing feature from above.</p>'
}

fmDropdown.addEventListener('change', drawMissing);
document.addEventListener('DOMContentLoaded', loadInstructionsM);

window.onresize = function() {
    Plotly.Plots.resize(document.getElementById('weightGraph'));
    Plotly.Plots.resize(document.getElementById('robustnessGraph'));
    Plotly.Plots.resize(document.getElementById('missingGraph'));
}