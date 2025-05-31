var stats = { x: x_labels, y: weights,
    error_y: { type: 'data', symmetric: false,
                array: weights_max,
                arrayminus: weights_min,
                visible: true },
    mode: 'markers' };
var data = [stats];
var layout = {
    title: '<b>Ranges of Abstract Model Weights</b>',
    xaxis: { title: 'Features' }, yaxis: { title: 'Weights', tickformat: '.0f' } //, height: 350
 };
Plotly.newPlot('weightGraph', data, layout);

function isFPattern(str) {
    return /^f\d+$/.test(str);
  }

weightGraph.on('plotly_click', function(eventData){
    var pt = eventData.points[0];
    var trace = pt.data;
    var tidx = pt.pointNumber;
    var ptx = pt.x;
    var pty = pt.y;
    var deltay = pty;
    if (trace.error_y) {
        deltay = trace.error_y.array[tidx] * 2;
    }
        
    var update = {
        'yaxis.range': [pty - deltay, pty + deltay]
    }; 
    Plotly.relayout('weightGraph', update);
});

var trace_robust = {
    x: [1, 2, 3, 4, 5, 6],
    y: robustness_ratios,
    type: 'scatter'
};
var data_robust = [trace_robust];
var layout_robust = {
    title: '<b>Robustness Ratio Under Different Robustness Radius</b>',
    xaxis: { title: 'Robustness Radius', tickvals: [1, 2, 3, 4, 5, 6], ticktext: robustness_radius },
    yaxis: { title: 'Robustness Ratio', tickformat: ',.0%' }, // height: 350
};
Plotly.newPlot('robustnessGraph', data_robust, layout_robust);

const fmDropdown = document.getElementById('featurem');
const missingG = document.getElementById('missingG');

function drawMissing() {
    if (!fmDropdown.value) {
        missingG.innerHTML = '<p class="card-text">Select a non-missing feature from above.</p>'
    } else {
        missingG.innerHTML = '<div id="missingGraph"></div>'

        if (isFPattern(fmDropdown.value)) {
            const idx = Number(fmDropdown.value.slice(1))
            var missing_x = missing_f[idx - 1];
            var clean_x = clean_f[idx - 1];
            var new_index = idx - 1;
            if (new_index >= missing_column) {
                new_index = new_index + 1;
            }
            var xlabel = x_labels[new_index];
            console.log("missing_x: ", missing_x);
            console.log("missing_y: ", missing_y);
        }

        // if (fmDropdown.value == 'f1') {
        //     const idx = Number(fmDropdown.value.slice(1))
        //     var missing_x = missing_f[idx - 1];
        //     var clean_x = clean_f[idx - 1];
        //     var xlabel = x_labels[0];
        // } else if (fmDropdown.value == 'f2') {
        //     const idx = Number(fmDropdown.value.slice(1))
        //     var missing_x = missing_f[idx - 1];
        //     var clean_x = clean_f[idx - 1];
        //     var xlabel = x_labels[2];
        // }

        var trace_missing = {
            x: missing_x,
            y: missing_y,
            name: 'Data Points with Features Missing',
            mode: 'markers',
            type: 'scatter',
            marker: { size: 10, color: "red" }
        };
        var trace_clean = {
            x: clean_x,
            y: clean_y,
            name: 'Clean Data Points',
            mode: 'markers',
            type: 'scatter',
            marker: { size: 10, color: "blue" }
        };
        var data_missing = [trace_clean, trace_missing];
        var layout_missing = {
            title: '<b>Distribution of Missing Data Points</b>',
            xaxis: { title: xlabel }, yaxis: { title: 'label', tickformat: '.0f' }, // height: 350
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