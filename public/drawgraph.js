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

window.onresize = function() {
    Plotly.Plots.resize(document.getElementById('weightGraph'));
    Plotly.Plots.resize(document.getElementById('robustnessGraph'));
};