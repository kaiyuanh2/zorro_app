const express = require('express');
const app = express();
const { exec } = require("child_process");
const path = require('path');
const ejsMate = require('ejs-mate');
const methodOverride = require('method-override');
const ExpressError = require('./utils/ExpressError');
const flash = require('connect-flash');
const {validateParameters, validateParametersPost} = require('./middleware');
const session = require('express-session');
const fs = require('fs');

app.use(express.urlencoded({extended: true}));
app.use(methodOverride('_method'));
app.use('/node_modules', express.static(__dirname + '/node_modules/'));

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, '/view'))

app.engine('ejs', ejsMate)

app.use(express.static(path.join(__dirname, 'public')))
const sessionConfig = {
    name: 'mapsession',
    secret: 'thisisasecret',
    resave: false,
    saveUninitialized: true,
    cookie: {
        httpOnly: true,
        expires: Date.now() + 86400000,
        maxAge: 86400000
    }
};
app.use(session(sessionConfig));
app.use(flash());

app.get('/', (req, res) => {
    res.render('index')
})

app.get('/visualizations', validateParameters, (req, res) => {
    var item = '';
    var test = '';
    var expression = '';
    var weight_max = [0];
    var weight_min = [0];
    var weight_mid = [0];
    var features = ["a"];
    var robustness = [0];
    var center = [0];
    var ub = [0];
    var lb = [0];
    var x_test = [0];
    const json2D = ["a"];
    const json3D = ["a"];
    res.render('vis', {item, test, expression, weight_max, weight_min, weight_mid, features, robustness, center, ub, lb, x_test, json2D, json3D, messages: req.flash('error')});
})

app.post('/visualizations', validateParametersPost, (req, res) => {
    // console.log(req.body);
    var item = req.body.item;
    var test = req.body.test;
    var expression = '';
    var weight_max = [0];
    var weight_min = [0];
    var weight_mid = [0];
    var features = ["a"];
    var robustness = [0];
    var center = [0];
    var ub = [0];
    var lb = [0];
    var x_test = [0];
    const command = `python public/process.py ${item} ${test}`;
    exec(command, (err, stdout, stderr) => {
        if (err) {
            console.error(`Error: ${err.message}`);
        }
        if (stderr) {
            console.error(`Stderr: ${stderr}`);
        }
        if (stdout) {
            console.log("Python success")
            const data = stdout;
            const output = JSON.parse(data.toString());
            expression = output.latex;
            weight_max = output.wt_max;
            weight_min = output.wt_min;
            weight_mid = output.wt_mid;
            features = JSON.stringify(output.features);
            robustness = output.robustness;
            center = output.centers;
            ub = output.ub;
            lb = output.lb;
            x_test = output.X_test;
        }
        
        console.log(features);
        const json2D = JSON.parse(fs.readFileSync('./models/ins/ins_30_2d.json', 'utf-8'));
        const json3D = JSON.parse(fs.readFileSync('./models/ins/ins_30_3d.json', 'utf-8'));
        res.render('vis', {item, test, expression, weight_max, weight_min, weight_mid, features, robustness, center, ub, lb, x_test, json2D, json3D, messages: req.flash('error')});
    });

})

app.get('/dataset', (req, res) => {
    const page_name = 'dataset';
    res.render('dataset', {page_name, success: req.flash('success'), error: req.flash('error')});
})

app.post('/dataset', (req, res) => {
    const page_name = 'dataset';
    res.render('dataset', {page_name, success: req.flash('success'), error: req.flash('error')});
})

app.all('*', (req, res, next) => {
    next(new ExpressError('Page Not Found', 404))
})

app.use((err, req, res, next) => {
    const { statusCode = 500 } = err;	
    if (!err.message) err.message = 'Error!'	
    res.status(statusCode).render('error', { err })
})

app.listen(3000, () => {
    console.log("LISTENING ON PORT 3000")
})