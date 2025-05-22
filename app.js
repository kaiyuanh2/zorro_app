const express = require('express');
const app = express();
const { exec } = require("child_process");
const path = require('path');
const ejsMate = require('ejs-mate');
const methodOverride = require('method-override');
const ExpressError = require('./utils/ExpressError');
const flash = require('connect-flash');
const {validateParametersPost, uploadProcess, uploadProcessTest} = require('./middleware');
const session = require('express-session');
const fs = require('fs');

const multer = require('multer')
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, path.join(__dirname, './uploads'));
    },
    filename: function (req, file, cb) {
      const fileExtension = path.extname(file.originalname);
      const nameWithoutExtension = path.basename(file.originalname, fileExtension);
      const newFilename = nameWithoutExtension + '_' + Date.now() + fileExtension;
      
      cb(null, newFilename);
    }
  });

const upload = multer({ storage }).fields([
    { name: 'dataSet', maxCount: 1 },
    { name: 'dataClean', maxCount: 1 },
    { name: 'dataLabel', maxCount: 1 }
]);

const uploadTest = multer({ storage }).fields([
    { name: 'dataSetTest', maxCount: 1 },
    { name: 'dataLabelTest', maxCount: 1 }
]);

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

app.get('/visualizations', (req, res) => {
    var item = '';
    var test = '';
    var lr = 0;
    var reg = 0;
    var expression = '';
    var weight_max = [0];
    var weight_min = [0];
    var weight_mid = [0];
    var features = ["a"];
    var robustness = [0];
    var ub = [0];
    var lb = [0];
    var x_test = [0];
    var missing = [0];
    var missingy = [0];
    var clean = [0];
    var cleany = [0];
    var oneimp = [0];
    var json2D = ["a"];
    var json3D = ["a"];
    var missing_f = '';
    var missing_c = 0;

    datasetJSString = fs.readFileSync("./public/custom.json").toString();
    dataset_map = new Map(Object.entries(JSON.parse(datasetJSString)));
    keys = Array.from(dataset_map.keys());
    console.log(keys);

    testJSString = fs.readFileSync("./public/custom_test.json").toString();
    test_js = JSON.parse(testJSString);

    var lengthMap = {};
    for (const key in test_js) {
      if (Array.isArray(test_js[key])) {
        lengthMap[key] = test_js[key].length;
      }
    }
    console.log(lengthMap);

    res.render('vis', {lengthMap, keys, item, test, expression, weight_max, weight_min, weight_mid, features, robustness, ub, lb, x_test, json2D, json3D, missing, missingy, clean, cleany, oneimp, missing_f, missing_c, messages: req.flash('error')});
})

app.post('/visualizations', validateParametersPost, (req, res) => {
    // console.log(req.body);
    var item = req.body.item;
    var test = req.body.test;
    var lr = req.body.lr;
    var reg = req.body.reg;
    var expression = '';
    var weight_max = [0];
    var weight_min = [0];
    var weight_mid = [0];
    var features = ["a"];
    var robustness = [0];
    var ub = [0];
    var lb = [0];
    var x_test = [0];
    var missing = [0];
    var missingy = [0];
    var clean = [0];
    var cleany = [0];
    var oneimp = [0];
    var missing_f = '';
    var missing_c = 0;

    datasetJSString = fs.readFileSync("./public/custom.json").toString();
    dataset_map = new Map(Object.entries(JSON.parse(datasetJSString)));
    keys = Array.from(dataset_map.keys());

    testJSString = fs.readFileSync("./public/custom_test.json").toString();
    test_js = JSON.parse(testJSString);

    var lengthMap = {};
    for (const key in test_js) {
      if (Array.isArray(test_js[key])) {
        lengthMap[key] = test_js[key].length;
      }
    }

    // modify between python and python3 according to your system
    const command = `python3 public/process.py ${item} ${test} ${lr} ${reg}`;
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
            ub = output.ub;
            lb = output.lb;
            x_test = output.X_test;
            missing = output.missing;
            missingy = output.missingy;
            clean = output.clean;
            cleany = output.cleany;
            oneimp = output.oneimp;
            missing_f = output.missing_feature;
            missing_c = output.missing_column;
        }
        
        console.log(features);

        if (item == 'Insurance') {
            var json2D = JSON.parse(fs.readFileSync('./models/ins/ins_30_2d.json', 'utf-8'));
            var json3D = JSON.parse(fs.readFileSync('./models/ins/ins_30_3d.json', 'utf-8'));
        }
        else {
            var json2D = JSON.parse(fs.readFileSync('./models/' + item + '/' + item + '_2d.json', 'utf-8'));
            var json3D = ["a"];
        }
        res.render('vis', {lengthMap, item, test, expression, weight_max, weight_min, weight_mid, features, robustness, ub, lb, x_test, json2D, json3D, missing, missingy, clean, cleany, oneimp, missing_f, missing_c, messages: req.flash('error')});
    });

})

app.get('/dataset', (req, res) => {
    const page_name = 'dataset';
    res.render('dataset', {page_name, success: req.flash('success'), error: req.flash('error')});
})

app.post('/dataset', upload, uploadProcess, (req, res) => {
    const page_name = 'dataset';
    res.render('dataset', {page_name, success: req.flash('success'), error: req.flash('error')});
})

app.get('/test-dataset', (req, res) => {
    const page_name = 'test-dataset';
    res.render('dataset_test', {page_name, success: req.flash('success'), error: req.flash('error')});
})

app.post('/test-dataset', uploadTest, uploadProcessTest, (req, res) => {
    const page_name = 'test-dataset';
    res.render('dataset_test', {page_name, success: req.flash('success'), error: req.flash('error')});
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