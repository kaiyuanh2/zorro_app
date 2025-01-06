const express = require('express');
const app = express();
const path = require('path');
const ejsMate = require('ejs-mate');
const methodOverride = require('method-override');
const ExpressError = require('./utils/ExpressError');
const flash = require('connect-flash');
const {validateParameters, validateParametersPost} = require('./middleware');
const session = require('express-session');

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
    var grade = '';
    var year = '';
    var checkboxStr = '';
    res.render('vis', {item, grade, year, checkboxStr, messages: req.flash('error')});
})

app.post('/visualizations', validateParametersPost, (req, res) => {
    console.log(req.body);
    var item = req.body.item;
    var grade = req.body.grade;
    var year = req.body.year;
    var checkboxStr = req.body.checkboxStr;
    res.render('vis', {item, grade, year, checkboxStr, messages: req.flash('error')});
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