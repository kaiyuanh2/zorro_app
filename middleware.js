const {PythonShell} = require("python-shell");
const path = require('path');
const fs = require('fs');
const Joi = require('joi');

const schema = Joi.object().keys({
    item: Joi.string().required().default('Insurance'),
    test: Joi.string().required().default('t1')
}).unknown(true);

module.exports.validateParametersPost = async (req, res, next) => {
    // console.log('validating...');
    const { error, value } = schema.validate(req.body);
    if (error) {
        console.log(error);
        req.body = schema.validate({}).value;
        req.flash('error', "Invalid parameter(s)! Displaying the page under default options.");
    } else {
        req.body = value;
    }
    next();
}

function createNewTrain(req, res) {
    return new Promise((resolve, reject) => {
        const py = new PythonShell('./train_script.py');
        py.on('message', function (message) {
            console.log(message);
        });

        py.send(req.files.dataSet[0].path);
        py.send(req.files.dataLabel[0].path);
        py.send(req.files.dataClean[0].path);
        py.send(req.body.dataName);
        py.send(req.body.dataDesc);

        // Switch comment here to change the dev mode of Python script
        // py.send('dev');
        py.send('');

        // console.log('arg sent');
        py.end(function (err) {
            flag = true;
            if (err){
                console.log(err);
                flag = false;
            }
            else {
                console.log('success');
            }
            resolve(flag);
            console.log('finished');
        });
    });
}

function createNewTest(req, res) {
    return new Promise((resolve, reject) => {
        const py = new PythonShell('./test_script.py');
        py.on('message', function (message) {
            console.log(message);
        });

        py.send(req.files.dataSetTest[0].path);
        py.send(req.files.dataLabelTest[0].path);
        py.send(req.body.testSetName);
        py.send(req.body.testSetNameSelf);

        // Switch comment here to change the dev mode of Python script
        // py.send('dev');
        py.send('');

        // console.log('arg sent');
        py.end(function (err) {
            flag = true;
            if (err){
                console.log(err);
                flag = false;
            }
            else {
                console.log('success');
            }
            resolve(flag);
            console.log('finished');
        });
    });
}

function deleteFile(filePath) {
    try {
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }
    } catch (err) {
      console.error('Error deleting file:', err);
    }
}

module.exports.uploadProcess = async (req, res, next) => {
    // req.flash('success', "Test");
    createNewTrain(req, res).then(flag => {
        msg = 'Upload ';
        if (flag) {
            // console.log('success flag');
            finalDirectory = path.join(__dirname, 'dataset', req.body.dataName);
            const finalFeaturePath = path.join(finalDirectory, req.files.dataSet[0].filename);
            const finalLabelPath = path.join(finalDirectory, req.files.dataLabel[0].filename);
            const finalCleanPath = path.join(finalDirectory, req.files.dataClean[0].filename);
            
            if (!fs.existsSync(finalDirectory)) {
                fs.mkdirSync(finalDirectory, { recursive: true });
            }

            fs.renameSync(req.files.dataSet[0].path, finalFeaturePath);
            fs.renameSync(req.files.dataLabel[0].path, finalLabelPath);
            fs.renameSync(req.files.dataClean[0].path, finalCleanPath);
            
            req.files.dataSet[0].path = finalFeaturePath;
            req.files.dataLabel[0].path = finalLabelPath;
            req.files.dataClean[0].path = finalCleanPath;
            req.flash('success', msg + 'success! It might take a while for the new training dataset to appear on the list.')
        }
        else {
            // console.log('fail flag');
            deleteFile(req.files.dataSet[0].path);
            deleteFile(req.files.dataLabel[0].path);
            deleteFile(req.files.dataClean[0].path);
            req.flash('error', msg + 'failed! Please double check your submission and try again.')
        }

        next();
    }).catch(err => {
        console.log(err);
    })
}

module.exports.uploadProcessTest = async (req, res, next) => {
    // req.flash('success', "Test");
    createNewTest(req, res).then(flag => {
        msg = 'Upload ';
        if (flag) {
            // console.log('success flag');
            finalDirectory = path.join(__dirname, 'dataset', req.body.testSetName);
            const finalFeaturePath = path.join(finalDirectory, req.files.dataSetTest[0].filename);
            const finalLabelPath = path.join(finalDirectory, req.files.dataLabelTest[0].filename);
            
            if (!fs.existsSync(finalDirectory)) {
                fs.mkdirSync(finalDirectory, { recursive: true });
            }

            fs.renameSync(req.files.dataSetTest[0].path, finalFeaturePath);
            fs.renameSync(req.files.dataLabelTest[0].path, finalLabelPath);
            
            req.files.dataSetTest[0].path = finalFeaturePath;
            req.files.dataLabelTest[0].path = finalLabelPath;
            req.flash('success', msg + 'success! It might take a while for the new testing dataset to appear on the list.')
        }
        else {
            // console.log('fail flag');
            deleteFile(req.files.dataSetTest[0].path);
            deleteFile(req.files.dataLabelTest[0].path);
            req.flash('error', msg + 'failed! Please double check your submission and try again.')
        }

        next();
    }).catch(err => {
        console.log(err);
    })
}