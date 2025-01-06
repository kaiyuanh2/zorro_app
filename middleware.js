const Joi = require('joi');

const schema = Joi.object().keys({
    item: Joi.string().valid('i1', 'i2', 'i3').required().default('i1'),
    imp: Joi.string().valid('s1', 's2', 's3').required().default('s1'),
    test: Joi.string().valid('y', 'n').required().default('n'),
    checkboxStr: Joi.string().min(2).required().default('aa')
}).unknown(true);

module.exports.validateParameters = async (req, res, next) => {
    // console.log('validating...');
    // const { error, value } = schema.validate(req.query);
    // if (error) {
    //     console.log('param error');
    //     req.query = schema.validate({}).value;
    //     req.flash('error', "Invalid parameter(s)! Displaying the map under default options: Aerobic Capacity, 5th Grade, and 2018-2019 School Year");
    // } else {
    //     req.query = value;
    // }
    next();
}

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