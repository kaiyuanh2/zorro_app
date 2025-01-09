const Joi = require('joi');

const schema = Joi.object().keys({
    item: Joi.string().valid('i1', 'i2', 'i3').required().default('i1'),
    test: Joi.string().valid('t1', 't2').required().default('t1')
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