const express = require('express');
const {
	createEmployee,
	getAllEmployees,
	getEmployeeById,
	updateEmployee,
	deleteEmployee,
} = require('../models/employee.model');

const router = express.Router();


// Employee routes
router.post('/employee', createEmployee);
router.get('/employee', getAllEmployees);
router.get('/employee/:id', getEmployeeById);
router.put('/employee/:id', updateEmployee);
router.delete('/employee/:id', deleteEmployee);

module.exports = router;