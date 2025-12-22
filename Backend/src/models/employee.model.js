const pool = require('../DB/config');
const fs = require('fs');
const path = require('path');

const REQUIRED_FIELDS = ['first_name', 'last_name', 'email'];
const UPDATABLE_FIELDS = [
	'first_name',
	'last_name',
	'email',
	'phone',
	'date_of_birth',
	'gender',
	'hire_date',
	'job_title',
	'salary',
	'is_active',
];

/**
 * Register face encoding with the Python face recognition system.
 * @param {string} name - Employee name (first_name + last_name)
 * @returns {Promise<number>} - Returns 0 for success, 1 for failure
 */
const registerFaceEncoding = async (name) => {
	try {
		// Call enroll script to enroll a new face
		const registerFace = path.join(__dirname, '../../../modelling/arc_face/arcface_enroll.py');
		const { exec } = require('child_process');

		const exitCode = await new Promise((resolve, reject) => {
			exec(`python "${registerFace}" "${name}"`, (error, stdout, stderr) => {
				// Log the output
				if (stdout) console.log('[INFO] Face encoding response:', stdout);
				if (stderr) console.log('[INFO] Python stderr:', stderr);

				if (error) {
					// Return the exit code from the error
					console.error('[ERROR] Face encoding failed with exit code:', error.code);
					resolve(error.code || 1);
				} else {
					// Success
					resolve(0);
				}
			});
		});

		return exitCode;
	} catch (error) {
		console.error('[ERROR] Error registering face encoding:', error.message);
		return 1; // Return failure code
	}
};

/**
 * Create a new employee with image and face encoding registration.
 */
const createEmployee = async (req, res) => {
	try {
		console.log('[INFO] Received request to create employee:', req.body);
		const missing = REQUIRED_FIELDS.filter((field) => !req.body[field]);
		if (missing.length) {
			return res.status(400).json({ message: `Missing required fields: ${missing.join(', ')}` });
		}
		const {
			first_name,
			last_name,
			email,
			phone,
			date_of_birth,
			gender,
			hire_date,
			job_title,
			salary,
			is_active,
		} = req.body;

		// First, attempt face enrollment
		const fullName = `${first_name} ${last_name}`;
		console.log(`[INFO] Starting face enrollment for ${fullName}`);
		
		const enrollmentStatus = await registerFaceEncoding(fullName);
		
		if (enrollmentStatus !== 0) {
			console.error(`[ERROR] Face enrollment failed for ${fullName}`);
			return res.status(400).json({ 
				message: 'Face enrollment failed. Please ensure the employee image is in the known_faces_arc directory and try again.',
				error: 'FACE_ENROLLMENT_FAILED'
			});
		}

		console.log(`[INFO] Face enrollment successful for ${fullName}. Proceeding with database insert.`);

		// Only insert into database if face enrollment succeeds
		const insertQuery = `
			INSERT INTO employee (
				first_name, last_name, email, phone, date_of_birth, gender,
				hire_date, job_title, salary, is_active
			)
			VALUES ($1, $2, $3, $4, $5, $6, COALESCE($7, CURRENT_DATE), $8, $9, COALESCE($10, TRUE))
			RETURNING *;
		`;

		const values = [
			first_name,
			last_name,
			email,
			phone || null,
			date_of_birth || null,
			gender || null,
			hire_date || null,
			job_title || null,
			salary || null,
			is_active,
		];
		
		const { rows } = await pool.query(insertQuery, values);
		const employee = rows[0];

		return res.status(201).json(employee);
	} catch (error) {
		console.error('Error creating employee:', error);
		return res.status(500).json({ message: 'Internal server error' });
	}
};

/**
 * Get all employees.
 */
const getAllEmployees = async (_req, res) => {
	try {
		const { rows } = await pool.query('SELECT * FROM employee ORDER BY employee_id;');
		return res.status(200).json(rows);
	} catch (error) {
		console.error('Error fetching employees:', error);
		return res.status(500).json({ message: 'Internal server error' });
	}
};

/**
 * Get employee by ID.
 */
const getEmployeeById = async (req, res) => {
	try {
		const { id } = req.params;
		const employeeId = Number(id);

		if (!Number.isInteger(employeeId)) {
			return res.status(400).json({ message: 'Employee id must be an integer' });
		}

		const { rows } = await pool.query('SELECT * FROM employee WHERE employee_id = $1;', [
			employeeId,
		]);

		if (!rows.length) {
			return res.status(404).json({ message: 'Employee not found' });
		}

		return res.status(200).json(rows[0]);
	} catch (error) {
		console.error('Error fetching employee:', error);
		return res.status(500).json({ message: 'Internal server error' });
	}
};

/**
 * Update employee by ID.
 */
const updateEmployee = async (req, res) => {
	try {
		const { id } = req.params;
		const employeeId = Number(id);

		if (!Number.isInteger(employeeId)) {
			return res.status(400).json({ message: 'Employee id must be an integer' });
		}

		const entries = Object.entries(req.body).filter(([key]) => UPDATABLE_FIELDS.includes(key));

		if (!entries.length) {
			return res.status(400).json({ message: 'No valid fields provided for update' });
		}

		const setClauses = entries.map(([key], index) => `${key} = $${index + 1}`);
		const values = entries.map(([, value]) => value);

		const updateQuery = `
			UPDATE employee
			SET ${setClauses.join(', ')}, updated_at = NOW()
			WHERE employee_id = $${entries.length + 1}
			RETURNING *;
		`;

		const { rows } = await pool.query(updateQuery, [...values, employeeId]);

		if (!rows.length) {
			return res.status(404).json({ message: 'Employee not found' });
		}

		return res.status(200).json(rows[0]);
	} catch (error) {
		console.error('Error updating employee:', error);
		return res.status(500).json({ message: 'Internal server error' });
	}
};

/**
 * Delete employee by ID.
 */
const deleteEmployee = async (req, res) => {
	try {
		const { id } = req.params;
		const employeeId = Number(id);

		if (!Number.isInteger(employeeId)) {
			return res.status(400).json({ message: 'Employee id must be an integer' });
		}

		const { rowCount } = await pool.query('DELETE FROM employee WHERE employee_id = $1;', [
			employeeId,
		]);

		if (rowCount === 0) {
			return res.status(404).json({ message: 'Employee not found' });
		}
		else{
			console.log(`Employee with ID ${employeeId} deleted successfully.`);
		}
		return res.status(204).send();
	} catch (error) {
		console.error('Error deleting employee:', error);
		return res.status(500).json({ message: 'Internal server error' });
	}
};

module.exports = {
	createEmployee,
	getAllEmployees,
	getEmployeeById,
	updateEmployee,
	deleteEmployee,
};