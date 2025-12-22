const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const dotenv = require('dotenv');
dotenv.config();
const employeeRoutes = require('./routes/employeeRoutes');

const app = express();
const port = process.env.PORT || 3000;
app.use(cors());
app.use(bodyParser.json());

//routes
app.use('/api', employeeRoutes);

//connection
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});