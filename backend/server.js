// const express = require('express');
// const mongoose = require('mongoose');
// const cors = require('cors');
// require('dotenv').config();

// const User = require('./models/User');

// const app = express();
// app.use(cors());
// app.use(express.json());

// mongoose.connect(process.env.MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
//   .then(() => console.log('MongoDB connected'))
//   .catch(err => console.log(err));

// app.post('/signup', async (req, res) => {
//   const { username, email, password } = req.body;
//   try {
//     const user = new User({ username, email, password });
//     await user.save();
//     res.status(201).json({ message: 'User registered!' });
//   } catch (err) {
//     res.status(500).json({ error: 'Signup failed' });
//   }
// });
// // ...existing code...

// app.post('/login', async (req, res) => {
//   const { email, password } = req.body;
//   try {
//     const user = await User.findOne({ email, password });
//     if (user) {
//       res.json({ message: 'Login successful' });
//     } else {
//       res.status(401).json({ error: 'Incorrect email or password' });
//     }
//   } catch (err) {
//     res.status(500).json({ error: 'Login failed' });
//   }
// });
// app.post('/logout', async (req, res) => {
//   const { email } = req.body;
//   try {
//     const result = await User.deleteOne({ email });
//     if (result.deletedCount === 1) {
//       res.json({ message: 'User deleted and logged out' });
//     } else {
//       res.status(404).json({ error: 'User not found' });
//     }
//   } catch (err) {
//     res.status(500).json({ error: 'Logout failed' });
//   }
// });
// app.post('/logout', async (req, res) => {
//   const { email } = req.body;
//   console.log("Logout request for email:", email); // Add this line
//   // ...rest of your code...
// });
// // ...existing code...



// // ...existing code...

// // ...existing code...

// app.listen(5000, () => console.log('Server running on port 5000'));
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
require('dotenv').config();

const fs = require('fs');
const path = require('path');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;

const User = require('./models/User');

const app = express();
app.use(cors());
app.use(express.json());

// MongoDB connection
mongoose.connect(process.env.MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('MongoDB connected'))
  .catch(err => console.log(err));

// CSV Writer setup
const csvPath = path.join(__dirname, 'academic_stress_answers.csv');
const csvWriter = createCsvWriter({
  path: csvPath,
  header: [
    {id: 'q1', title: 'Q1'},
    {id: 'q2', title: 'Q2'},
    {id: 'q3', title: 'Q3'},
    {id: 'q4', title: 'Q4'},
    {id: 'q5', title: 'Q5'},
    {id: 'q6', title: 'Q6'},
    {id: 'timestamp', title: 'Timestamp'}
  ],
  append: true
});

// Signup route
app.post('/signup', async (req, res) => {
  const { username, email, password } = req.body;
  try {
    const user = new User({ username, email, password });
    await user.save();
    res.status(201).json({ message: 'User registered!' });
  } catch (err) {
    res.status(500).json({ error: 'Signup failed' });
  }
});

// Login route
app.post('/login', async (req, res) => {
  const { email, password } = req.body;
  try {
    const user = await User.findOne({ email, password });
    if (user) {
      res.json({ message: 'Login successful' });
    } else {
      res.status(401).json({ error: 'Incorrect email or password' });
    }
  } catch (err) {
    res.status(500).json({ error: 'Login failed' });
  }
});

// Logout route
app.post('/logout', async (req, res) => {
  const { email } = req.body;
  try {
    const result = await User.deleteOne({ email });
    if (result.deletedCount === 1) {
      res.json({ message: 'User deleted and logged out' });
    } else {
      res.status(404).json({ error: 'User not found' });
    }
  } catch (err) {
    res.status(500).json({ error: 'Logout failed' });
  }
});

// Academic Stress Answers - Save to CSV
app.post('/submit-academic-stress', async (req, res) => {
  const { answers } = req.body; // answers should be an array of selected options
  if (!answers || answers.length !== 6) {
    return res.status(400).json({ error: 'Invalid answers' });
  }
  try {
    await csvWriter.writeRecords([{
      q1: answers[0],
      q2: answers[1],
      q3: answers[2],
      q4: answers[3],
      q5: answers[4],
      q6: answers[5],
      timestamp: new Date().toISOString()
    }]);
    res.json({ message: 'Answers saved to CSV!' });
  } catch (err) {
    res.status(500).json({ error: 'Failed to save answers' });
  }
});

app.listen(5000, () => console.log('Server running on port 5000'));