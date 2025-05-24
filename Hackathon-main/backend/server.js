require('dotenv').config(); // Load environment variables from .env file
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises; // Using promises for cleaner async operations
const { GoogleGenerativeAI } = require("@google/generative-ai");
const axios = require('axios');



const app = express();
const port = process.env.PORT || 5000;

app.use(express.json()); // To parse JSON request bodies

// Update CORS configuration to allow requests from the frontend
const allowedOrigins = ['http://localhost:5173', 'http://your-frontend-domain.com'];
app.use(cors({
    origin: (origin, callback) => {
        if (!origin || allowedOrigins.includes(origin)) {
            callback(null, true);
        } else {
            callback(new Error('Not allowed by CORS'));
        }
    },
}));

// Add a health check endpoint
app.get('/api/health', (req, res) => {
    res.status(200).json({ message: 'Server is healthy!' });
});

// Configure Gemini API
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.geminiProVision;

// Configure multer for file uploads to local storage
const storage = multer.diskStorage({
    destination: async (req, file, cb) => {
        const uploadDir = path.join(__dirname, 'uploads');
        try {
            await fs.mkdir(uploadDir, { recursive: true }); // Create directory if it doesn't exist
            cb(null, uploadDir);
        } catch (err) {
            console.error('Error creating upload directory:', err);
            cb(err, null);
        }
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        const fileExtension = path.extname(file.originalname);
        cb(null, file.fieldname + '-' + uniqueSuffix + fileExtension);
    },
});

const upload = multer({ storage: storage });

// API endpoint to handle daily log submission with Gemini Insights
app.post('/api/daily-log', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ message: 'Please upload an image.' });
        }

        const { productsUsed } = req.body;
        const parsedProducts = JSON.parse(productsUsed); // Products are sent as a string

        const imageUrl = `/uploads/${req.file.filename}`; // Construct URL for local access
        const timestamp = new Date().toISOString(); // Use ISO string for consistency

        console.log('Image saved locally:', req.file.path);
        console.log('Products used:', parsedProducts);
        console.log('Timestamp:', timestamp);
        console.log('Image URL:', imageUrl);

        let aiInsight = null;
        try {
            const prompt = `Analyze the user's selfie. Based on visual cues, provide a brief (1-2 sentences) insight into their potential mood or general well-being. Consider aspects like facial expression, skin appearance (e.g., tiredness, radiance), and anything else visually apparent. If the user mentioned using specific products (${parsedProducts.join(', ')}), also briefly consider how these products might relate to the observed visual cues. the output should be strictly in the json format.`;

            const imagePart = {
                inlineData: {
                    data: req.file.buffer.toString('base64'),
                    mimeType: req.file.mimetype,
                },
            };

            const result = await model.generateContent([prompt, imagePart]);
            console.log('Gemini API Raw Result:', JSON.stringify(result, null, 2)); // Log the entire result object
            const responseText = result.response?.candidates?.[0]?.content?.parts?.[0]?.text;
            aiInsight = responseText || "Could not generate AI insight.";
            console.log('Gemini AI Insight Text:', aiInsight);
        } catch (geminiError) {
            console.error('Error generating AI insights:', geminiError);
            aiInsight = "Error generating AI insight.";
        }

        const JSONDATA = {
            timestamp: timestamp,
            imageUrl: imageUrl,
            productsUsed: parsedProducts,
            aiInsight: aiInsight,
            // You can add more fields here if needed
        };

        console.log('Incoming Data as JSON:', JSON.stringify(jsonData, null, 2));

        // TODO: Save jsonData to your database instead of individual variables

        res.status(200).json({ message: 'Daily log saved successfully!', data: jsonData });

    } catch (error) {
        console.error('Error saving daily log:', error);
        res.status(500).json({ message: 'Failed to save daily log.' });
    }
});

// Serve static files from the 'uploads' directory
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Endpoint for Dashboard data
app.get('/api/dashboard', (req, res) => {
    res.status(200).json({ message: 'Dashboard data fetched successfully!' });
});

// Endpoint for Mood Tracker data
app.post('/api/mood-tracker', (req, res) => {
    const { mood } = req.body;
    res.status(200).json({ message: `Mood '${mood}' recorded successfully!` });
});

// Endpoint for Voice Assistant
app.post('/api/voice-assistant', (req, res) => {
    const { command } = req.body;
    res.status(200).json({ message: `Voice command '${command}' processed successfully!` });
});

// Endpoint for History data
app.get('/api/history', (req, res) => {
    res.status(200).json({ message: 'History data fetched successfully!' });
});

// Endpoint for Risk-Based Alert System
app.get('/api/risk-alerts', (req, res) => {
    res.status(200).json({ message: 'Risk alerts fetched successfully!' });
});

app.listen(port, () => {
    console.log(`Server is running on port: ${port}`);
});