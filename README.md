# AI Copilot for Senior Citizens - AWS Integration

This project integrates Claude 3.5 Sonnet (via AWS Bedrock) with a React frontend and FastAPI backend for medical information extraction and management.

## 🚀 Features

- **AI-Powered Medical Information Extraction**: Uses Claude 3.5 Sonnet to extract structured medical data
- **Multi-Modal Input**: Text input, file upload, and voice recording
- **Patient Management**: Complete patient profile management system
- **Medical Records**: Upload and manage medical documents
- **Medication Tracking**: Track medications, dosages, and schedules
- **Appointment Management**: Schedule and manage appointments
- **Dashboard**: Comprehensive dashboard for doctors, caregivers, and patients

## 🏗️ Architecture

- **Frontend**: React with TypeScript
- **Backend**: FastAPI with Python
- **Database**: MongoDB
- **AI**: AWS Bedrock (Claude 3.5 Sonnet)
- **File Storage**: GridFS (MongoDB)

## 📋 Prerequisites

1. **Python 3.8+**
2. **Node.js 16+**
3. **MongoDB** (local or cloud)
4. **AWS Account** with Bedrock access
5. **AWS Credentials** configured

## 🔧 Setup Instructions

### 1. AWS Credentials Setup

```bash
# Configure AWS credentials
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### 2. Backend Setup

```bash
cd aws/backend

# Install dependencies
pip install -r requirements.txt

# Set MongoDB URL (optional, defaults to localhost)
export MONGO_URL=mongodb://localhost:27017/

# Start the backend server
python start_server.py
```

The backend will run on `http://localhost:8000`

### 3. Frontend Setup

```bash
cd aws/frontend

# Install dependencies
npm install

# Start the development server
npm start
```

The frontend will run on `http://localhost:3000`

## 🎯 How It Works

### AI Integration Flow

1. **User Input**: Patient provides information via text, file upload, or voice
2. **Backend Processing**: FastAPI receives the data and sends it to Claude 3.5 Sonnet
3. **AI Extraction**: Claude extracts structured medical information (name, age, diagnosis, medications, allergies, notes)
4. **Database Storage**: Extracted information is stored in MongoDB
5. **Frontend Display**: React displays the extracted information in a user-friendly format

### API Endpoints

- `POST /patient/{patient_id}/onboarding` - Process onboarding with AI extraction
- `POST /ai/analyze` - Analyze medical text with Claude
- `GET /patient/{patient_id}` - Get patient profile
- `POST /medicines/{patient_id}` - Add medication
- `POST /appointments/{patient_id}` - Add appointment
- `POST /patient/{patient_id}/reports` - Upload medical reports

## 🔑 Demo Credentials

The system comes with pre-seeded demo accounts:

### Doctors
- Username: `drsmith`, Password: `password123`
- Username: `drlee`, Password: `password123`

### Caregivers
- Username: `caregiver1`, Password: `password123`

### Patients
- Username: `patjane`, Password: `password123`
- Username: `patlee`, Password: `password123`

## 🎤 Voice Integration (Ready for NLX)

The system is prepared for NLX voice integration:

1. **Frontend**: Voice recording interface is already implemented
2. **Backend**: Ready to receive audio files and send to NLX API
3. **Integration Point**: Update the `transcribeAudio` function in the frontend

### To Add NLX Integration:

1. Get NLX API credentials
2. Update the `transcribeAudio` function in `aws/frontend/src/App.tsx`
3. Add NLX API endpoint to the backend
4. Test with real audio files

## 🛠️ Development

### Backend Development

```bash
cd aws/backend
python start_server.py
```

### Frontend Development

```bash
cd aws/frontend
npm start
```

### Database Management

The system uses MongoDB with the following collections:
- `users` - User accounts and authentication
- `patients` - Patient profiles and medical data
- `medicines` - Medication tracking
- `appointments` - Appointment scheduling
- `logs` - Activity logs

## 🔍 Troubleshooting

### Common Issues

1. **AWS Credentials Error**: Ensure AWS credentials are properly configured
2. **MongoDB Connection**: Check if MongoDB is running and accessible
3. **CORS Issues**: Backend is configured for `http://localhost:3000`
4. **Port Conflicts**: Ensure ports 3000 (frontend) and 8000 (backend) are available

### Debug Mode

```bash
# Backend with debug logging
cd aws/backend
uvicorn api:app --reload --log-level debug

# Frontend with debug logging
cd aws/frontend
REACT_APP_DEBUG=true npm start
```

## 📝 Environment Variables

Create a `.env` file in the backend directory:

```env
MONGO_URL=mongodb://localhost:27017/
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and hackathon purposes. 