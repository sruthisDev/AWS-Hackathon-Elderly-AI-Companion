from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
import gridfs
import os
from datetime import datetime, timedelta
from typing import Optional
import boto3
import json
import time
import io
import base64
import requests
from google.cloud import speech
from google.cloud import storage

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup
client = MongoClient(os.getenv('MONGO_URL', 'mongodb://localhost:27017/'))
db = client['caremate']
users_col = db['users']
patients_col = db['patients']
medicines_col = db['medicines']
appointments_col = db['appointments']
logs_col = db['logs']
onboarding_col = db['onboarding_data']  # New collection for onboarding data
fs = gridfs.GridFS(db)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# AWS Bedrock setup
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

# NLX Configuration - Removed for new repository
NLX_ENABLED = False  # NLX integration disabled

def schedule_nlx_appointment(appointment_data):
    """Schedule appointment using NLX API - Disabled"""
    return {"success": False, "error": "NLX API integration disabled"}

def call_bedrock_with_retry(request_body, max_retries=3, base_delay=1):
    """Call Bedrock API with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            response = bedrock.invoke_model(
                modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
                body=json.dumps(request_body)
            )
            return response
        except Exception as e:
            if 'ThrottlingException' in str(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                print(f"Throttling detected, waiting {delay} seconds before retry {attempt + 1}")
                time.sleep(delay)
                continue
            else:
                raise e

# Google Speech-to-Text setup
try:
    speech_client = speech.SpeechClient()
    GOOGLE_SPEECH_AVAILABLE = True
except Exception as e:
    print(f"Google Speech-to-Text not available: {e}")
    speech_client = None
    GOOGLE_SPEECH_AVAILABLE = False

# Pydantic models
class LoginRequest(BaseModel):
    username: str
    password: str

class SignupRequest(BaseModel):
    username: str
    password: str
    name: str
    email: str

class PatientProfileUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    notes: Optional[str] = None
    doctorId: Optional[str] = None
    caregiverId: Optional[str] = None

class OnboardingData(BaseModel):
    textData: Optional[str] = None
    summary: Optional[str] = None

class MedicineModel(BaseModel):
    name: str
    dosage: str
    frequency: str
    time: str
    notes: Optional[str] = None

class AppointmentModel(BaseModel):
    title: str
    date: str
    time: str
    doctor: str
    notes: Optional[str] = None

class VoiceAgentRequest(BaseModel):
    patient_id: str
    message: str
    history: Optional[list] = []

class AIAnalysisRequest(BaseModel):
    text: str

class AIAnalysisResponse(BaseModel):
    extracted_info: dict
    original_text: str

class VoiceTranscriptionRequest(BaseModel):
    audio_data: str  # Base64 encoded audio data
    language_code: str = "en-US"

class OnboardingDataStorage(BaseModel):
    patient_id: str
    raw_text_data: str
    ai_extracted_info: dict
    voice_transcription: Optional[str] = None
    file_uploads: Optional[list] = None
    timestamp: str
    processing_status: str

# Helper functions
def fix_id(doc):
    if not doc: return doc
    doc = dict(doc)
    if '_id' in doc: doc['_id'] = str(doc['_id'])
    return doc

def fix_ids(docs):
    return [fix_id(doc) for doc in docs]

# AI extraction function using Claude 3.5 Sonnet
def extract_medical_info(text):
    if not text.strip():
        return {"error": "Please provide some text to analyze."}

    # Claude 3.5 Sonnet requires the Messages API format
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Extract the main medical information from the following text and format it as JSON with these fields: "
                    "name, age, diagnosis, medications, allergies, notes. If any field is not found, use null.\n\n"
                    f"Text: {text}\n\n"
                    "Response (JSON only):"
                )
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.1
    })

    try:
        response = call_bedrock_with_retry(json.loads(body))
        result = json.loads(response['body'].read())
        # For Claude 3.5 Sonnet Messages API, extract the text from the first content part
        if 'content' in result and isinstance(result['content'], list) and result['content']:
            output_text = result['content'][0].get('text', str(result))
        else:
            output_text = str(result)
        
        # Try to parse the JSON response
        try:
            # Clean up the response to extract JSON
            if output_text.strip().startswith('{') and output_text.strip().endswith('}'):
                extracted_json = json.loads(output_text)
            else:
                # Try to find JSON in the response
                start_idx = output_text.find('{')
                end_idx = output_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = output_text[start_idx:end_idx]
                    extracted_json = json.loads(json_str)
                else:
                    extracted_json = {"raw_response": output_text}
            
            return extracted_json
        except json.JSONDecodeError:
            return {"raw_response": output_text, "error": "Could not parse JSON response"}
            
    except Exception as e:
        return {"error": f"Error calling Bedrock API: {str(e)}"}

# Simple voice transcription function (placeholder for browser-based speech recognition)
def transcribe_audio(audio_data: str, language_code: str = "en-US"):
    # This is a placeholder - actual transcription will be done in the browser
    # using the Web Speech API
    return {
        "transcription": "Voice transcription will be handled by browser's Web Speech API",
        "confidence": 0.0,
        "success": True,
        "note": "Using browser's built-in speech recognition"
    }

# Voice Agent Processing function
def process_voice_agent_message(message: str, patient_id: str, history: list = []):
    """Process voice agent messages and extract actions"""
    try:
        # Create a comprehensive prompt for voice agent
        prompt = f"""
        You are a helpful AI voice assistant for a senior citizen's health management system. 
        The user is speaking to you to add medicine reminders or doctor appointments.
        
        Patient ID: {patient_id}
        User Message: "{message}"
        
        Your task is to:
        1. Understand the user's intent
        2. Extract relevant information (medicine name, dosage, time, date, doctor name, etc.)
        3. Determine what action to take (add medicine reminder or appointment)
        4. Provide a helpful response
        
        IMPORTANT: For dates, use YYYY-MM-DD format. For times, use HH:MM AM/PM format.
        
        Return your response in this JSON format:
        {{
            "response": "Your helpful response to the user",
            "action": "add_medicine" or "add_appointment" or "none",
            "extracted_data": {{
                "medicine_name": "name if medicine",
                "dosage": "dosage if medicine",
                "frequency": "frequency if medicine (e.g., Once daily, Twice daily, Every morning)",
                "time": "time in HH:MM AM/PM format",
                "date": "date in YYYY-MM-DD format",
                "doctor_name": "doctor name if appointment",
                "appointment_title": "title if appointment",
                "notes": "any additional notes"
            }},
            "confidence": 0.0-1.0
        }}
        
        Examples:
        - "Add a medicine reminder for Metformin at 8 AM tomorrow" → add_medicine with date=tomorrow, time=08:00 AM
        - "Schedule a doctor appointment with Dr. Smith on Friday at 2 PM" → add_appointment with date=friday, time=02:00 PM
        - "Remind me to take my blood pressure medication every morning" → add_medicine with frequency=Every morning, time=08:00 AM
        - "Book an appointment with Dr. Johnson next Tuesday at 10 AM" → add_appointment with date=tuesday, time=10:00 AM
        - "Hello" → none
        
        IMPORTANT: For relative dates like "tomorrow", "friday", "tuesday", use the exact word. For times, use HH:MM format.
        
        Only return valid JSON, no additional text.
        """
        
        # Use Claude 3.5 Sonnet for voice agent processing
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.1
        })
        
        response = call_bedrock_with_retry(json.loads(body))
        
        result = json.loads(response['body'].read())
        
        # Extract the response text
        if 'content' in result and isinstance(result['content'], list) and result['content']:
            output_text = result['content'][0].get('text', str(result))
        else:
            output_text = str(result)
        
        # Try to parse the JSON response
        try:
            if output_text.strip().startswith('{') and output_text.strip().endswith('}'):
                parsed_result = json.loads(output_text)
            else:
                # Try to find JSON in the response
                start_idx = output_text.find('{')
                end_idx = output_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = output_text[start_idx:end_idx]
                    parsed_result = json.loads(json_str)
                else:
                    parsed_result = {"response": "I'm sorry, I didn't understand that. Could you please try again?", "action": "none", "extracted_data": {}, "confidence": 0.0}
            
            return parsed_result
        except json.JSONDecodeError:
            return {"response": "I'm sorry, I didn't understand that. Could you please try again?", "action": "none", "extracted_data": {}, "confidence": 0.0}
            
    except Exception as e:
        return {"response": f"Sorry, I encountered an error: {str(e)}", "action": "none", "extracted_data": {}, "confidence": 0.0}

# AI Medicine Recommendation function
def recommend_medicines(patient_data: dict):
    """Use AI to recommend medicines based on patient data"""
    try:
        # Create a comprehensive prompt for medicine recommendation
        prompt = f"""
        Based on the following patient information, recommend appropriate medications with dosages and schedules.
        
        Patient Information:
        - Name: {patient_data.get('name', 'Unknown')}
        - Age: {patient_data.get('age', 'Unknown')}
        - Diagnosis: {patient_data.get('diagnosis', 'None provided')}
        - Current Medications: {patient_data.get('medications', 'None')}
        - Allergies: {patient_data.get('allergies', 'None')}
        - Notes: {patient_data.get('notes', 'None')}
        
        Please provide recommendations in the following JSON format:
        {{
            "recommended_medicines": [
                {{
                    "name": "Medicine Name",
                    "dosage": "Dosage (e.g., 10mg)",
                    "frequency": "Frequency (e.g., Twice daily)",
                    "time": "Best time to take (e.g., Morning, Evening)",
                    "notes": "Important notes about the medicine",
                    "reason": "Why this medicine is recommended"
                }}
            ],
            "general_recommendations": "General health recommendations",
            "warnings": "Any warnings or precautions",
            "follow_up": "When to follow up with doctor"
        }}
        
        Only return valid JSON, no additional text.
        """
        
        # Use Claude 3.5 Sonnet for medicine recommendation
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.1
        })
        
        response = call_bedrock_with_retry(json.loads(body))
        
        result = json.loads(response['body'].read())
        
        # Extract the response text
        if 'content' in result and isinstance(result['content'], list) and result['content']:
            output_text = result['content'][0].get('text', str(result))
        else:
            output_text = str(result)
        
        # Try to parse the JSON response
        try:
            if output_text.strip().startswith('{') and output_text.strip().endswith('}'):
                recommendations = json.loads(output_text)
            else:
                # Try to find JSON in the response
                start_idx = output_text.find('{')
                end_idx = output_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = output_text[start_idx:end_idx]
                    recommendations = json.loads(json_str)
                else:
                    recommendations = {"error": "Could not parse AI response", "raw_response": output_text}
            
            return recommendations
        except json.JSONDecodeError:
            return {"error": "Could not parse JSON response", "raw_response": output_text}
            
    except Exception as e:
        return {"error": f"Error getting medicine recommendations: {str(e)}"}

# Seed data
def seed_data():
    if users_col.count_documents({}) == 0:
        users_col.insert_many([
            {"_id": "doctor1", "username": "drsmith", "password": "password123", "role": "doctor", "name": "Dr. Smith", "profileComplete": True},
            {"_id": "doctor2", "username": "drlee", "password": "password123", "role": "doctor", "name": "Dr. Lee", "profileComplete": True},
            {"_id": "caregiver1", "username": "caregiver1", "password": "password123", "role": "caregiver", "name": "John Caregiver", "profileComplete": True},
            {"_id": "patient1", "username": "patjane", "password": "password123", "role": "patient", "name": "Jane Patient", "doctorId": "doctor1", "caregiverId": "caregiver1", "profileComplete": True},
            {"_id": "patient2", "username": "patlee", "password": "password123", "role": "patient", "name": "Lee Patient", "doctorId": "doctor2", "caregiverId": "caregiver1", "profileComplete": True},
        ])
    
    if patients_col.count_documents({}) == 0:
        patients_col.insert_many([
            {"_id": "patient1", "name": "Jane Patient", "age": 65, "address": "123 Main St", "phone": "555-0101", "notes": "Hypertension", "doctorId": "doctor1", "caregiverId": "caregiver1"},
            {"_id": "patient2", "name": "Lee Patient", "age": 72, "address": "456 Oak Ave", "phone": "555-0102", "notes": "Diabetes", "doctorId": "doctor2", "caregiverId": "caregiver1"},
        ])
    
    if medicines_col.count_documents({}) == 0:
        medicines_col.insert_many([
            {"_id": "med1", "patientId": "patient1", "name": "Lisinopril", "dosage": "10mg", "frequency": "Daily", "time": "Morning", "notes": "For blood pressure"},
            {"_id": "med2", "patientId": "patient1", "name": "Metformin", "dosage": "500mg", "frequency": "Twice daily", "time": "Morning, Evening", "notes": "For diabetes"},
            {"_id": "med3", "patientId": "patient2", "name": "Insulin", "dosage": "20 units", "frequency": "Daily", "time": "Evening", "notes": "Type 2 diabetes"},
        ])
    
    if appointments_col.count_documents({}) == 0:
        # Get current date and create appointments for the next few weeks
        today = datetime.now()
        appointments_col.insert_many([
            {"_id": "app1", "patientId": "patient1", "title": "Checkup", "date": (today + timedelta(days=2)).strftime("%Y-%m-%d"), "time": "10:00 AM", "doctor": "Dr. Smith", "notes": "Regular checkup"},
            {"_id": "app2", "patientId": "patient1", "title": "Cardiology", "date": (today + timedelta(days=9)).strftime("%Y-%m-%d"), "time": "2:00 PM", "doctor": "Dr. Lee", "notes": "Heart checkup"},
            {"_id": "app3", "patientId": "patient1", "title": "Blood Test", "date": (today + timedelta(days=16)).strftime("%Y-%m-%d"), "time": "9:00 AM", "doctor": "Dr. Smith", "notes": "Routine blood work"},
            {"_id": "app4", "patientId": "patient2", "title": "Diabetes Review", "date": (today + timedelta(days=5)).strftime("%Y-%m-%d"), "time": "11:00 AM", "doctor": "Dr. Lee", "notes": "Diabetes management"},
            {"_id": "app5", "patientId": "patient2", "title": "Eye Exam", "date": (today + timedelta(days=12)).strftime("%Y-%m-%d"), "time": "3:00 PM", "doctor": "Dr. Lee", "notes": "Annual eye checkup"},
        ])
    
    if logs_col.count_documents({}) == 0:
        logs_col.insert_many([
            {"_id": "log1", "patientId": "patient1", "type": "missed_medicine", "medicine": "Lisinopril", "date": (today - timedelta(days=1)).strftime("%Y-%m-%d"), "time": "Morning"},
            {"_id": "log2", "patientId": "patient1", "type": "missed_appointment", "appointment": "Checkup", "date": (today - timedelta(days=3)).strftime("%Y-%m-%d"), "time": "10:00 AM"},
            {"_id": "log3", "patientId": "patient2", "type": "missed_medicine", "medicine": "Insulin", "date": (today - timedelta(days=2)).strftime("%Y-%m-%d"), "time": "Evening"},
        ])

# Seed data on startup
seed_data()

# API endpoints
@app.post("/signup")
def signup(req: SignupRequest):
    # Check if username already exists
    existing_user = users_col.find_one({"username": req.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Create new user with profileComplete: false
    new_user = {
        "_id": f"patient_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "username": req.username,
        "password": req.password,
        "name": req.name,
        "email": req.email,
        "role": "patient",
        "profileComplete": False,
        "createdAt": datetime.now().isoformat()
    }
    
    users_col.insert_one(new_user)
    
    return {
        "_id": new_user["_id"],
        "username": new_user["username"],
        "role": new_user["role"],
        "name": new_user["name"],
        "profileComplete": new_user["profileComplete"]
    }

@app.post("/login")
def login(req: LoginRequest):
    user = users_col.find_one({"username": req.username, "password": req.password})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {
        "_id": user["_id"],
        "username": user["username"],
        "role": user["role"],
        "name": user["name"],
        "profileComplete": user.get("profileComplete", True)  # Default to True for existing users
    }

@app.get("/me/{user_id}")
def get_user(user_id: str):
    user = users_col.find_one({"_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return fix_id(user)

@app.get("/patients/{user_id}")
def get_patients(user_id: str):
    user = users_col.find_one({"_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user["role"] == "patient":
        # Return self as patient
        patient = patients_col.find_one({"_id": user_id})
        return [fix_id(patient)] if patient else []
    else:
        # Return patients assigned to this doctor/caregiver
        field = "doctorId" if user["role"] == "doctor" else "caregiverId"
        patients = patients_col.find({field: user_id})
        return fix_ids(patients)

@app.get("/patient/{patient_id}")
def get_patient(patient_id: str):
    patient = patients_col.find_one({"_id": patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return fix_id(patient)

@app.patch("/patient/{patient_id}")
def update_patient(patient_id: str, update: PatientProfileUpdate):
    update_data = {k: v for k, v in update.dict().items() if v is not None}
    
    if update_data:
        result = patients_col.update_one({"_id": patient_id}, {"$set": update_data})
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Patient not found")
    
    # Mark user as profile complete
    users_col.update_one({"_id": patient_id}, {"$set": {"profileComplete": True}})
    
    return {"message": "Patient updated successfully"}

@app.post("/patient/{patient_id}/onboarding")
def complete_onboarding(patient_id: str, data: OnboardingData):
    """Complete onboarding and store data as JSON in MongoDB"""
    
    # Process the onboarding data and create/update patient profile
    summary = data.summary or data.textData or "Patient information provided during onboarding"
    
    # Use AI to extract medical information if text data is provided
    extracted_info = {}
    if data.textData and data.textData.strip():
        extracted_info = extract_medical_info(data.textData)
    
    # Create comprehensive onboarding data structure for JSON storage
    onboarding_data = {
        "_id": f"onboarding_{patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "patient_id": patient_id,
        "raw_text_data": data.textData or "",
        "ai_extracted_info": extracted_info,
        "voice_transcription": None,  # Will be updated if voice data is available
        "file_uploads": [],
        "timestamp": datetime.now().isoformat(),
        "processing_status": "completed",
        "summary": summary,
        "metadata": {
            "source": "onboarding_form",
            "ai_model": "claude-3-5-sonnet",
            "extraction_method": "ai_analysis"
        }
    }
    
    # Store onboarding data as JSON in MongoDB
    onboarding_col.insert_one(onboarding_data)
    
    # Create or update patient profile with AI-extracted information
    patient_data = {
        "_id": patient_id,
        "name": "New Patient",
        "age": None,
        "address": "",
        "phone": "",
        "notes": summary,
        "doctorId": None,
        "caregiverId": None,
        "ai_extracted_info": extracted_info,
        "onboarding_data_id": onboarding_data["_id"]  # Reference to stored onboarding data
    }
    
    # Update patient data with AI-extracted information if available
    if extracted_info and not extracted_info.get("error"):
        if extracted_info.get("name"):
            patient_data["name"] = extracted_info["name"]
        if extracted_info.get("age"):
            try:
                patient_data["age"] = int(extracted_info["age"])
            except (ValueError, TypeError):
                pass
        if extracted_info.get("diagnosis"):
            patient_data["diagnosis"] = extracted_info["diagnosis"]
        if extracted_info.get("medications"):
            patient_data["medications"] = extracted_info["medications"]
        if extracted_info.get("allergies"):
            patient_data["allergies"] = extracted_info["allergies"]
        if extracted_info.get("notes"):
            patient_data["notes"] = extracted_info["notes"]
    
    # Fallback to simple parsing if AI extraction failed
    if data.textData and (not extracted_info or extracted_info.get("error")):
        lines = data.textData.split('\n')
        for line in lines:
            line = line.strip().lower()
            if 'name:' in line:
                patient_data["name"] = line.split('name:')[-1].strip()
            elif 'age:' in line:
                try:
                    patient_data["age"] = int(line.split('age:')[-1].strip())
                except:
                    pass
            elif 'phone:' in line or 'contact:' in line:
                patient_data["phone"] = line.split(':')[-1].strip()
            elif 'address:' in line:
                patient_data["address"] = line.split('address:')[-1].strip()
    
    # Upsert patient profile
    patients_col.replace_one({"_id": patient_id}, patient_data, upsert=True)
    
    # Mark user as profile complete
    users_col.update_one({"_id": patient_id}, {"$set": {"profileComplete": True}})
    
    return {
        "message": "Onboarding completed successfully",
        "ai_extracted_info": extracted_info,
        "onboarding_id": onboarding_data["_id"],
        "stored_data": onboarding_data
    }

@app.post("/ai/analyze")
def analyze_medical_text(request: AIAnalysisRequest):
    """Analyze medical text using Claude 3.5 Sonnet"""
    extracted_info = extract_medical_info(request.text)
    return AIAnalysisResponse(
        extracted_info=extracted_info,
        original_text=request.text
    )

@app.post("/voice/transcribe")
def transcribe_voice(request: VoiceTranscriptionRequest):
    """Transcribe voice to text using Google Speech-to-Text"""
    result = transcribe_audio(request.audio_data, request.language_code)
    
    if result["success"]:
        # Store transcription in database for logging
        transcription_log = {
            "_id": f"trans_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "transcription": result["transcription"],
            "confidence": result["confidence"],
            "language_code": request.language_code,
            "timestamp": datetime.now().isoformat()
        }
        
        # You can store this in a separate collection if needed
        # db['transcriptions'].insert_one(transcription_log)
        
        return {
            "transcription": result["transcription"],
            "confidence": result["confidence"],
            "success": True
        }
    else:
        raise HTTPException(status_code=400, detail=result["error"])

@app.post("/patient/{patient_id}/reports")
def upload_report(patient_id: str, file: UploadFile = File(...)):
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    # Store file in GridFS
    file_id = fs.put(file.file, filename=file.filename, patientId=patient_id)
    
    return {"file_id": str(file_id), "filename": file.filename}

@app.get("/patient/{patient_id}/reports")
def get_reports(patient_id: str):
    files = fs.find({"patientId": patient_id})
    reports = []
    for file in files:
        reports.append({
            "file_id": str(file._id),
            "filename": file.filename,
            "upload_date": file.upload_date.isoformat()
        })
    return reports

@app.get("/patient/{patient_id}/reports/{file_id}")
def download_report(patient_id: str, file_id: str):
    try:
        file_obj = fs.get(ObjectId(file_id))
        return StreamingResponse(
            iter([file_obj.read()]),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={file_obj.filename}"}
        )
    except:
        raise HTTPException(status_code=404, detail="File not found")

@app.delete("/patient/{patient_id}/reports/{file_id}")
def delete_report(patient_id: str, file_id: str):
    try:
        fs.delete(ObjectId(file_id))
        return {"message": "Report deleted successfully"}
    except:
        raise HTTPException(status_code=404, detail="File not found")

@app.post("/medicines/{patient_id}")
def add_medicine(patient_id: str, medicine: MedicineModel):
    med_data = medicine.dict()
    med_data["patientId"] = patient_id
    med_data["_id"] = f"med_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    medicines_col.insert_one(med_data)
    return fix_id(med_data)

@app.patch("/medicines/{med_id}")
def update_medicine(med_id: str, medicine: MedicineModel):
    update_data = medicine.dict()
    result = medicines_col.update_one({"_id": med_id}, {"$set": update_data})
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Medicine not found")
    
    return {"message": "Medicine updated successfully"}

@app.delete("/medicines/{med_id}")
def delete_medicine(med_id: str):
    result = medicines_col.delete_one({"_id": med_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Medicine not found")
    
    return {"message": "Medicine deleted successfully"}

@app.get("/medicines/{patient_id}")
def get_medicines(patient_id: str):
    medicines = medicines_col.find({"patientId": patient_id})
    return fix_ids(medicines)

@app.post("/appointments/{patient_id}")
def add_appointment(patient_id: str, appointment: AppointmentModel):
    app_data = appointment.dict()
    app_data["patientId"] = patient_id
    app_data["_id"] = f"app_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    appointments_col.insert_one(app_data)
    return fix_id(app_data)

@app.patch("/appointments/{app_id}")
def update_appointment(app_id: str, appointment: AppointmentModel):
    update_data = appointment.dict()
    result = appointments_col.update_one({"_id": app_id}, {"$set": update_data})
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Appointment not found")
    
    return {"message": "Appointment updated successfully"}

@app.delete("/appointments/{app_id}")
def delete_appointment(app_id: str):
    result = appointments_col.delete_one({"_id": app_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Appointment not found")
    
    return {"message": "Appointment deleted successfully"}

@app.get("/appointments/{patient_id}")
def get_appointments(patient_id: str):
    appointments = appointments_col.find({"patientId": patient_id})
    return fix_ids(appointments)

@app.get("/logs/{patient_id}")
def get_logs(patient_id: str):
    logs = logs_col.find({"patientId": patient_id})
    return fix_ids(logs)

@app.get("/onboarding/{patient_id}")
def get_onboarding_data(patient_id: str):
    """Retrieve stored onboarding data as JSON"""
    onboarding_records = onboarding_col.find({"patient_id": patient_id}).sort("timestamp", -1)
    return fix_ids(onboarding_records)

@app.get("/onboarding/{patient_id}/latest")
def get_latest_onboarding_data(patient_id: str):
    """Retrieve the most recent onboarding data"""
    latest_record = onboarding_col.find_one(
        {"patient_id": patient_id}, 
        sort=[("timestamp", -1)]
    )
    return fix_id(latest_record)

@app.post("/patient/{patient_id}/recommend-medicines")
def get_medicine_recommendations(patient_id: str):
    """Get AI-powered medicine recommendations based on patient data"""
    try:
        # Get patient data
        patient = patients_col.find_one({"_id": patient_id})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Get latest onboarding data
        latest_onboarding = onboarding_col.find_one(
            {"patient_id": patient_id}, 
            sort=[("timestamp", -1)]
        )
        
        # Combine patient data with onboarding data
        combined_data = {
            "name": patient.get("name", ""),
            "age": patient.get("age"),
            "diagnosis": patient.get("diagnosis", ""),
            "medications": patient.get("medications", ""),
            "allergies": patient.get("allergies", ""),
            "notes": patient.get("notes", "")
        }
        
        # Add AI-extracted info from onboarding if available
        if latest_onboarding and latest_onboarding.get("ai_extracted_info"):
            ai_info = latest_onboarding["ai_extracted_info"]
            if ai_info.get("diagnosis"):
                combined_data["diagnosis"] = ai_info["diagnosis"]
            if ai_info.get("medications"):
                combined_data["medications"] = ai_info["medications"]
            if ai_info.get("allergies"):
                combined_data["allergies"] = ai_info["allergies"]
        
        # Get AI recommendations
        recommendations = recommend_medicines(combined_data)
        
        # Store recommendations in database
        recommendation_record = {
            "_id": f"rec_{patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "patient_id": patient_id,
            "patient_data": combined_data,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        # Store in a new collection for recommendations
        db['medicine_recommendations'].insert_one(recommendation_record)
        
        return {
            "patient_data": combined_data,
            "recommendations": recommendations,
            "recommendation_id": recommendation_record["_id"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@app.post("/voice-agent/chat")
def voice_agent_chat(request: VoiceAgentRequest):
    """Process voice agent messages and perform actions"""
    try:
        # Process the voice agent message
        result = process_voice_agent_message(request.message, request.patient_id, request.history)
        
        actions_performed = []
        
        # Perform actions based on the result
        if result.get("action") == "add_medicine" and result.get("confidence", 0) > 0.7:
            extracted_data = result.get("extracted_data", {})
            
            # Parse date - handle relative dates
            date_str = extracted_data.get("date", "")
            if not date_str or date_str == "today":
                target_date = datetime.now().strftime('%Y-%m-%d')
            elif date_str == "tomorrow":
                target_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            elif date_str == "next week":
                target_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
            elif "friday" in date_str.lower():
                # Calculate next Friday
                today = datetime.now()
                days_until_friday = (4 - today.weekday()) % 7
                if days_until_friday == 0:  # Today is Friday
                    days_until_friday = 7
                target_date = (today + timedelta(days=days_until_friday)).strftime('%Y-%m-%d')
            elif "tuesday" in date_str.lower():
                # Calculate next Tuesday
                today = datetime.now()
                days_until_tuesday = (1 - today.weekday()) % 7
                if days_until_tuesday == 0:  # Today is Tuesday
                    days_until_tuesday = 7
                target_date = (today + timedelta(days=days_until_tuesday)).strftime('%Y-%m-%d')
            else:
                # Try to parse the date string
                try:
                    target_date = date_str
                except:
                    target_date = datetime.now().strftime('%Y-%m-%d')
            
            # Create medicine record
            medicine_data = {
                "_id": f"med_{request.patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "patientId": request.patient_id,
                "name": extracted_data.get("medicine_name", "Unknown Medicine"),
                "dosage": extracted_data.get("dosage", ""),
                "frequency": extracted_data.get("frequency", "Once daily"),
                "time": extracted_data.get("time", "08:00 AM"),
                "date": target_date,
                "notes": extracted_data.get("notes", "Added via voice assistant")
            }
            
            medicines_col.insert_one(medicine_data)
            actions_performed.append("medicine_added")
            
        elif result.get("action") == "add_appointment" and result.get("confidence", 0) > 0.7:
            extracted_data = result.get("extracted_data", {})
            
            # Parse date - handle relative dates
            date_str = extracted_data.get("date", "")
            if not date_str or date_str == "today":
                target_date = datetime.now().strftime('%Y-%m-%d')
            elif date_str == "tomorrow":
                target_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            elif date_str == "next week":
                target_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
            elif "friday" in date_str.lower():
                # Calculate next Friday
                today = datetime.now()
                days_until_friday = (4 - today.weekday()) % 7
                if days_until_friday == 0:  # Today is Friday
                    days_until_friday = 7
                target_date = (today + timedelta(days=days_until_friday)).strftime('%Y-%m-%d')
            elif "tuesday" in date_str.lower():
                # Calculate next Tuesday
                today = datetime.now()
                days_until_tuesday = (1 - today.weekday()) % 7
                if days_until_tuesday == 0:  # Today is Tuesday
                    days_until_tuesday = 7
                target_date = (today + timedelta(days=days_until_tuesday)).strftime('%Y-%m-%d')
            else:
                # Try to parse the date string
                try:
                    target_date = date_str
                except:
                    target_date = datetime.now().strftime('%Y-%m-%d')
            
            # Create appointment record
            appointment_data = {
                "_id": f"app_{request.patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "patientId": request.patient_id,
                "title": extracted_data.get("appointment_title", "Doctor Appointment"),
                "date": target_date,
                "time": extracted_data.get("time", "09:00 AM"),
                "doctor": extracted_data.get("doctor_name", "Doctor"),
                "notes": extracted_data.get("notes", "Scheduled via voice assistant")
            }
            
            # Schedule appointment locally
            appointment_data["scheduled_with"] = "local_database"
            actions_performed.append("appointment_added")
            
            appointments_col.insert_one(appointment_data)
        
        return {
            "response": result.get("response", "I'm sorry, I didn't understand that."),
            "action": result.get("action", "none"),
            "confidence": result.get("confidence", 0.0),
            "actions_performed": actions_performed
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing voice agent message: {str(e)}")

@app.get("/nlx/status")
def get_nlx_status():
    """Check NLX API status and configuration"""
    return {
        "nlx_enabled": False,
        "message": "NLX API integration has been removed"
    }

@app.get("/nlx/available-slots/{doctor_name}")
def get_available_slots(doctor_name: str, date: str = None):
    """Get available appointment slots for a doctor"""
    raise HTTPException(status_code=400, detail="NLX API integration has been removed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 