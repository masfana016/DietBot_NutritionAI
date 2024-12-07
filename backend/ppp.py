# models.py
from sqlmodel import SQLModel, Field, SQLModel, create_engine, Session, select
from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from datetime import datetime
# main.py
from datetime import datetime

app = FastAPI()


class Appointment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    patient_name: str
    patient_email: str
    appointment_date: datetime
    notes: Optional[str] = None

    def __repr__(self):
        return f"<Appointment(id={self.id}, patient_name={self.patient_name}, appointment_date={self.appointment_date})>"
    

DATABASE_URL = "postgresql://postgres.sfnnnclrbowjaxxhndcd:masfaansari1999@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"  # Use SQLite for simplicity

engine = create_engine(DATABASE_URL, echo=True)

def create_db():
    SQLModel.metadata.create_all(bind=engine)


def book_appointment(patient_name: str, patient_email: str, appointment_date: datetime, notes: Optional[str] = None):
    with Session(engine) as session:
        appointment = Appointment(
            patient_name=patient_name,
            patient_email=patient_email,
            appointment_date=appointment_date,
            notes=notes,
        )
        session.add(appointment)
        session.commit()
        print(f"Appointment booked for {patient_name} on {appointment_date}")


def list_appointments():
    with Session(engine) as session:
        appointments = session.exec(select(Appointment)).all()
        if appointments:
            for appointment in appointments:
                print(appointment)
        else:
            print("No appointments available.")


def get_appointment_by_id(appointment_id: int):
    with Session(engine) as session:
        appointment = session.get(Appointment, appointment_id)
        if appointment:
            print(appointment)
        else:
            print("Appointment not found.")

# Pydantic models for request validation
class AppointmentCreate(BaseModel):
    patient_name: str
    patient_email: str
    appointment_date: datetime
    notes: Optional[str] = None

    class Config:
        orm_mode = True


class AppointmentResponse(AppointmentCreate):
    id: int


@app.post("/appointments/", response_model=AppointmentResponse)
def book_appointment(appointment: AppointmentCreate):
    """Book a new appointment"""
    with Session(engine) as session:
        new_appointment = Appointment(
            patient_name=appointment.patient_name,
            patient_email=appointment.patient_email,
            appointment_date=appointment.appointment_date,
            notes=appointment.notes,
        )
        session.add(new_appointment)
        session.commit()
        session.refresh(new_appointment)
        return new_appointment


@app.get("/appointments/", response_model=List[AppointmentResponse])
def list_appointments():
    """Get all appointments"""
    with Session(engine) as session:
        appointments = session.exec(select(Appointment)).all()
        return appointments


@app.get("/appointments/{appointment_id}", response_model=AppointmentResponse)
def get_appointment_by_id(appointment_id: int):
    """Get an appointment by ID"""
    with Session(engine) as session:
        appointment = session.get(Appointment, appointment_id)
        if appointment:
            return appointment
        else:
            raise HTTPException(status_code=404, detail="Appointment not found")

