# Empathy Map: ICU Clinician / Intensivist

**Target Persona:** Dr. Sarah, a senior ICU attending physician managing 15 critically ill patients simultaneously. She is overwhelmed by data streams and alarm fatigue.

---

### SAYS
* What the user explicitly states out loud.

- "I need to know which of these patients is going to crash in the next 12 hours so I can intervene."
- "I don't trust this AI score. Why is it telling me the patient has a high risk of Sepsis? Show me the data."
- "Turn off that alarm! It's been beeping all morning for a patient who is completely stable."
- "I don't have time to review 48 hours of tabular lab results for every single patient on the ward."

---

### THINKS
* The user's internal monologue, doubts, and core motivations.

- "If I discharge this patient to the step-down unit, will they develop Acute Kidney Injury (AKI) tonight?"
- "Is this machine learning model factoring in the patient's underlying chronic heart failure (CHF), or is it just looking at today's heart rate?"
- "A false negative means a patient dies. A false positive means I waste nursing resources and suffer from alarm fatigue. I need high precision."
- "I wish the system could just connect the dots between the patient's baseline metabolic risks and their acute infection markers."

---

### DOES
* Actions and behaviors the user performs in their daily workflow.

- Rapidly triages patients during morning rounds by scanning disparate EHR screens (vitals, labs, demographics).
- Mentally calculates comorbidity interactions (e.g., "This patient has Diabetes, so their threshold for kidney failure during this infection is lower").
- Ignores automated clinical decision support alerts that have historically had high false-positive rates.
- Administers broad-spectrum antibiotics or adjusts vasopressors based on gut instinct and fragmented data before lab cultures return.

---

### FEELS
* The user's emotional state, frustrations, and desires.

- **Overwhelmed:** Too much fragmented data coming from monitors, labs, and nursing notes.
- **Frustrated:** Annoyed by "black box" algorithms that give a risk score without explaining *why* the score is high.
- **Stressed:** Bears the ultimate medical and legal responsibility for life-and-death decisions under severe time constraints.
- **Relieved:** When a system accurately flags a subtle deterioration trend and provides the specific physiological features driving the alert (explainability), allowing them to act confidently.
