# Diabetic Retinopathy AI WebApp Engineering Requirements

https://physionet.org/content/mbrset/1.0/

0. SQL Database
1. Patient Profile Builder
    - Name
    - Age
    - Sex 
    - Obesity (Weight / Height)
    - Insulin
    - Smoking
    - Alcohol
    - Vascular Disease

2. **Upload Retinal Photos (2x)**
    - Model Runs 1 time: 
        - Batched Eye Inference (Batch / Both Eyes [2], Height [224], Width [224], Channels [3])
    - Model Runs 2 times: Separate Eye Inference
3. **Inference (Ideally AI Inference FastAPI, )**

4. JSON file of Retinopathy Diagnosis, including Metadata
5. PDF Report Generation
    - Procedural Template
    - \+ LLM 

5. Update SQL Database (CRUD) 
    - CREATE Profile
    - READ /VIEW Profile
    - EDIT / UPDATE Profile
    - DELETE Profile

Demo: 
1. Fill out profile builder, Upload Images, Process, Generates Json -> Report.
2. Same Profile Next / Different Visit (AutoFill Information with Different Date)
3. New Profile or **Look at 5 other Existing Profile**


Github
- Single Repo which has all of the code for many components 

- -*>
1. WebApp Repo (UI)
    - AI Core (Addon) 
        - Separate Github and git clone
        - Submodule (By itself it looks normal) but to the WebApp it's like depency

git clone qsight (only WebApp) 

git clone qsight --include-submodules (Include every Dependent Repo)

1. Download a Desktop Application
2. Install it (include Addons)
3. Install Feature (git clone)
4. X-Ray Scans | Diabetic Retinopathy 

