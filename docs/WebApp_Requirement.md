# Diabetic Retinopathy AI WebApp Engineering Requirements

https://physionet.org/content/mbrset/1.0/

# Bryan
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


# Jana
4. JSON file of Retinopathy Diagnosis, including Metadata
    - Patient Info
    - Image Info
    - Model Info
    - Diagnosis Info
    - LLM Narrative Summary
5. PDF Report Generation (Python Automation Accounting / Audit etc for LLM Narrative Summary)
    - Procedural Template
    - \+ -> Free LLM API to provide Dynamic Summary  
    - Update SQL Database with Report Link
6. Update SQL Database (CRUD) 
    - CREATE Profile
    - READ /VIEW Profile
    - EDIT / UPDATE Profile
    - DELETE Profile

# Demo: 
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


- AI Core Repo (Model Inference, Dataset, Training, LLM Report Generation)
    - Separate Repo
    - Model Inference (FastAPI)
    - Dataset + Training Code
    - LLM Report Generation Code
        - Prompt Engineering
        - LLM Integration (OpenAI, Azure, etc)
        - Report Template


Looked for on Job Resumes for Data Science / ML Engineering:
- Web Application Development (e.g., Flask, Django, FastAPI)
- -> Database Management (SQL, NoSQL)
- Model Deployment (e.g., Docker, Kubernetes)
- -> API Development and Integration
- Frontend Development (e.g., React, Vue.js)
- Cloud Services (AWS, GCP, Azure)
- Version Control (Git, GitHub)
- CI/CD Pipelines (Jenkins, GitHub Actions)
- LLM Integration (OpenAI, Azure OpenAI, etc)
- PDF Report Generation (e.g., ReportLab, FPDF)
- Image Processing (e.g., OpenCV, PIL)
- Data Handling and Preprocessing (Pandas, NumPy)
- Testing and Debugging (Unit Tests, Integration Tests)
- Security Best Practices (Authentication, Authorization)
- Performance Optimization (Caching, Load Balancing)
- Containerization (Docker)
- Monitoring and Logging (Prometheus, ELK Stack)
- Collaboration Tools (Jira, Trello, Slack)
- Agile Methodologies (Scrum, Kanban)

Skills:
- Proficient in Python and JavaScript for web application development.
- Experience with FastAPI for building scalable APIs.
- Strong knowledge of SQL databases for managing patient profiles.
- Familiarity with Docker for containerizing applications.
- Experience with LLMs for generating dynamic reports.
- Skilled in image processing techniques for retinal photo analysis.
- Ability to generate PDF reports programmatically.
- Experience with Git and GitHub for version control and collaboration.
