$dst = (Get-Location).Path
$src = Join-Path $dst "CS221"

# Create directories
New-Item -ItemType Directory -Force -Path "$dst\data\raw"
New-Item -ItemType Directory -Force -Path "$dst\docs\references"
New-Item -ItemType Directory -Force -Path "$dst\docs\lectures"
New-Item -ItemType Directory -Force -Path "$dst\docs\admin"
New-Item -ItemType Directory -Force -Path "$dst\docs\images"
New-Item -ItemType Directory -Force -Path "$dst\docs\projects"
New-Item -ItemType Directory -Force -Path "$dst\notebooks"
New-Item -ItemType Directory -Force -Path "$dst\scripts"
New-Item -ItemType Directory -Force -Path "$dst\models"
New-Item -ItemType Directory -Force -Path "$dst\logs"

# Move data
Move-Item -Path "$src\ASQE-DPT.zip" -Destination "$dst\data\raw\" -Force
Move-Item -Path "$src\EduRABSA_AOPE.zip" -Destination "$dst\data\raw\" -Force
Move-Item -Path "$src\EduRABSA_ASTE.zip" -Destination "$dst\data\raw\" -Force
Move-Item -Path "$src\EduRABSA_Dataset.zip" -Destination "$dst\data\raw\" -Force
Move-Item -Path "$src\CS221_REPORT_ABSA.zip" -Destination "$dst\data\raw\" -Force
Move-Item -Path "$src\private_test_6QTOAB.zip" -Destination "$dst\data\raw\" -Force

# Directories to move
if (Test-Path "$src\ASQE-DPT") { Move-Item -Path "$src\ASQE-DPT" -Destination "$dst\data\raw\" -Force }
if (Test-Path "$src\EduRABSA_ASTE") { Move-Item -Path "$src\EduRABSA_ASTE" -Destination "$dst\data\raw\" -Force }
if (Test-Path "$src\EduRABSA_Dataset") { Move-Item -Path "$src\EduRABSA_Dataset" -Destination "$dst\data\raw\" -Force }

# Move docs/references
Move-Item -Path "$src\10_POS.pdf" -Destination "$dst\docs\references\" -Force
Move-Item -Path "$src\11_hmm_memm.pdf" -Destination "$dst\docs\references\" -Force
Move-Item -Path "$src\16_06_SemEval_ABSA2016.pdf" -Destination "$dst\docs\references\" -Force
Move-Item -Path "$src\2311.10777v6.pdf" -Destination "$dst\docs\references\" -Force
Move-Item -Path "$src\81e82a93-2c14-415f-9038-10d4b6f87336.pdf" -Destination "$dst\docs\references\" -Force
Move-Item -Path "$src\CS221.pdf" -Destination "$dst\docs\references\" -Force
Move-Item -Path "$src\eduRABSA.pdf" -Destination "$dst\docs\references\" -Force
Move-Item -Path "$src\Imbalanced_Data_Problem_in_Machine_Learning_A_Review.pdf" -Destination "$dst\docs\references\" -Force
Move-Item -Path "$src\PhongBART - Multilabel movie genres classification.pdf" -Destination "$dst\docs\references\" -Force
Move-Item -Path "$src\S16-1002.pdf" -Destination "$dst\docs\references\" -Force
Move-Item -Path "$src\S16-1050.pdf" -Destination "$dst\docs\references\" -Force
Move-Item -Path "$src\DATA-EFFICIENT ADAPTATION AND A NOVEL EVALUATION.pdf" -Destination "$dst\docs\references\" -Force
Move-Item -Path "$src\s10462-024-10906-z.pdf" -Destination "$dst\docs\references\" -Force
Move-Item -Path "$src\s41598-021-90345-w.pdf" -Destination "$dst\docs\references\" -Force

# Move docs/lectures (use wildcards to avoid unicode character issues in script like 'Hàm ẩn danh')
Move-Item -Path "$src\*Lambda*.h5p" -Destination "$dst\docs\lectures\" -Force
Move-Item -Path "$src\*gi?i thu?t*.h5p" -Destination "$dst\docs\lectures\" -Force

# Move docs/admin & images
Move-Item -Path "$src\lichthi_dotthi_1_l2_hk1_nh2025_thong_bao.xlsx" -Destination "$dst\docs\admin\" -Force
Move-Item -Path "$src\logo-uit.png" -Destination "$dst\docs\images\" -Force

# Move docs/projects
if (Test-Path "$src\BAO CAO") { Move-Item -Path "$src\BAO CAO" -Destination "$dst\docs\projects\" -Force }
if (Test-Path "$src\DOAN") { Move-Item -Path "$src\DOAN" -Destination "$dst\docs\projects\" -Force }

# Move notebooks
Move-Item -Path "$src\NLP_TH1.ipynb" -Destination "$dst\notebooks\" -Force
Move-Item -Path "$src\NLP_TH1_1.ipynb" -Destination "$dst\notebooks\" -Force
Move-Item -Path "$src\PIPELINE_NO_PREPROCESSING_CS221.ipynb" -Destination "$dst\notebooks\" -Force
Move-Item -Path "$src\TEST.ipynb" -Destination "$dst\notebooks\" -Force
if (Test-Path "$src\TH2") { Move-Item -Path "$src\TH2" -Destination "$dst\notebooks\" -Force }
if (Test-Path "$src\TH3") { Move-Item -Path "$src\TH3" -Destination "$dst\notebooks\" -Force }

# Move scripts
Move-Item -Path "$src\pipeline_debug copy.py" -Destination "$dst\scripts\" -Force
if (Test-Path "$src\demo") { Move-Item -Path "$src\demo" -Destination "$dst\scripts\" -Force }

# Move models & logs
Move-Item -Path "$src\model.zip" -Destination "$dst\models\" -Force
Move-Item -Path "$src\demo.log" -Destination "$dst\logs\" -Force

# Output result
Get-ChildItem -Path $src
