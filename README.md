# resume-reality-check

<!-- TODO: project description -->

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
uvicorn app.main:app --reload
```

## API

`POST /analyze` - multipart form with a `resume` file (PDF or text) and a
`job_description` string field. Returns a per-skill table scoring how well the
resume's evidence backs each required/preferred skill from the job description.
