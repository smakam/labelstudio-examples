# Label Studio Projects

Collection of Label Studio projects for various annotation tasks.

## Projects

- **audio-trans-dia**: Audio transcription and diarization
- **audio-trans-dia-qc**: Audio transcription QC with AssemblyAI
- **bengali-audio-trans-dia**: Bengali audio transcription and translation
- **bengali-audio-trans-dia-qc**: Bengali audio QC
- **images-face-det**: Image face detection
- **images-qc**: Image quality control with bounding boxes
- **imdb-sentiment**: IMDB sentiment analysis

## Structure

Each project typically contains:
- `config/` or `configs/`: Label Studio configuration files (`.xml`)
- `datasets/`: Input data files (CSV, JSON with task definitions)
- `results/`: Exported annotation results from Label Studio
- `scripts/`: Python scripts for data processing and integration

## Usage

1. Import tasks into Label Studio using the JSON files in `datasets/`
2. Use the configuration files in `config/` or `configs/` to set up your labeling interface
3. Export results using Label Studio API with `download_all_tasks=true` to include unannotated tasks

## Exporting All Tasks

To export all tasks including unannotated ones with predictions:

```bash
curl -X GET "http://localhost:8080/api/projects/{project_id}/export?exportType=JSON&download_all_tasks=true" \
     -H "Authorization: Token YOUR_API_TOKEN" \
     --output results.json
```

