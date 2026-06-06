import sys
import json
from pathlib import Path
import asyncio

sys.path.insert(0, "pipeline_v2")

from pipeline_v2.stages.stage_4_render import Stage4Render
from pipeline_v2.models import StageTwoOutput, JobOutput

async def run_stage_4():
    job_dir = Path("output/full_video_shorts_v2/job_53")
    meta_path = job_dir / "editor_meta.json"
    
    # Actually, Stage4Render needs a StageTwoOutput object!
    # Does Stage 2 output exist?
    s2_path = job_dir / "stage2_output.json"
    
    # Let's check what files we have in job_53
    print("Files in job_53:")
    for f in job_dir.glob("*"):
        print(f.name)
        
run_stage_4()
