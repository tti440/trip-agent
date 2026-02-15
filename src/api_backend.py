from fastapi import FastAPI, HTTPException
from generation_pipeline import backend  # This triggers the 1-time initialization
import uvicorn
import os

app = FastAPI(title="Landmark Identification Service")

@app.get("/identify")
async def identify_landmark(image_path: str, text_input: str = ""):
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        # Calls the 'hot' models in RAM
        result, caption = backend.run_generation_pipeline(image_path, text_input=text_input)
        return {
            "candidates_raw": result, 
            "caption": caption,
            "text_input": text_input
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during landmark identification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)