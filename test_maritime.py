from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Maritime Survival Prediction API is running!"}
    
def test_predict_valid():

    payload = {

        "Age": 25,

        "Gender": "male",

        "BoardingPort": "S",

        "Title": "Mr",

        "TicketTier": 3,

        "TicketCost": 10,

        "FamilySize": 1,

        "FarePerPerson": 10,

        "RelativesAboard": 0,

        "ParentsChildren": 0,

        "CLass": "A123"

    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "survived" in data

    assert "survival_probability" in data    