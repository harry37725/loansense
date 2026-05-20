import requests

response = requests.post('http://localhost:5000/predict', json={
    "revolving_util"   : 0.8,
    "age"              : 35,
    "late_30_59"       : 2,
    "debt_ratio"       : 0.6,
    "monthly_income"   : 4000,
    "open_credit_lines": 5,
    "late_90"          : 1,
    "real_estate_loans": 1,
    "late_60_89"       : 0,
    "dependents"       : 2
})

print(response.json())