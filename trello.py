import requests
from datetime import date, timedelta

# Replace with your own values
API_KEY = 'your_api_key'
TOKEN = 'your_token'
BOARD_ID = 'your_board_id'
LIST_ID = 'your_list_id'

# Checklist items
checklist_items = ["Calls and Interactions", "Agile Stand-ups", "Project Work", "Others"]

# Generate all days of May
start_date = date(2025, 5, 1)
end_date = date(2025, 5, 31)
delta = timedelta(days=1)

current = start_date
while current <= end_date:
    card_name = current.strftime("May %d, %A")
    
    # Create the card
    card_url = f"https://api.trello.com/1/cards"
    card_query = {
        'key': API_KEY,
        'token': TOKEN,
        'idList': LIST_ID,
        'name': card_name
    }
    card_response = requests.post(card_url, params=card_query)
    card = card_response.json()
    
    # Create checklist
    checklist_url = f"https://api.trello.com/1/cards/{card['id']}/checklists"
    checklist_query = {
        'key': API_KEY,
        'token': TOKEN,
        'name': "Daily Tasks"
    }
    checklist_response = requests.post(checklist_url, params=checklist_query)
    checklist = checklist_response.json()

    # Add items to checklist
    for item in checklist_items:
        item_url = f"https://api.trello.com/1/checklists/{checklist['id']}/checkItems"
        item_query = {
            'key': API_KEY,
            'token': TOKEN,
            'name': item
        }
        requests.post(item_url, params=item_query)
    
    current += delta

print("All May task cards created successfully.")
