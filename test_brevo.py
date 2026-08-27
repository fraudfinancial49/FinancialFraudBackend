import os
import requests
import sys

def test_brevo(api_key: str, sender_email: str, recipient_email: str):
    print(f"Testing Brevo API...")
    print(f"Sender: {sender_email}")
    print(f"Recipient: {recipient_email}")
    
    url = "https://api.brevo.com/v3/smtp/email"
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "sender": {
            "name": "PaySim Security Test",
            "email": sender_email,
        },
        "to": [{"email": recipient_email}],
        "subject": "Terminal Test - PaySim OTP",
        "textContent": "This is a test email sent directly from the terminal. Your OTP is 123456.",
    }
    
    try:
        print("\nSending request to Brevo...")
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Body:")
        print(response.text)
        
        if response.status_code in (200, 201):
            print("\n✅ SUCCESS! The email was accepted by Brevo.")
        else:
            print("\n❌ FAILED. Look at the response body above for the exact error.")
            
    except Exception as e:
        print(f"\n❌ FAILED with Python exception: {e}")

if __name__ == "__main__":
    api_key = input("Enter your Brevo API Key (xkeysib-...): ").strip()
    sender_email = input("Enter your Brevo Sender Email (e.g. fraudfinancial49@gmail.com): ").strip()
    recipient_email = input("Enter the destination email to send the test to: ").strip()
    
    print("-" * 40)
    test_brevo(api_key, sender_email, recipient_email)
