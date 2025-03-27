import requests
import os
import sys
import time

# Wait for the server to start
print("Waiting for the web server to start...")
time.sleep(3)

# Get the list of backtest results
try:
    results_response = requests.get('http://localhost:5000/get_backtest_results')
    results = results_response.json()
    
    if not results:
        print("No backtest results found.")
        sys.exit(1)
    
    # Get the first result file
    first_result = results[0]
    print(f"Testing summary view for: {first_result['name']}")
    
    # Request the summary view
    summary_url = f"http://localhost:5000/view_backtest_result/{first_result['name']}"
    summary_response = requests.get(summary_url)
    
    # Check if the summary section is present
    html_content = summary_response.text
    
    if "<h3>Backtest Summary</h3>" in html_content:
        print("✅ Summary section is present in the response")
    else:
        print("❌ Summary section is NOT present in the response")
    
    if "Win Rate" in html_content:
        print("✅ Win Rate metric is present")
    else:
        print("❌ Win Rate metric is NOT present")
    
    if "Profit Factor" in html_content:
        print("✅ Profit Factor metric is present")
    else:
        print("❌ Profit Factor metric is NOT present")
    
    if "Total Return" in html_content:
        print("✅ Total Return metric is present")
    else:
        print("❌ Total Return metric is NOT present")
    
    if "Initial Capital" in html_content:
        print("✅ Initial Capital metric is present")
    else:
        print("❌ Initial Capital metric is NOT present")
    
    if "Final Capital" in html_content:
        print("✅ Final Capital metric is present")
    else:
        print("❌ Final Capital metric is NOT present")
    
    if "Trades" in html_content and "winning" in html_content and "losing" in html_content:
        print("✅ Trade statistics are present")
    else:
        print("❌ Trade statistics are NOT present")
    
    # Save a sample of the HTML to a file for inspection
    with open("summary_view_sample.html", "w") as f:
        f.write(html_content[:5000])  # First 5000 characters
    
    print(f"\nSaved a sample of the HTML to summary_view_sample.html")
    print(f"To view the full summary, open a browser and go to: {summary_url}")
    
except Exception as e:
    print(f"Error testing summary view: {str(e)}")
