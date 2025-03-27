from flask import Flask, request, Response
import requests

app = Flask(__name__)

@app.route('/<path:url>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy(url):
    # Construct the target URL
    target_url = f"http://127.0.0.1:5000/{url}"
    
    # Forward the request to the target URL
    if request.method == 'GET':
        resp = requests.get(target_url, params=request.args)
    elif request.method == 'POST':
        resp = requests.post(target_url, json=request.get_json())
    elif request.method == 'PUT':
        resp = requests.put(target_url, json=request.get_json())
    elif request.method == 'DELETE':
        resp = requests.delete(target_url)
    elif request.method == 'OPTIONS':
        resp = requests.options(target_url)
    
    # Create a response with the same content and status code
    response = Response(resp.content, resp.status_code)
    
    # Add CORS headers
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    
    # Copy other headers from the original response
    for key, value in resp.headers.items():
        if key.lower() not in ['access-control-allow-origin', 'access-control-allow-headers', 'access-control-allow-methods']:
            response.headers[key] = value
    
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
