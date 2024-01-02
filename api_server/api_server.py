from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/send-data', methods=['POST'])
def send_data():
    data = request.json
    # Send data to the data-processor container
    requests.post('http://data-processor:5001/process', json=data)
    return jsonify({"message": "Data sent to processor"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
