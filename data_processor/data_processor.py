from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_data():
    data = request.json
    print("Data received:", data)
    # Add your data processing logic here
    return jsonify({"message": "Data processed"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
