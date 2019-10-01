from flask import Flask, jsonify
from flask import abort
from flask import request
from text_embedding import text_to_emb


app = Flask(__name__)

@app.route('/todo/api/v1.0/tasks', methods=['POST'])
def create_task():
    if not request.json or not 'text' in request.json:
        abort(400)
    task = {
        'text': request.json['text'],
        'embedding': str(text_to_emb(request.json['text']))
    }

    return jsonify({'task': task}), 201

if __name__ == '__main__':
    app.run(debug=True)