from flask import Flask, Response
import json
import time
from threading import Thread

app = Flask(__name__)

# Define a queue to store the generated data
data_queue = []
events_per_second = 1000

# Define a generator function that continuously yields data
def generate_data():
    start_time = time.time()
    counter = 0
    while True:
        timestamp_difference = int(time.time() - start_time)
        data = {'counter': counter, 'timestamp': timestamp_difference}
        data_queue.append(data)
        counter += 1
        time.sleep(1/events_per_second)  # Adjust the delay between each data update as per your needs

# Define a route to stream data from the queue
@app.route('/stream')
def stream():
    def generate():
        for data in data_queue:
            yield json.dumps(data) + '\n'
        
        while True:
            time.sleep(0.1)  # Adjust the delay to control the frequency of checking for new data
            if len(data_queue) > 0:
                for data in data_queue:
                    yield json.dumps(data) + '\n'
                data_queue.clear()

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    # Start the data generation in a separate thread
    data_generation_thread = Thread(target=generate_data)
    data_generation_thread.start()

    # Run the Flask app
    app.run()
