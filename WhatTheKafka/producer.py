import requests
import time
import json

from confluent_kafka import Producer

# p = Producer({'bootstrap.servers': 'mybroker1,mybroker2'})

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        pass


def main():
    # Replace STREAM_ENDPOINT with the URL of the streaming endpoint
    STREAM_ENDPOINT = 'http://localhost:5000/stream'

    # Make a GET request to the streaming endpoint with stream=True parameter to enable streaming
    config = {
        "security.protocol":"PLAINTEXT",
        "bootstrap.servers":"b-2.eidoscluster3.0d6zs2.c12.kafka.us-east-1.amazonaws.com:9092,b-1.eidoscluster3.0d6zs2.c12.kafka.us-east-1.amazonaws.com:9092,b-3.eidoscluster3.0d6zs2.c12.kafka.us-east-1.amazonaws.com:9092"
    }
    kafka_producer = Producer(config)

    response = requests.get(STREAM_ENDPOINT, stream=True)
    # Iterate over each line of the streaming response
    for line in response.iter_lines(chunk_size=512):

        # Skip empty lines
        if not line:
            continue

        kafka_producer.produce(topic="msk-demo-topic", value=line, callback=delivery_report)
        kafka_producer.poll(0)
    kafka_producer.flush()

if __name__ == "__main__":
    main()
